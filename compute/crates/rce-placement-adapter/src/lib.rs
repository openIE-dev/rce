//! `rce-placement-adapter` — production-shape qmm adapter that goes
//! through the **placement** layer (`rce-placement::pick_priced`):
//! every `qmm_1d(a, w)` call runs on the lowest-joule backend for its
//! op and shape.
//!
//! ## What the placement layer adds
//!
//! A fixed dispatcher picks ONE backend at construction time, then dispatches
//! every op through that backend regardless of shape. The placement layer
//! consults a `CostTable` per call and picks the lowest-joule backend
//! per-op-per-shape — the same primitive that lets the demo route matmul-128
//! to CPU and matmul-2048 to GPU on the same workload. For qmm the win-shape
//! is similar: small qmm's GPU dispatch overhead dominates; large qmm
//! amortizes it.
//!
//! ## Architecture seam
//!
//! `QuantWeight` doesn't fit the f32-tensor `Value` model cleanly (custom
//! dtype, custom block layout, custom dequant). Rather than coerce it
//! through `Value::Tensor(I8)` with side-band scalars, this crate splits
//! the trait surface:
//!
//! - **`PricedBackend`** (from `rce-vm-scheduler`) is implemented for
//!   *cost reasoning only*. `execute` returns a Stage-N error explaining
//!   that qmm dispatch goes through `QmmDispatch`. `pick_priced` selects
//!   the lowest-joule backend exactly as in the multi-op demo.
//! - **`QmmDispatch`** (this crate) is the typed qmm execute path. Each
//!   backend impls both; the adapter holds a parallel registry keyed by
//!   `PricedBackend.id()` so `pick_priced` decisions resolve to the right
//!   `QmmDispatch` instance for execution.
//!
//! This keeps `Value` clean (still just Scalar/Tensor/DeviceTensor) and
//! lets `rce-placement-adapter` add backends (CPU, future Metal-qmm,
//! future silicon) without touching the placement trait.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use rce_ingest_gguf::GgufDtype;
use rce_op_qmm::{mm_1d, QuantWeight};
use rce_op_qmm_metal::MetalQmmContext;
use rce_placement::{BackendId, CostTable, LinearCoeffs};
use rce_vm_backend_cpu::Value;
use rce_vm_scheduler::{pick_priced, PricedBackend};

pub mod trace;
pub use trace::{DispatchEvent, DispatchTranscript};

pub mod substrate_meter;
pub use substrate_meter::SubstrateEnergyMeter;

/// Plug-in energy meter for the self-calibration loop. The adapter
/// brackets each dispatch with `start` / `end`; if both return Some, the
/// observed picojoules feed into `CostTable::observe` to drift that
/// (op, backend) cell's coefficients toward reality.
///
/// Deliberately decoupled from `rce-energy` so the adapter stays
/// cross-platform: any meter — IOReport, RAPL, NVML, a simulated
/// fixture for tests — drops in via this trait.
pub trait EnergyMeter: Send + Sync {
    /// Begin a measurement window. Returns an opaque start token, or
    /// `None` if measurement is unavailable for this call. The token's
    /// only requirement is round-trip stability with `end`.
    fn start(&self) -> Option<u64>;

    /// End the measurement window opened by `start`. Returns observed
    /// energy in picojoules, or `None` if measurement failed.
    fn end(&self, start_token: u64) -> Option<u64>;
}

/// Stable wire IDs for the dtype slot in the qmm shape encoding. `0` is
/// "unknown/other" — format-specific backends (e.g. `MetalQmm`) return
/// `INFINITY` from `cost()` for unknown IDs so `pick_priced` skips them
/// and the generalist backend (CpuQmm) wins.
pub const QMM_DTYPE_UNKNOWN: usize = 0;
pub const QMM_DTYPE_Q8_0:    usize = 1;
pub const QMM_DTYPE_Q4_K:    usize = 2;
pub const QMM_DTYPE_Q5_K:    usize = 3;
pub const QMM_DTYPE_Q6_K:    usize = 4;
pub const QMM_DTYPE_Q4_0:    usize = 5;
pub const QMM_DTYPE_Q5_0:    usize = 6;
pub const QMM_DTYPE_Q4_1:    usize = 7;
pub const QMM_DTYPE_Q1_0:    usize = 8;
pub const QMM_DTYPE_Q2_0:    usize = 9;
pub const QMM_DTYPE_Q3_K:    usize = 10;

/// True iff `MetalQmm` has a native kernel for `id`. Adding a new
/// Metal kernel = add an arm here + a `qmm_dtype_id` mapping.
pub fn metal_qmm_supports(id: usize) -> bool {
    matches!(
        id,
        QMM_DTYPE_Q8_0 | QMM_DTYPE_Q4_K | QMM_DTYPE_Q5_K | QMM_DTYPE_Q6_K
        | QMM_DTYPE_Q4_0 | QMM_DTYPE_Q5_0 | QMM_DTYPE_Q4_1
        | QMM_DTYPE_Q1_0 | QMM_DTYPE_Q2_0
        | QMM_DTYPE_Q3_K
    )
}

/// Map a `GgufDtype` to its qmm dispatch ID. Formats with a Metal kernel
/// get a non-zero ID; everything else is `QMM_DTYPE_UNKNOWN` so only
/// the generalist backend (CpuQmm) handles them.
pub fn qmm_dtype_id(dtype: GgufDtype) -> usize {
    match dtype {
        GgufDtype::Q8_0 => QMM_DTYPE_Q8_0,
        GgufDtype::Q3K  => QMM_DTYPE_Q3_K,
        GgufDtype::Q4K  => QMM_DTYPE_Q4_K,
        GgufDtype::Q5K  => QMM_DTYPE_Q5_K,
        GgufDtype::Q6K  => QMM_DTYPE_Q6_K,
        GgufDtype::Q4_0 => QMM_DTYPE_Q4_0,
        GgufDtype::Q5_0 => QMM_DTYPE_Q5_0,
        GgufDtype::Q4_1 => QMM_DTYPE_Q4_1,
        GgufDtype::Q1_0 => QMM_DTYPE_Q1_0,
        GgufDtype::Q2_0 => QMM_DTYPE_Q2_0,
        _ => QMM_DTYPE_UNKNOWN,
    }
}

// ── Specialized qmm dispatch trait ────────────────────────────────

/// Typed `qmm_1d` execute path. Each priced qmm backend implements this in
/// addition to `PricedBackend`; the adapter selects via `pick_priced` (cost
/// reasoning over `PricedBackend`) then dispatches via `QmmDispatch`.
pub trait QmmDispatch: Send + Sync {
    /// Compute `out = w @ a` where `w: [n_rows, n_cols]`, `a: [n_cols]`,
    /// `out: [n_rows]`. Matches `rce_op_qmm::mm_1d` semantics.
    fn qmm_1d(&self, a: &[f32], w: &QuantWeight) -> Vec<f32>;
}

// ── Cost helpers ──────────────────────────────────────────────────

/// FLOPs and bytes-touched for `qmm_1d(a: [k], w: [n,k]) → [n]`. FLOPs
/// match dense matmul (2·n·k); bytes are the weight byte size + activation
/// + output. The quantization ratio (dtype-specific) is folded into the
/// per-byte coefficient in the `CostTable`, not the byte count.
pub fn qmm_1d_flops_bytes(n: u64, k: u64, weight_bytes: u64) -> (u64, u64) {
    let flops = 2 * n * k;
    let bytes = weight_bytes + 4 * k + 4 * n;
    (flops, bytes)
}

/// Build the default cost table for qmm dispatch. Analytical priors only —
/// a real calibration would run `rce-energy-xover` over each (dtype, n, k)
/// and emit measured cells via `CostTable::from_toml`.
///
/// Two backends are priced:
/// - `cpu:op-qmm` — direct `rce_op_qmm::mm_1d`. Low per-call overhead
///   (500 µJ), higher per-FLOP (10 pJ). Wins below the crossover.
/// - `metal:op-qmm` — native Metal Q8_0 (`rce-op-qmm-metal`). Higher
///   per-call overhead (2 mJ — buffer fill + commit + wait), low per-FLOP
///   (0.5 pJ). Wins above the crossover, for the formats it covers.
///
/// With these coefficients the crossover sits at roughly 158k FLOPs.
/// Below: small attention head matmuls, output projection on tiny
/// activations → CPU. Above: weight matmuls in transformer FFN/attn
/// layers → Metal where it is available.
pub fn default_qmm_cost_table() -> Arc<RwLock<CostTable>> {
    let mut t = CostTable::new(LinearCoeffs {
        a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
    });
    t.insert(
        "qmm",
        &BackendId::new("cpu:op-qmm"),
        LinearCoeffs { a_pj_per_flop: 10.0, b_pj_per_byte: 0.05, c_pj_overhead: 500_000.0 },
        false,
    );
    t.insert(
        "qmm",
        &BackendId::new("metal:op-qmm"),
        LinearCoeffs { a_pj_per_flop: 0.5, b_pj_per_byte: 0.05, c_pj_overhead: 2_000_000.0 },
        false,
    );
    Arc::new(RwLock::new(t))
}

// ── CPU qmm priced backend ────────────────────────────────────────

/// `PricedBackend` + `QmmDispatch` for the CPU `rce_op_qmm::mm_1d` path.
/// Handles every quantization the underlying `mm_1d` does (dispatches by
/// `QuantWeight::dtype` internally).
pub struct CpuQmm {
    table: Arc<RwLock<CostTable>>,
}

impl CpuQmm {
    pub fn new(table: Arc<RwLock<CostTable>>) -> Self { Self { table } }
}

impl PricedBackend for CpuQmm {
    fn id(&self) -> &str { "cpu:op-qmm" }
    fn handles(&self) -> &[&'static str] { &["qmm"] }

    fn cost(&self, op: &str, shapes: &[&[usize]]) -> (f64, bool) {
        if op != "qmm" || shapes.len() < 2 { return (f64::INFINITY, false); }
        // shapes: [a_shape=[k], w_shape=[n_rows, n_cols, weight_bytes_packed?]]
        // Use a 3-element w_shape so cost has access to the byte count
        // without dragging a side-band — see PlacementQmmAdapter::qmm_1d.
        let k = shapes[0][0] as u64;
        let n_rows = shapes[1][0] as u64;
        let weight_bytes = shapes[1].get(2).copied().unwrap_or(0) as u64;
        let (flops, bytes) = qmm_1d_flops_bytes(n_rows, k, weight_bytes);
        let est = self.table.read().unwrap().estimate(op, &BackendId::new(self.id().to_string()), flops, bytes);
        (est.picojoules, est.measured)
    }

    fn execute(&self, _op: &str, _inputs: &[Value]) -> Result<Value, String> {
        Err("CpuQmm: qmm dispatch uses QmmDispatch::qmm_1d, not PricedBackend::execute \
             (QuantWeight doesn't fit the Value::Tensor model)".into())
    }
}

impl QmmDispatch for CpuQmm {
    fn qmm_1d(&self, a: &[f32], w: &QuantWeight) -> Vec<f32> {
        let mut out = vec![0.0f32; w.n_rows];
        mm_1d(a, w, &mut out);
        out
    }
}

// ── Native Metal Q8_0 priced backend ──────────────────────────────

/// `PricedBackend` + `QmmDispatch` that dispatches Q8_0 directly through
/// `rce_op_qmm_metal::MetalQmmContext`, for the format the kernel covers.
///
/// **Selective handling**: `cost` returns `INFINITY` for any weight whose
/// `qmm_dtype_id` isn't `QMM_DTYPE_Q8_0`, so `pick_priced` never selects
/// MetalQmm for unsupported formats — `CpuQmm` wins those.
/// This is how the placement layer cleanly extends to format-specific
/// backends without adding a "can this op be done?" predicate to the
/// trait.
///
/// On non-Apple targets (or when Metal initialization fails), `new`
/// returns `None`; the adapter's `with_metal` constructor then registers
/// only the CPU backend, so cross-target builds work.
pub struct MetalQmm {
    ctx: Arc<MetalQmmContext>,
    table: Arc<RwLock<CostTable>>,
}

impl MetalQmm {
    pub fn new(table: Arc<RwLock<CostTable>>) -> Option<Self> {
        let ctx = Arc::new(MetalQmmContext::new()?);
        Some(Self { ctx, table })
    }
}

impl PricedBackend for MetalQmm {
    fn id(&self) -> &str { "metal:op-qmm" }
    fn handles(&self) -> &[&'static str] { &["qmm"] }

    fn cost(&self, op: &str, shapes: &[&[usize]]) -> (f64, bool) {
        if op != "qmm" || shapes.len() < 2 { return (f64::INFINITY, false); }
        // Reject unsupported formats — let pick_priced route them to a
        // generalist backend (CPU).
        let dtype_id = shapes[1].get(3).copied().unwrap_or(QMM_DTYPE_UNKNOWN);
        if !metal_qmm_supports(dtype_id) { return (f64::INFINITY, false); }

        let k = shapes[0][0] as u64;
        let n_rows = shapes[1][0] as u64;
        let weight_bytes = shapes[1].get(2).copied().unwrap_or(0) as u64;
        let (flops, bytes) = qmm_1d_flops_bytes(n_rows, k, weight_bytes);
        let est = self.table.read().unwrap().estimate(op, &BackendId::new(self.id().to_string()), flops, bytes);
        (est.picojoules, est.measured)
    }

    fn execute(&self, _op: &str, _inputs: &[Value]) -> Result<Value, String> {
        Err("MetalQmm: qmm dispatch uses QmmDispatch::qmm_1d, not PricedBackend::execute".into())
    }
}

impl QmmDispatch for MetalQmm {
    fn qmm_1d(&self, a: &[f32], w: &QuantWeight) -> Vec<f32> {
        // pick_priced rejected non-Q8_0 weights via cost()=INFINITY, so
        // this should always succeed in normal use. The unwrap surfaces
        // any shape/dtype contract violation loudly rather than silently
        // routing to the wrong backend.
        self.ctx.qmm_1d(a, w)
            .expect("MetalQmm dispatched on unsupported weight (cost should have rejected)")
    }
}

// ── Adapter stats ────────────────────────────────────────────────

/// Per-adapter observability — calls + estimated joules + per-backend
/// dispatch counts. The `measured` fraction distinguishes analytical
/// priors (default) from calibration-derived cells (after a
/// `CostTable::from_toml` load). The per-backend map proves the
/// shape-routed hybrid is actually flipping between backends.
#[derive(Default, Debug, Clone)]
pub struct PlacementAdapterStats {
    pub calls: u64,
    pub measured_calls: u64,
    /// Sum of estimated picojoules across all calls (uses each call's
    /// chosen backend's cost cell — what placement decided to spend).
    pub estimated_picojoules: f64,
    /// Per-backend dispatch count, keyed by `PricedBackend::id()`.
    pub calls_per_backend: HashMap<String, u64>,
    /// Times the dispatch cache had the (op, shape) decision pre-cached
    /// — `pick_priced` was skipped.
    pub dispatch_cache_hits: u64,
    /// Times `pick_priced` ran (cache miss). After warmup this should
    /// stop incrementing in workloads with bounded shape diversity.
    pub dispatch_cache_misses: u64,
}

impl PlacementAdapterStats {
    pub fn estimated_microjoules(&self) -> f64 {
        self.estimated_picojoules / 1.0e6
    }
    pub fn record(&mut self, backend_id: &str, pj: f64, measured: bool) {
        self.calls += 1;
        self.estimated_picojoules += pj;
        if measured { self.measured_calls += 1; }
        *self.calls_per_backend.entry(backend_id.to_string()).or_default() += 1;
    }
}

// ── The adapter ──────────────────────────────────────────────────

/// The qmm adapter: `qmm_1d(a, w)` selects the lowest-joule backend per
/// call via `pick_priced`.
///
/// **Verifiable receipts**: every dispatch is also appended to
/// `transcript` as a `DispatchEvent`. The transcript's FNV-1a digest is
/// a deterministic fold over the sequence of (op, shapes, backend,
/// estimated_pj) tuples — two honest replicas with the same model + cost
/// table produce the same digest, and any byzantine deviation (backend
/// swap, dropped event, dtype lie, cost-table drift) is caught at the
/// digest. Mirrors mesh-core's `transcript_digest` discipline.
pub struct PlacementQmmAdapter {
    backends: Vec<Arc<dyn PricedBackend>>,
    dispatchers: HashMap<String, Arc<dyn QmmDispatch>>,
    /// Shared cost table — the same `Arc<RwLock<CostTable>>` every
    /// backend reads for `cost()`. The adapter `.write()`s into it from
    /// `observe()` / `qmm_1d`'s metered path; backends `.read()` from
    /// it from the hot dispatch path.
    table: Arc<RwLock<CostTable>>,
    /// Optional pluggable energy meter — when `Some`, every `qmm_1d`
    /// dispatch is bracketed by `meter.start()` / `meter.end()` and the
    /// observed picojoules feed `CostTable::observe`. Adapter behaviour
    /// is unchanged when `None`: the priors-only path is identical to
    /// pre-self-calibration runs.
    meter: Option<Arc<dyn EnergyMeter>>,
    /// **Two-tier dispatch cache** (mesh-core's control-plane / replay-
    /// path boundary): keyed by `(op, n_rows, n_cols, dtype_id)`. First
    /// time a shape is seen, `pick_priced` runs and the chosen
    /// `backend_id` is installed. Every subsequent dispatch of the same
    /// shape is an O(1) `HashMap` lookup — no `pick_priced` overhead in
    /// the steady-state hot path.
    ///
    /// Cache decisions are *sticky*: calibration drift in the cost
    /// table changes future-shape decisions but not cached ones.
    /// Callers wanting fresh decisions after recalibration should call
    /// `clear_dispatch_cache()` to invalidate the table.
    dispatch_cache: HashMap<(&'static str, usize, usize, usize), String>,
    pub stats: PlacementAdapterStats,
    pub transcript: DispatchTranscript,
}

impl PlacementQmmAdapter {
    /// Build an adapter with the default CPU qmm backend + analytical
    /// priors. The shape-routed hybrid (CPU + native Metal) is
    /// [`Self::with_metal`].
    pub fn new() -> Self {
        let table = default_qmm_cost_table();
        let cpu = Arc::new(CpuQmm::new(table.clone()));
        let mut s = Self {
            backends: vec![],
            dispatchers: HashMap::new(),
            table: table.clone(),
            meter: None,
            stats: PlacementAdapterStats::default(),
            dispatch_cache: HashMap::new(),
            transcript: DispatchTranscript::new(),
        };
        s.register(cpu);
        s
    }

    /// **The hybrid**: CPU + native Metal Q8_0.
    /// `pick_priced` routes each call to the lowest-joule backend that
    /// handles the dtype:
    ///   * Q8_0 small  → cpu:op-qmm   (low overhead wins below ~158k FLOPs)
    ///   * Q8_0 large  → metal:op-qmm (native GPU)
    ///   * non-Q8_0    → cpu:op-qmm   (MetalQmm rejects via cost()=INFINITY)
    /// Registers only the CPU backend if Metal initialization fails (e.g.,
    /// non-Apple target or sandboxed CI).
    pub fn with_metal() -> Self {
        let table = default_qmm_cost_table();
        let cpu = Arc::new(CpuQmm::new(table.clone()));
        let mut s = Self {
            backends: vec![],
            dispatchers: HashMap::new(),
            table: table.clone(),
            meter: None,
            stats: PlacementAdapterStats::default(),
            dispatch_cache: HashMap::new(),
            transcript: DispatchTranscript::new(),
        };
        s.register(cpu);
        if let Some(metal) = MetalQmm::new(table.clone()) {
            s.register(Arc::new(metal));
        }
        s
    }

    /// Register a backend that implements both `PricedBackend` (cost) and
    /// `QmmDispatch` (execute). Concrete trait objects let the caller add
    /// e.g. a future `MetalQmm` without touching this crate.
    pub fn register<B: PricedBackend + QmmDispatch + 'static>(&mut self, backend: Arc<B>) {
        let id = backend.id().to_string();
        self.backends.push(backend.clone() as Arc<dyn PricedBackend>);
        self.dispatchers.insert(id, backend as Arc<dyn QmmDispatch>);
    }

    /// Picks the lowest-joule
    /// backend via `pick_priced`, records the estimate in `stats`, then
    /// dispatches.
    ///
    /// `w_shape` carries 4 slots: `[n_rows, n_cols, weight_bytes,
    /// dtype_id]`. Generalist backends (Cpu) read only the first
    /// three; format-specific backends (e.g. MetalQmm) consult `dtype_id`
    /// and return `INFINITY` from `cost` for formats they don't cover.
    pub fn qmm_1d(&mut self, a: &[f32], w: &QuantWeight) -> Vec<f32> {
        debug_assert_eq!(a.len(), w.n_cols);
        let weight_bytes = w.bytes().len();
        let dtype_id = qmm_dtype_id(w.dtype);
        let a_shape = [a.len()];
        let w_shape = [w.n_rows, w.n_cols, weight_bytes, dtype_id];

        // Two-tier dispatch: check the shape cache first; on hit, skip
        // pick_priced entirely and re-derive the recorded (pj, measured)
        // from the table directly. Cache key intentionally excludes
        // weight_bytes so different weights of the same shape share an
        // entry (typical for transformer layers — same shape, many
        // weights). Cost-table values are still read live so calibration
        // EWMA on the chosen cell affects the receipt.
        let cache_key = ("qmm", w.n_rows, w.n_cols, dtype_id);
        let (be_id, pj, measured): (String, f64, bool) =
            if let Some(cached_id) = self.dispatch_cache.get(&cache_key).cloned() {
                self.stats.dispatch_cache_hits += 1;
                let (flops, bytes) = qmm_1d_flops_bytes(
                    w.n_rows as u64, w.n_cols as u64, weight_bytes as u64);
                let est = self.table.read().unwrap().estimate(
                    "qmm", &BackendId::new(cached_id.clone()), flops, bytes);
                (cached_id, est.picojoules, est.measured)
            } else {
                self.stats.dispatch_cache_misses += 1;
                let shapes: Vec<&[usize]> = vec![&a_shape, &w_shape];
                let (be, pj, measured) = pick_priced(&self.backends, "qmm", &shapes)
                    .expect("rce-placement-adapter: no priced backend handles `qmm`");
                let id = be.id().to_string();
                self.dispatch_cache.insert(cache_key, id.clone());
                (id, pj, measured)
            };
        self.stats.record(&be_id, pj, measured);

        // Record the dispatch as an event in the verifiable trace BEFORE
        // executing. Two honest replicas with the same model + cost
        // table append the same event; the transcript digest catches
        // any byzantine swap, drop, or cost-drift.
        self.transcript.append(trace::make_event(
            "qmm",
            &a_shape,
            &w_shape,
            &be_id,
            pj,
            measured,
        ));

        let dispatcher = self.dispatchers.get(&be_id)
            .expect("dispatcher missing for selected backend");

        // Self-calibration loop: bracket the dispatch with the energy
        // meter when configured. Bad readings (None at either end)
        // silently skip the cost-table update; good readings EWMA the
        // (op, backend) cell's coefficients toward the observed pj.
        let meter_token = self.meter.as_ref().and_then(|m| m.start());
        let result = dispatcher.qmm_1d(a, w);
        if let Some(token) = meter_token {
            if let Some(observed_pj) = self.meter.as_ref().and_then(|m| m.end(token)) {
                let (flops, bytes) = qmm_1d_flops_bytes(
                    w.n_rows as u64, w.n_cols as u64, weight_bytes as u64);
                self.table.write().unwrap().observe(
                    "qmm",
                    &BackendId::new(be_id),
                    flops,
                    bytes,
                    observed_pj as f64,
                );
            }
        }
        result
    }

    /// Install an energy meter for the self-calibration loop. Returns
    /// `self` so the call chains in builder style:
    /// `PlacementQmmAdapter::with_metal().with_meter(my_meter)`.
    pub fn with_meter(mut self, meter: Arc<dyn EnergyMeter>) -> Self {
        self.meter = Some(meter);
        self
    }

    /// Install the production rce-energy meter — macOS IOReport
    /// on Apple Silicon, Linux RAPL on x86 / ARM64 Linux, universal
    /// TDP fallback elsewhere. This is the **production wire-up of
    /// the self-calibration loop**: every dispatch is bracketed by a
    /// real hardware energy snapshot pair; observed pJ feeds
    /// `CostTable::observe`; after `measured_after` samples per cell
    /// the `measured` flag flips and downstream consumers see
    /// calibration-derived coefficients.
    ///
    /// Falls back to `self` (no meter) if `SubstrateEnergyMeter`
    /// can't open a backend (in practice this never happens —
    /// `fallback_tdp` is universal).
    pub fn with_rce_energy_meter(self) -> Self {
        match SubstrateEnergyMeter::open_default() {
            Some(meter) => self.with_meter(Arc::new(meter)),
            None => self,
        }
    }

    /// Borrow the shared cost table. Useful for tests, for snapshotting
    /// the table state after a calibration run, or for asserting on
    /// `observations_for(op, backend)` to verify the loop is closed.
    pub fn cost_table(&self) -> Arc<RwLock<CostTable>> {
        self.table.clone()
    }

    /// Drop every cached `(op, shape) → backend_id` entry. The next
    /// dispatch of any shape re-runs `pick_priced` against the current
    /// cost-table coefficients. Call this after substantial
    /// recalibration to let EWMA-drifted cells affect routing decisions.
    pub fn clear_dispatch_cache(&mut self) {
        self.dispatch_cache.clear();
    }

    /// Count of distinct shapes in the dispatch cache. Useful for tests
    /// asserting "warmup converged" — for a model with a known set of
    /// weight shapes, the cache size should stabilise at that count.
    pub fn dispatch_cache_size(&self) -> usize {
        self.dispatch_cache.len()
    }

    /// FNV-1a digest of the full dispatch transcript. **The verifiable
    /// energy receipt**: a pure function of the dispatch sequence, so two
    /// honest replicas with the same model + cost table produce the same
    /// digest. A replica that deviated (different backend, dropped event,
    /// drift) is caught at the digest with no need to compare outputs.
    pub fn transcript_digest(&self) -> u64 {
        self.transcript.digest()
    }

    /// IDs of all registered backends, in the order placement sees them.
    pub fn backend_ids(&self) -> Vec<String> {
        self.backends.iter().map(|b| b.id().to_string()).collect()
    }

    // ── verity-cascade integration handshake ───────────────────────
    //
    // The verity-cascade engine (`verity_cascade::CascadeConfig.max_energy_uj`)
    // enforces a per-query energy budget by deciding which L0..L10 layers
    // to traverse. L3 is the LLM-inference layer; when an L3 runtime
    // (tinyllama / qwen3 / granite / bitnet with `--features placement`)
    // dispatches through `PlacementQmmAdapter::qmm_1d`, every call adds
    // to `stats.estimated_picojoules`. The handshake the cascade needs:
    //
    //   1. Reset the adapter's session accumulators at L3 entry.
    //   2. Check `over_budget(remaining_uj)` after each forward (or each
    //      qmm, for tighter loops) and short-circuit if exceeded.
    //   3. On L3 exit, read `stats.estimated_microjoules()` and
    //      `stats.measured_calls` to populate the L3 portion of the
    //      `CascadeEnergyReceipt`.
    //
    // The cascade-side patch (modifying verity-cascade's L3 layer) is
    // its own commit — it depends on the L3 architecture beyond what's
    // in scope here. The methods below are the adapter-side handshake
    // surface that patch will call.

    /// Reset session-scoped accumulators (`stats`, `transcript`) without
    /// touching the `cost_table`, `dispatch_cache`, or registered
    /// backends. Intended to be called at the start of each new
    /// cascade query so per-call receipts attribute correctly to the
    /// owning query.
    pub fn reset_session(&mut self) {
        self.stats = PlacementAdapterStats::default();
        self.transcript = DispatchTranscript::new();
    }

    /// `true` iff the cumulative estimated energy since the last
    /// `reset_session()` (or adapter creation) exceeds `budget_uj`.
    /// The cascade checks this after each forward / qmm to short-
    /// circuit before paying the next layer's cost.
    pub fn over_budget(&self, budget_uj: u64) -> bool {
        self.stats.estimated_microjoules() > budget_uj as f64
    }

    /// Microjoules of estimated headroom under `budget_uj`. Negative
    /// values mean over-budget; the cascade typically treats anything
    /// less than the smallest priced-op cost as "no headroom for one
    /// more dispatch." Saturates at `i64::MAX` for budget = 0 (the
    /// cascade's "unlimited" sentinel).
    pub fn remaining_budget_uj(&self, budget_uj: u64) -> i64 {
        if budget_uj == 0 { return i64::MAX; }
        let spent = self.stats.estimated_microjoules();
        (budget_uj as f64 - spent).round() as i64
    }
}

impl Default for PlacementQmmAdapter {
    fn default() -> Self { Self::new() }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rce_ingest_gguf::GgufDtype;

    /// Synthetic Q8_0 weight on the heap.
    fn synthetic_q8(n_rows: usize, n_cols: usize) -> QuantWeight {
        assert_eq!(n_cols % 32, 0, "Q8_0 needs n_cols % 32 == 0");
        let blocks_per_row = n_cols / 32;
        let bytes_per_row = blocks_per_row * 34;
        let mut bytes = vec![0u8; n_rows * bytes_per_row];
        for r in 0..n_rows {
            for b in 0..blocks_per_row {
                let off = r * bytes_per_row + b * 34;
                bytes[off] = 0x00;
                bytes[off + 1] = 0x3C; // f16 1.0 LE
                for i in 0..32 {
                    let v = ((r + b * 32 + i) as i32 % 7 - 3) as i8;
                    bytes[off + 2 + i] = v as u8;
                }
            }
        }
        QuantWeight::from_owned_bytes(bytes, GgufDtype::Q8_0, n_rows, n_cols)
    }

    #[test]
    fn qmm_1d_matches_direct_reference() {
        let mut adapter = PlacementQmmAdapter::new();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.03).sin()).collect();

        let out_adapter = adapter.qmm_1d(&a, &w);
        let mut out_ref = vec![0.0f32; w.n_rows];
        mm_1d(&a, &w, &mut out_ref);

        assert_eq!(out_adapter.len(), out_ref.len());
        for (i, (m, d)) in out_adapter.iter().zip(out_ref.iter()).enumerate() {
            assert_eq!(m, d, "out[{i}] diverged: placement={m}, reference={d}");
        }
    }

    #[test]
    fn second_call_stats_accumulate() {
        let mut adapter = PlacementQmmAdapter::new();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).cos()).collect();

        let _ = adapter.qmm_1d(&a, &w);
        let after_one = adapter.stats.clone();
        assert_eq!(after_one.calls, 1);
        assert!(after_one.estimated_picojoules > 0.0,
            "first call should record a positive pJ estimate");

        let _ = adapter.qmm_1d(&a, &w);
        let after_two = adapter.stats.clone();
        assert_eq!(after_two.calls, 2);
        assert!(after_two.estimated_picojoules > after_one.estimated_picojoules,
            "second call should add more pJ to the running total");

        // Analytical priors — `measured` defaults to false.
        assert_eq!(after_two.measured_calls, 0,
            "default cost table is analytical (measured=false); calibration via from_toml flips this");
    }

    #[test]
    fn picks_cheaper_backend_when_multiple_handle_qmm() {
        // Register a second backend with a deliberately *cheaper* cost
        // cell so pick_priced selects it; verify dispatch ran on the
        // chosen one by checking output shape (the fixture echoes a
        // zero vector of length n_rows).
        struct ZeroQmm { table: Arc<RwLock<CostTable>> }
        impl PricedBackend for ZeroQmm {
            fn id(&self) -> &str { "test:zero" }
            fn handles(&self) -> &[&'static str] { &["qmm"] }
            fn cost(&self, _op: &str, shapes: &[&[usize]]) -> (f64, bool) {
                let est = self.table.read().unwrap().estimate(
                    "qmm", &BackendId::new("test:zero".to_string()),
                    2 * shapes[1][0] as u64 * shapes[0][0] as u64,
                    shapes[1].get(2).copied().unwrap_or(0) as u64,
                );
                (est.picojoules, est.measured)
            }
            fn execute(&self, _op: &str, _inputs: &[Value]) -> Result<Value, String> {
                Err("unused".into())
            }
        }
        impl QmmDispatch for ZeroQmm {
            fn qmm_1d(&self, _a: &[f32], w: &QuantWeight) -> Vec<f32> {
                vec![-7.0; w.n_rows] // sentinel — distinguishable from CPU output
            }
        }

        let mut t = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        });
        t.insert(
            "qmm", &BackendId::new("test:zero"),
            LinearCoeffs { a_pj_per_flop: 0.001, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 },
            true,
        );
        let cheap_table = Arc::new(RwLock::new(t));

        let mut adapter = PlacementQmmAdapter::new();
        adapter.register(Arc::new(ZeroQmm { table: cheap_table }));

        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.04).sin()).collect();
        let out = adapter.qmm_1d(&a, &w);

        assert_eq!(out, vec![-7.0; 16],
            "cheaper backend should have won; output sentinel matches ZeroQmm");
    }

    // ── Hybrid: native Metal Q8_0 + CPU ──────────────────────────────

    #[test]
    fn hybrid_routes_small_to_cpu_large_to_metal() {
        let mut adapter = PlacementQmmAdapter::with_metal();
        let ids = adapter.backend_ids();
        assert!(ids.contains(&"cpu:op-qmm".to_string()));
        if !ids.contains(&"metal:op-qmm".to_string()) {
            eprintln!("Metal unavailable in this environment — skipping hybrid test");
            return;
        }

        // Small Q8_0: 16×32 ⇒ 2·n·k = 1024 FLOPs ≪ 158k crossover → CPU.
        let w_small = synthetic_q8(16, 32);
        let a_small: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
        let out_small = adapter.qmm_1d(&a_small, &w_small);
        let mut ref_small = vec![0.0f32; w_small.n_rows];
        mm_1d(&a_small, &w_small, &mut ref_small);
        assert_eq!(out_small, ref_small, "small Q8_0 routing should be bit-exact (CPU path)");

        // Large Q8_0: 1024×512 ⇒ 2·n·k = ~1M FLOPs ≫ crossover → Metal.
        let w_large = synthetic_q8(1024, 512);
        let a_large: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();
        let out_large = adapter.qmm_1d(&a_large, &w_large);
        let mut ref_large = vec![0.0f32; w_large.n_rows];
        mm_1d(&a_large, &w_large, &mut ref_large);
        // Metal path is bit-exact against mm_1d (validated in
        // rce-op-qmm-metal's own parity tests up to f32 reduction).
        for (i, (g, r)) in out_large.iter().zip(ref_large.iter()).enumerate() {
            let abs = (g - r).abs();
            let denom = r.abs().max(1e-6);
            let rel = abs / denom;
            assert!(abs < 1e-4 || rel < 1e-3,
                "large Q8_0 out[{i}]: metal={g}, ref={r}, abs={abs:.3e}, rel={rel:.3e}");
        }

        let cpu_calls    = adapter.stats.calls_per_backend.get("cpu:op-qmm").copied().unwrap_or(0);
        let metal_calls  = adapter.stats.calls_per_backend.get("metal:op-qmm").copied().unwrap_or(0);
        assert_eq!(cpu_calls,    1, "small Q8_0 should route to CPU; per-backend: {:?}",
            adapter.stats.calls_per_backend);
        assert_eq!(metal_calls,  1, "large Q8_0 should route to native Metal; per-backend: {:?}",
            adapter.stats.calls_per_backend);
    }

    // ── Verifiable dispatch trace (mesh-core's transcript_digest port) ─

    #[test]
    fn two_honest_adapters_produce_the_same_transcript_digest() {
        // The single most load-bearing property: two replicas running
        // the same model with the same cost table produce bit-identical
        // transcript digests. That's what makes RCE's energy receipts
        // verifiable instead of just measurable.
        let run = || -> u64 {
            let mut a = PlacementQmmAdapter::with_metal();
            let w_small = synthetic_q8(16, 32);
            let w_large = synthetic_q8(1024, 512);
            let a_small: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
            let a_large: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();
            let _ = a.qmm_1d(&a_small, &w_small);
            let _ = a.qmm_1d(&a_large, &w_large);
            let _ = a.qmm_1d(&a_small, &w_small);
            a.transcript_digest()
        };
        let d1 = run();
        let d2 = run();
        assert_eq!(d1, d2, "honest replicas with same workload agree on transcript digest");
    }

    #[test]
    fn transcript_records_one_event_per_qmm_call() {
        let mut adapter = PlacementQmmAdapter::with_metal();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
        for _ in 0..5 { let _ = adapter.qmm_1d(&a, &w); }
        assert_eq!(adapter.transcript.len(), 5);
        assert_eq!(adapter.transcript.len() as u64, adapter.stats.calls,
            "transcript length must match the stats call count exactly");
    }

    #[test]
    fn byzantine_backend_choice_is_caught_at_the_digest() {
        // Build two adapters that see the same workload but resolve to
        // different backends — simulated by hand-crafted transcripts
        // since the live adapter is honest. The test proves: if a
        // byzantine replica reported a different backend for the same
        // dispatch, its transcript digest would diverge from honest.
        use trace::{make_event, DispatchTranscript};
        let mut honest = DispatchTranscript::new();
        honest.append(make_event("qmm", &[32], &[16, 32, 544, 1], "metal:op-qmm", 1.0e6, false));
        let mut byzantine = DispatchTranscript::new();
        byzantine.append(make_event("qmm", &[32], &[16, 32, 544, 1], "cpu:op-qmm",   1.0e6, false));
        assert_ne!(honest.digest(), byzantine.digest(),
            "swapping the backend on the same dispatch must diverge");
    }

    // ── Self-calibration loop (EnergyMeter + CostTable::observe) ────

    /// A test meter that returns a caller-controlled picojoule count
    /// per dispatch — lets us assert calibration converges to a known
    /// value without depending on real IOReport/RAPL/NVML.
    struct FakeMeter {
        pj_per_call: u64,
        token_counter: std::sync::Mutex<u64>,
    }
    impl FakeMeter {
        fn new(pj_per_call: u64) -> Self {
            Self { pj_per_call, token_counter: std::sync::Mutex::new(0) }
        }
    }
    impl EnergyMeter for FakeMeter {
        fn start(&self) -> Option<u64> {
            let mut c = self.token_counter.lock().unwrap();
            *c += 1;
            Some(*c)
        }
        fn end(&self, _start: u64) -> Option<u64> {
            Some(self.pj_per_call)
        }
    }

    #[test]
    fn meter_observations_flip_cost_cell_to_measured() {
        // The headline self-calibration property: enough metered
        // observations of a (op, backend) cell flip its `measured` flag
        // to true. After that, downstream consumers can distinguish
        // calibration-derived cells from analytical priors.
        let mut adapter = PlacementQmmAdapter::new()
            .with_meter(Arc::new(FakeMeter::new(5_000_000))); // 5 µJ per call
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();

        // Default measured_after = 20. Run that many calls + 1 to cross.
        for _ in 0..21 { let _ = adapter.qmm_1d(&a, &w); }
        let table = adapter.cost_table();
        let observations = table.read().unwrap()
            .observations_for("qmm", &BackendId::new("cpu:op-qmm"));
        assert_eq!(observations, 21);

        let (_, measured) = table.read().unwrap()
            .lookup("qmm", &BackendId::new("cpu:op-qmm"));
        assert!(measured, "≥20 observations should flip the cost cell to measured");
    }

    #[test]
    fn no_meter_means_no_observation() {
        // Adapter behaviour MUST be unchanged when meter is None — the
        // priors-only path is byte-identical to pre-self-calibration runs.
        let mut adapter = PlacementQmmAdapter::new();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
        for _ in 0..30 { let _ = adapter.qmm_1d(&a, &w); }
        let table = adapter.cost_table();
        let n = table.read().unwrap()
            .observations_for("qmm", &BackendId::new("cpu:op-qmm"));
        assert_eq!(n, 0, "no meter ⇒ no observations recorded");
        let (_, measured) = table.read().unwrap()
            .lookup("qmm", &BackendId::new("cpu:op-qmm"));
        assert!(!measured, "without observations the cell stays analytical");
    }

    #[test]
    fn calibration_drives_estimated_pj_toward_observed_value() {
        // After many metered calls, the estimated_picojoules the adapter
        // reports per dispatch should converge toward the actual
        // observed value — the EWMA's whole point.
        let observed_pj: u64 = 7_500_000; // 7.5 µJ per call
        let mut adapter = PlacementQmmAdapter::new()
            .with_meter(Arc::new(FakeMeter::new(observed_pj)));
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();

        // Warm up with many observations.
        for _ in 0..100 { let _ = adapter.qmm_1d(&a, &w); }

        // After warmup, the next call's recorded estimate should be
        // within 10% of the observed value (EWMA convergence).
        let stats_before = adapter.stats.clone();
        let _ = adapter.qmm_1d(&a, &w);
        let dispatch_pj = adapter.stats.estimated_picojoules - stats_before.estimated_picojoules;

        let observed = observed_pj as f64;
        let ratio = dispatch_pj / observed;
        assert!(ratio > 0.9 && ratio < 1.1,
            "after 100 metered calls of {observed} pJ, next estimate should be within ±10%; \
             got {dispatch_pj} pJ (ratio={ratio:.3})");
    }

    // ── Two-tier dispatch cache (warmup → hot-path lookup) ──────────

    #[test]
    fn first_dispatch_of_a_shape_misses_subsequent_hit() {
        let mut adapter = PlacementQmmAdapter::with_metal();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
        assert_eq!(adapter.dispatch_cache_size(), 0);

        // First call: pick_priced runs, decision cached.
        let _ = adapter.qmm_1d(&a, &w);
        assert_eq!(adapter.stats.dispatch_cache_misses, 1);
        assert_eq!(adapter.stats.dispatch_cache_hits, 0);
        assert_eq!(adapter.dispatch_cache_size(), 1);

        // Next 4 calls: cache hit, pick_priced never runs again.
        for _ in 0..4 { let _ = adapter.qmm_1d(&a, &w); }
        assert_eq!(adapter.stats.dispatch_cache_misses, 1,
            "pick_priced should run exactly once per distinct shape");
        assert_eq!(adapter.stats.dispatch_cache_hits, 4);
        assert_eq!(adapter.dispatch_cache_size(), 1);
    }

    #[test]
    fn distinct_shapes_each_take_one_miss_then_hit() {
        // TinyLlama-shaped workload (different n_rows per layer): the
        // cache should converge to exactly the count of distinct shapes
        // after the first pass.
        let mut adapter = PlacementQmmAdapter::with_metal();
        let shapes = [(16, 32), (32, 32), (16, 64), (32, 64)];
        let weights: Vec<QuantWeight> = shapes.iter()
            .map(|&(r, c)| synthetic_q8(r, c)).collect();
        let activations: Vec<Vec<f32>> = shapes.iter()
            .map(|&(_, c)| (0..c).map(|i| (i as f32 * 0.01).sin()).collect()).collect();

        // First pass: 4 misses.
        for (w_i, w) in weights.iter().enumerate() {
            let _ = adapter.qmm_1d(&activations[w_i], w);
        }
        assert_eq!(adapter.stats.dispatch_cache_misses, 4);
        assert_eq!(adapter.dispatch_cache_size(), 4);

        // Second pass: all 4 hit.
        for (w_i, w) in weights.iter().enumerate() {
            let _ = adapter.qmm_1d(&activations[w_i], w);
        }
        assert_eq!(adapter.stats.dispatch_cache_misses, 4, "no new misses on the second pass");
        assert_eq!(adapter.stats.dispatch_cache_hits, 4);
    }

    #[test]
    fn clear_dispatch_cache_forces_pick_priced_to_re_run() {
        // After recalibration drifts the cost table, callers can
        // invalidate cached decisions so they're recomputed against
        // the updated coefficients.
        let mut adapter = PlacementQmmAdapter::with_metal();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();

        let _ = adapter.qmm_1d(&a, &w);
        let _ = adapter.qmm_1d(&a, &w);
        assert_eq!(adapter.stats.dispatch_cache_misses, 1);
        assert_eq!(adapter.dispatch_cache_size(), 1);

        adapter.clear_dispatch_cache();
        assert_eq!(adapter.dispatch_cache_size(), 0);

        let _ = adapter.qmm_1d(&a, &w);
        assert_eq!(adapter.stats.dispatch_cache_misses, 2,
            "clear should force pick_priced to run again on next dispatch");
    }

    #[test]
    fn cached_decision_still_uses_live_cost_table_for_receipt() {
        // The cache stores the *decision* (backend_id), not the cost
        // estimate. The receipt's `estimated_picojoules` is always read
        // live from the current cost table — so calibration EWMA still
        // affects the receipt even when the decision is cached.
        let observed_pj: u64 = 7_500_000;
        let mut adapter = PlacementQmmAdapter::new()
            .with_meter(Arc::new(FakeMeter::new(observed_pj)));
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();

        // Prime the cache once (first call is a miss).
        let _ = adapter.qmm_1d(&a, &w);
        let before = adapter.stats.estimated_picojoules;

        // Warm up calibration with many metered calls — all cache hits,
        // but each one still observes the meter and updates the cost
        // table. After the EWMA settles, the *next* estimated_pj should
        // be close to observed_pj.
        for _ in 0..100 { let _ = adapter.qmm_1d(&a, &w); }
        let stats_before_last = adapter.stats.clone();
        let _ = adapter.qmm_1d(&a, &w);
        let last_dispatch_pj = adapter.stats.estimated_picojoules
            - stats_before_last.estimated_picojoules;

        let _ = before;
        assert!(adapter.stats.dispatch_cache_hits >= 100,
            "should be all-hits after the first call");
        let ratio = last_dispatch_pj / observed_pj as f64;
        assert!(ratio > 0.85 && ratio < 1.15,
            "cached-decision path must still see calibration drift; \
             got {last_dispatch_pj} pJ (ratio {ratio:.3} vs observed {observed_pj})");
    }

    // ── Federation: cross-replica transcript-digest verification ────

    /// Three "honest" workers each run the same workload; their
    /// transcript digests are reported to a shared Federation. The
    /// consensus is reached on the agreed digest. A fourth byzantine
    /// worker (one whose internal state diverged — simulated here by
    /// running a different workload) is caught by digest disagreement.
    ///
    /// This is the federation-scale extension of the single-worker
    /// trust primitive: a single worker proves its claim by replay
    /// against its own transcript digest; a fleet proves agreement by
    /// comparing transcript digests across workers.
    #[test]
    fn federation_detects_byzantine_worker_via_transcript_digest() {
        use rce_federation::Federation;

        let run_honest_workload = |adapter: &mut PlacementQmmAdapter| {
            let w_small = synthetic_q8(16, 32);
            let w_large = synthetic_q8(1024, 512);
            let a_small: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
            let a_large: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();
            let _ = adapter.qmm_1d(&a_small, &w_small);
            let _ = adapter.qmm_1d(&a_large, &w_large);
            let _ = adapter.qmm_1d(&a_small, &w_small);
        };
        let run_byzantine_workload = |adapter: &mut PlacementQmmAdapter| {
            // Same call count, different SHAPE on the third call ⇒
            // dispatch trace diverges from the honest sequence.
            let w_small = synthetic_q8(16, 32);
            let w_large = synthetic_q8(1024, 512);
            let a_small: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
            let a_large: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();
            let _ = adapter.qmm_1d(&a_small, &w_small);
            let _ = adapter.qmm_1d(&a_large, &w_large);
            let _ = adapter.qmm_1d(&a_large, &w_large); // Different from honest
        };

        // 3 honest workers + 1 byzantine, all members of the federation.
        let mut a1 = PlacementQmmAdapter::with_metal();
        let mut a2 = PlacementQmmAdapter::with_metal();
        let mut a3 = PlacementQmmAdapter::with_metal();
        let mut a4 = PlacementQmmAdapter::with_metal();
        run_honest_workload(&mut a1);
        run_honest_workload(&mut a2);
        run_honest_workload(&mut a3);
        run_byzantine_workload(&mut a4);

        let mut fed = Federation::new();
        for w in [1u64, 2, 3, 4] { fed.members.admit(w, w * 0x10); }
        fed.report(1, a1.transcript_digest());
        fed.report(2, a2.transcript_digest());
        fed.report(3, a3.transcript_digest());
        fed.report(4, a4.transcript_digest());

        assert!(fed.consensus().is_none(),
            "byzantine worker should break consensus");
        let bad = fed.disagreements();
        assert_eq!(bad.len(), 1, "exactly one byzantine should be identified");
        assert_eq!(bad[0].0, 4,
            "worker 4 should be the identified byzantine (id, digest) = {:?}", bad);
    }

    #[test]
    fn federation_consensus_when_all_workers_run_same_workload() {
        // The honest-case property: 4 workers with identical workloads
        // produce identical digests, and the federation reports
        // consensus.
        use rce_federation::Federation;
        let workload = |adapter: &mut PlacementQmmAdapter| {
            let w = synthetic_q8(16, 32);
            let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();
            for _ in 0..5 { let _ = adapter.qmm_1d(&a, &w); }
        };
        let mut adapters: Vec<PlacementQmmAdapter> =
            (0..4).map(|_| PlacementQmmAdapter::with_metal()).collect();
        for a in &mut adapters { workload(a); }

        let mut fed = Federation::new();
        for (i, a) in adapters.iter().enumerate() {
            let w_id = (i + 1) as u64;
            fed.members.admit(w_id, w_id * 0x10);
            fed.report(w_id, a.transcript_digest());
        }

        let consensus = fed.consensus().expect("4 honest replicas should agree");
        assert_eq!(consensus, adapters[0].transcript_digest());
        assert!(fed.disagreements().is_empty());
    }

    // ── Real hardware meter drives the calibration loop ───────────

    // ── verity-cascade integration handshake ───────────────────────

    #[test]
    fn reset_session_clears_stats_and_transcript_but_keeps_cost_table() {
        let mut adapter = PlacementQmmAdapter::with_metal();
        let w = synthetic_q8(16, 32);
        let a: Vec<f32> = (0..32).map(|i| (i as f32 * 0.01).sin()).collect();

        // Build up some state across a few calls.
        for _ in 0..5 { let _ = adapter.qmm_1d(&a, &w); }
        let before_table = adapter.cost_table();
        let before_cells = before_table.read().unwrap()
            .lookup("qmm", &BackendId::new("cpu:op-qmm"));
        assert_eq!(adapter.stats.calls, 5);
        assert_eq!(adapter.transcript.len(), 5);
        assert!(adapter.stats.estimated_picojoules > 0.0);

        adapter.reset_session();

        // Stats + transcript zeroed.
        assert_eq!(adapter.stats.calls, 0);
        assert_eq!(adapter.transcript.len(), 0);
        assert_eq!(adapter.stats.estimated_picojoules, 0.0);
        assert!(adapter.stats.calls_per_backend.is_empty());

        // Cost table coefficients preserved across reset (calibration
        // accumulates across queries; only per-query receipts reset).
        let after_cells = adapter.cost_table().read().unwrap()
            .lookup("qmm", &BackendId::new("cpu:op-qmm"));
        assert_eq!(before_cells.0.a_pj_per_flop, after_cells.0.a_pj_per_flop);
        assert_eq!(before_cells.1, after_cells.1);

        // Dispatch cache preserved across reset (cached decisions
        // survive a cascade query boundary; only invalidated on
        // explicit `clear_dispatch_cache`).
        assert_eq!(adapter.dispatch_cache_size(), 1);
    }

    #[test]
    fn over_budget_tracks_session_spend() {
        let mut adapter = PlacementQmmAdapter::with_metal();
        let w = synthetic_q8(1024, 512); // ~1 µJ per call at default priors
        let a: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();

        // 0 = unlimited (cascade's sentinel).
        assert!(!adapter.over_budget(0));
        assert_eq!(adapter.remaining_budget_uj(0), i64::MAX);

        // No spend yet → any positive budget is plenty.
        assert!(!adapter.over_budget(1_000_000));

        // After dispatches, spend accumulates; if a low budget gets
        // crossed we report over.
        for _ in 0..5 { let _ = adapter.qmm_1d(&a, &w); }
        let spent_uj = adapter.stats.estimated_microjoules();
        assert!(spent_uj > 0.0);
        let tight_budget = (spent_uj * 0.5) as u64;
        assert!(adapter.over_budget(tight_budget),
            "spent ~{spent_uj:.3} µJ, budget {tight_budget} should be over");
        let generous_budget = (spent_uj * 2.0).max(100.0) as u64;
        assert!(!adapter.over_budget(generous_budget),
            "spent ~{spent_uj:.3} µJ, budget {generous_budget} should NOT be over");
    }

    #[test]
    fn remaining_budget_uj_decreases_with_spend() {
        let mut adapter = PlacementQmmAdapter::with_metal();
        let w = synthetic_q8(1024, 512);
        let a: Vec<f32> = (0..512).map(|i| (i as f32 * 0.005).cos()).collect();

        let budget_uj = 100_000_000_u64; // 100 J
        let r0 = adapter.remaining_budget_uj(budget_uj);
        let _ = adapter.qmm_1d(&a, &w);
        let r1 = adapter.remaining_budget_uj(budget_uj);
        let _ = adapter.qmm_1d(&a, &w);
        let r2 = adapter.remaining_budget_uj(budget_uj);
        assert!(r1 < r0, "remaining should decrease after a dispatch; {r0} -> {r1}");
        assert!(r2 < r1, "remaining should decrease again; {r1} -> {r2}");
    }

    #[test]
    fn rce_energy_meter_observes_real_qmm_dispatches() {
        // The end-to-end self-calibration proof: the production
        // `rce-energy` meter (macOS IOReport / Linux RAPL /
        // TDP fallback — whichever the host gives us) brackets each
        // qmm dispatch and feeds observed pJ to CostTable::observe.
        // After enough calls the (op, backend) cell flips to
        // measured = true. No fixtures — real hardware reading.
        //
        // Workload sizing note: `CostTable::observe` silently drops
        // 0-pJ readings to keep noise out of the EWMA. The TDP
        // fallback backend computes pJ from `process_cpu_time × TDP`,
        // which legitimately rounds to zero for sub-millisecond
        // dispatches. We size the per-call workload large enough that
        // every backend in the family (IOReport, RAPL, TDP fallback)
        // sees a non-zero interval. 1024×1024 Q8_0 qmm runs ~tens of
        // ms on CPU — well above any backend's noise floor.
        let mut adapter = PlacementQmmAdapter::new()
            .with_rce_energy_meter();
        let w = synthetic_q8(1024, 1024);
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();

        // Run enough calls to clear the measured_after threshold (20)
        // even if a few early dispatches read as zero on a fresh
        // backend subscription.
        for _ in 0..30 { let _ = adapter.qmm_1d(&a, &w); }

        let table = adapter.cost_table();
        let observations = table.read().unwrap()
            .observations_for("qmm", &BackendId::new("cpu:op-qmm"));
        // At minimum the threshold (20) must land for the cell to
        // flip. We don't insist all 30 land — real measurement is
        // allowed to drop noise-floor readings.
        assert!(observations >= 20,
            "expected ≥20 observations to land from 30 real dispatches; got {observations}. \
             This indicates either the backend isn't producing useful readings or the \
             workload is below noise floor on this host.");
        let (_, measured) = table.read().unwrap()
            .lookup("qmm", &BackendId::new("cpu:op-qmm"));
        assert!(measured,
            "after ≥20 real-hardware observations the cell should be measured");
    }

    #[test]
    fn transcript_digest_changes_when_cost_table_changes() {
        // The "calibration drift" property: a replica running an updated
        // cost table (different estimated_picojoules per dispatch) diverges
        // from one running the prior table. This is what lets the verifier
        // detect "this trace was produced under a different calibration
        // than mine" cleanly via the digest.
        use trace::{make_event, DispatchTranscript};
        let mut t_v1 = DispatchTranscript::new();
        t_v1.append(make_event("qmm", &[32], &[16, 32, 544, 1], "cpu:op-qmm", 1.0e6, false));
        let mut t_v2 = DispatchTranscript::new();
        // Same dispatch, different cost cell → different estimate.
        t_v2.append(make_event("qmm", &[32], &[16, 32, 544, 1], "cpu:op-qmm", 9.5e5, false));
        assert_ne!(t_v1.digest(), t_v2.digest());
    }
}
