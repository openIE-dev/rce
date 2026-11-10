//! RCE placement — Stage 3+4 prototype.
//!
//! See `docs/rce-unified-ir-and-energy-routing.md`. This crate implements
//! the keystone the doc identifies as missing today:
//!
//! - **Stage 3 (HAL):** `CostTable` + `LinearCoeffs` (the §4.3 cost model) and
//!   a concrete `Backend` impl (`MatmulBackend`) that wraps a leaf and prices
//!   it via the table. `EnergyEstimate.basis` (D.11, in `vm-core::cost`)
//!   is the seam between an analytical cell (`est`) and a cell whose
//!   basis is the meter's (`reported` or `measured`).
//!
//! - **Stage 4 (placement):** `pick_backend` — the cost-model-driven selector
//!   that replaces the cost-blind first-claim-wins dispatch chain in the VM
//!   scheduler. Each candidate backend is asked to compile the slice; the one
//!   with the lowest `cost(plan).energy.picojoules` wins, with capability
//!   constraints surfaced naturally as `compile` errors.
//!
//! Dispatch is a stub in this prototype (returns `DispatchError::Driver` with
//! a clear "Stage 5" message). The artifact this crate proves is **the
//! routing decision** — that with calibrated coefficients (derived from
//! `rce-energy-xover` on real hardware), the cheapest-by-joules backend
//! flips between CPU and GPU at the measured crossover, and a hypothetical
//! silicon leaf (the §1 `BackendId::silicon:tier-a:*` naming already in the
//! `Backend` trait) wins both once present.

use std::collections::HashMap;
use std::sync::Arc;

pub use rce_vm_core::backend::{
    Backend, BackendId, BackendKind, Capabilities, DispatchError, GraphSlice,
};
pub use rce_vm_core::cost::{Cost, EnergyBasis, EnergyEstimate};
pub use rce_vm_core::plan::{Plan, PlanError};
pub use rce_vm_core::tensor::TensorHandle;
pub use rce_vm_core::value::Value;

// ── Op-slice encoding ──────────────────────────────────────────────
//
// A real `GraphSlice` carries a CBOR-encoded `rce-format::Graph`. For this
// prototype, `graph_cbor` carries a flat tagged record:
//
//   [magic: 0xF0 0x90]            (2 bytes — distinguishes from arbitrary CBOR)
//   [op_name_len: u8][op_name: utf-8 bytes]
//   [n_inputs: u8]
//   for each input:
//     [rank: u8]
//     [dim_0..rank: u64 little-endian each]
//     [dtype_id: u8]              (1 = F32, 2 = F16, 3 = BF16, 4 = I8, 5 = U8, 6 = I32)
//
// A `Backend::compile` that gets a payload with a different magic returns
// `CapabilityDenied`. Backward compat: the legacy 24-byte `[m,k,n]` matmul
// format is still accepted via `unpack_matmul`, but new code should call
// `op_slice` which yields the tagged record.

pub const SLICE_MAGIC: [u8; 2] = [0xF0, 0x90];

/// DType identifiers in the slice encoding. Stable wire IDs.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceDType {
    F32 = 1, F16 = 2, BF16 = 3, I8 = 4, U8 = 5, I32 = 6,
}

impl SliceDType {
    pub fn bytes(self) -> u64 {
        match self {
            SliceDType::F32 | SliceDType::I32 => 4,
            SliceDType::F16 | SliceDType::BF16 => 2,
            SliceDType::I8 | SliceDType::U8 => 1,
        }
    }
    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            1 => Some(SliceDType::F32),
            2 => Some(SliceDType::F16),
            3 => Some(SliceDType::BF16),
            4 => Some(SliceDType::I8),
            5 => Some(SliceDType::U8),
            6 => Some(SliceDType::I32),
            _ => None,
        }
    }
}

/// Pack a single-node op slice: op name + one shape per input + dtype per input.
///
/// `shapes` and `dtypes` are parallel arrays; `shapes.len() == dtypes.len()`.
/// Returns the tagged bytes (magic-prefixed) suitable for `GraphSlice.graph_cbor`.
pub fn pack_op_slice(op: &str, shapes: &[&[u64]], dtypes: &[SliceDType]) -> Vec<u8> {
    assert_eq!(shapes.len(), dtypes.len(), "shapes and dtypes must be parallel");
    assert!(op.len() <= u8::MAX as usize, "op name too long");
    assert!(shapes.len() <= u8::MAX as usize, "too many inputs");
    let mut v = Vec::with_capacity(2 + 1 + op.len() + 1 + shapes.len() * (1 + 8 + 1));
    v.extend_from_slice(&SLICE_MAGIC);
    v.push(op.len() as u8);
    v.extend_from_slice(op.as_bytes());
    v.push(shapes.len() as u8);
    for (shape, dt) in shapes.iter().zip(dtypes.iter()) {
        assert!(shape.len() <= u8::MAX as usize, "shape rank too high");
        v.push(shape.len() as u8);
        for &d in *shape { v.extend_from_slice(&d.to_le_bytes()); }
        v.push(*dt as u8);
    }
    v
}

/// Unpack what `pack_op_slice` produced. Returns `(op_name, shapes_per_input,
/// dtypes_per_input)`. Returns `None` on magic mismatch / truncation / bad
/// dtype id.
pub fn unpack_op_slice(bytes: &[u8]) -> Option<(String, Vec<Vec<u64>>, Vec<SliceDType>)> {
    if bytes.len() < 2 + 1 + 1 { return None; }
    if bytes[0..2] != SLICE_MAGIC { return None; }
    let mut p = 2usize;
    let op_len = *bytes.get(p)? as usize; p += 1;
    if p + op_len > bytes.len() { return None; }
    let op = std::str::from_utf8(&bytes[p..p + op_len]).ok()?.to_string();
    p += op_len;
    let n_inputs = *bytes.get(p)? as usize; p += 1;
    let mut shapes = Vec::with_capacity(n_inputs);
    let mut dtypes = Vec::with_capacity(n_inputs);
    for _ in 0..n_inputs {
        let rank = *bytes.get(p)? as usize; p += 1;
        if p + rank * 8 + 1 > bytes.len() { return None; }
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            let d = u64::from_le_bytes(bytes[p..p + 8].try_into().ok()?);
            dims.push(d);
            p += 8;
        }
        let dt = SliceDType::from_id(bytes[p])?;
        p += 1;
        shapes.push(dims);
        dtypes.push(dt);
    }
    Some((op, shapes, dtypes))
}

/// Build a tagged `GraphSlice` for a single op node.
pub fn op_slice(
    op: &str,
    shapes: &[&[u64]],
    dtypes: &[SliceDType],
    strict: bool,
) -> GraphSlice {
    GraphSlice {
        graph_cbor: pack_op_slice(op, shapes, dtypes),
        entry_node: 0,
        strict_deterministic: strict,
    }
}

// ── Backward-compat: legacy [m,k,n] matmul slice ───────────────────

pub const MATMUL_SLICE_LEN: usize = 24;

pub fn pack_matmul(m: u64, k: u64, n: u64) -> Vec<u8> {
    let mut v = Vec::with_capacity(MATMUL_SLICE_LEN);
    v.extend_from_slice(&m.to_le_bytes());
    v.extend_from_slice(&k.to_le_bytes());
    v.extend_from_slice(&n.to_le_bytes());
    v
}

pub fn unpack_matmul(bytes: &[u8]) -> Option<(u64, u64, u64)> {
    if bytes.len() != MATMUL_SLICE_LEN { return None; }
    let m = u64::from_le_bytes(bytes[0..8].try_into().ok()?);
    let k = u64::from_le_bytes(bytes[8..16].try_into().ok()?);
    let n = u64::from_le_bytes(bytes[16..24].try_into().ok()?);
    Some((m, k, n))
}

pub fn matmul_slice(m: u64, k: u64, n: u64, strict: bool) -> GraphSlice {
    GraphSlice {
        graph_cbor: pack_matmul(m, k, n),
        entry_node: 0,
        strict_deterministic: strict,
    }
}

// ── Shape models — FLOPs / bytes-touched per op ────────────────────

/// FLOPs and bytes-touched for a square or rectangular matmul: the two
/// features the §4.4 cost model regresses on.
pub fn matmul_flops_bytes(m: u64, k: u64, n: u64, dtype_bytes: u64) -> (u64, u64) {
    let flops = 2 * m * k * n;
    let bytes = dtype_bytes * (m * k + k * n + m * n);
    (flops, bytes)
}

/// FLOPs and bytes-touched for an element-wise unary (silu, gelu, rms_norm
/// approx) over `n` elements. ~4 FLOPs/element (mul, exp, add, div for the
/// sigmoid-style ones) + in+out memory traffic.
pub fn elementwise_flops_bytes(n: u64, dtype_bytes: u64) -> (u64, u64) {
    let flops = 4 * n;
    let bytes = 2 * dtype_bytes * n;
    (flops, bytes)
}

/// Compute `(flops, bytes)` for a tagged op-slice. Returns `None` for unknown
/// ops; callers can fall back to a zero/default estimate.
pub fn shape_to_flops_bytes(
    op: &str,
    shapes: &[Vec<u64>],
    dtype_bytes: u64,
) -> Option<(u64, u64)> {
    match op {
        "matmul" => {
            let a = shapes.first()?;
            let b = shapes.get(1)?;
            if a.len() < 2 || b.len() < 2 { return None; }
            Some(matmul_flops_bytes(a[0], a[1], b[1], dtype_bytes))
        }
        "silu" | "gelu" | "rms_norm" | "layer_norm" => {
            let s = shapes.first()?;
            let n: u64 = s.iter().product();
            Some(elementwise_flops_bytes(n, dtype_bytes))
        }
        _ => None,
    }
}

// ── Cost model ─────────────────────────────────────────────────────

/// Feature-model coefficients: pJ ≈ a·FLOPs + b·bytes + c.
///
/// The OLS fit emitted by `rce-energy-xover` drops directly into this struct
/// — `a` is the per-FLOP cost (the silicon-spec's `pj_per_flop`), `b` is the
/// per-byte transfer/memory cost (dominates the GPU's small-matmul loss), `c`
/// is the per-call fixed overhead (dispatch + kernel launch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCoeffs {
    pub a_pj_per_flop: f64,
    pub b_pj_per_byte: f64,
    pub c_pj_overhead: f64,
}

impl LinearCoeffs {
    pub fn estimate_pj(&self, flops: u64, bytes: u64) -> f64 {
        self.a_pj_per_flop * flops as f64
            + self.b_pj_per_byte * bytes as f64
            + self.c_pj_overhead
    }
}

/// Per-(op, backend) cost-model coefficients with a D.11 basis.
/// Missing cells fall back to the analytical `fallback` prior (`est`).
/// `lookup` returns the coefficients and that basis.
///
/// `observe` keeps the cell's basis until `observed_after` samples have
/// landed. The cell then takes the weakest basis among those samples, so
/// the label is the meter's basis and a fallback sample cannot be stored
/// as `reported` or `measured`.
#[derive(Debug, Clone)]
pub struct CostTable {
    cells: HashMap<(String, BackendId), (LinearCoeffs, EnergyBasis)>,
    /// Per-cell count of `observe()` calls.
    observations: HashMap<(String, BackendId), u32>,
    /// Weakest D.11 basis among the samples `observe` has kept.
    sample_basis: HashMap<(String, BackendId), EnergyBasis>,
    fallback: LinearCoeffs,
    /// EWMA gain on the observed-vs-estimated ratio. Default 0.125
    /// (Jacobson's SRTT gain, the same gain mesh-core's LinkStats uses).
    alpha: f64,
    /// After this many observations a cell takes the meter's basis.
    /// Sample count alone, not a confidence interval. The EWMA is the
    /// noise filter.
    observed_after: u32,
}

impl CostTable {
    pub fn new(fallback: LinearCoeffs) -> Self {
        Self {
            cells: HashMap::new(),
            observations: HashMap::new(),
            sample_basis: HashMap::new(),
            fallback,
            alpha: 0.125,
            observed_after: 20,
        }
    }

    /// Tune the EWMA gain (0 < α ≤ 1). Higher = faster convergence,
    /// noisier. 0.125 is Jacobson's pick and a good default.
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        debug_assert!(alpha > 0.0 && alpha <= 1.0);
        self.alpha = alpha;
        self
    }

    /// Tune the sample count at which a cell takes the meter's basis.
    pub fn with_observed_after(mut self, n: u32) -> Self {
        self.observed_after = n;
        self
    }

    /// Previous name of [`CostTable::with_observed_after`].
    pub fn with_measured_after(self, n: u32) -> Self {
        self.with_observed_after(n)
    }

    pub fn insert(
        &mut self,
        op: impl Into<String>,
        backend: &BackendId,
        coeffs: LinearCoeffs,
        basis: EnergyBasis,
    ) {
        self.cells.insert((op.into(), backend.clone()), (coeffs, basis));
    }
    pub fn lookup(&self, op: &str, backend: &BackendId) -> (LinearCoeffs, EnergyBasis) {
        match self.cells.get(&(op.to_string(), backend.clone())) {
            Some((c, b)) => (*c, *b),
            None => (self.fallback, EnergyBasis::Est),
        }
    }
    pub fn estimate(&self, op: &str, backend: &BackendId, flops: u64, bytes: u64) -> EnergyEstimate {
        let (c, basis) = self.lookup(op, backend);
        EnergyEstimate { picojoules: c.estimate_pj(flops, bytes), basis }
    }

    /// **The self-calibration loop**: an actual dispatch ran on (op,
    /// backend) with the given (flops, bytes) feature inputs and was
    /// observed to consume `observed_pj` picojoules. EWMA the
    /// observed-vs-estimated ratio into the cell's coefficients —
    /// preserves the a/b/c shape (which the analytical prior got
    /// approximately right by hypothesis), rescales the magnitude.
    ///
    /// This is the in-product equivalent of running `rce-energy-xover`
    /// against a quiesced host, but tolerant of noise the way Jacobson's
    /// SRTT estimator is — the EWMA absorbs single-sample outliers, and
    /// enough samples for the cell to take `meter_basis` (the weakest
    /// basis if the samples disagree).
    ///
    /// Non-finite / non-positive observations are silently ignored
    /// (boundary safety: observe() may be called from a hot loop and
    /// must not panic). A non-positive sample is not stored as 0.
    pub fn observe(
        &mut self,
        op: &str,
        backend: &BackendId,
        flops: u64,
        bytes: u64,
        observed_pj: f64,
        meter_basis: EnergyBasis,
    ) {
        if !observed_pj.is_finite() || observed_pj <= 0.0 { return; }
        let key = (op.to_string(), backend.clone());
        let (coeffs, prior) = self.cells.get(&key).copied()
            .unwrap_or((self.fallback, EnergyBasis::Est));
        let estimated = coeffs.estimate_pj(flops, bytes);
        if !estimated.is_finite() || estimated <= 0.0 { return; }

        let ratio = observed_pj / estimated;
        // Scale a/b/c uniformly: pj = a·flops + b·bytes + c, so
        // multiplying all coefficients by k multiplies the estimate by k.
        // We move the multiplicative factor by α toward the observed
        // ratio. The scale converges for a consistent observed_pj sequence.
        let scale = 1.0 + self.alpha * (ratio - 1.0);
        let new_coeffs = LinearCoeffs {
            a_pj_per_flop: coeffs.a_pj_per_flop * scale,
            b_pj_per_byte: coeffs.b_pj_per_byte * scale,
            c_pj_overhead: coeffs.c_pj_overhead * scale,
        };
        let count = self.observations.entry(key.clone()).or_insert(0);
        *count = count.saturating_add(1);
        let weakest = self.sample_basis.get(&key).copied()
            .map(|b| b.weaker(meter_basis))
            .unwrap_or(meter_basis);
        self.sample_basis.insert(key.clone(), weakest);
        let basis = if *count >= self.observed_after { weakest } else { prior };
        self.cells.insert(key, (new_coeffs, basis));
    }

    /// Observation count for `(op, backend)` — what the cell has seen.
    pub fn observations_for(&self, op: &str, backend: &BackendId) -> u32 {
        self.observations.get(&(op.to_string(), backend.clone())).copied().unwrap_or(0)
    }

    /// Parse a TOML CostTable in the exact format `rce-energy-xover` emits:
    ///
    /// ```toml
    /// [[cells]]
    /// op = "matmul"
    /// backend = "cpu:op-blas"
    /// a_pj_per_flop = 5.06
    /// b_pj_per_byte = 0.0
    /// c_pj_overhead = 1024.7
    /// basis = "reported"
    /// ```
    ///
    /// Hand-rolled parser scoped to this exact dialect (one-line key=value,
    /// `[[cells]]` per cell, `#` comments, blank-line separators) — keeps
    /// `rce-placement` dep-free. `fallback` is the analytical prior used for
    /// any (op, backend) not present in the file.
    pub fn from_toml(s: &str, fallback: LinearCoeffs) -> Result<Self, String> {
        let mut table = CostTable::new(fallback);
        let mut cur: Option<HashMap<String, String>> = None;
        for (i, raw) in s.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if line == "[[cells]]" {
                if let Some(c) = cur.take() {
                    flush_cell(&mut table, c).map_err(|e| format!("line {}: {e}", i + 1))?;
                }
                cur = Some(HashMap::new());
                continue;
            }
            let (k, v) = line.split_once('=').ok_or_else(|| {
                format!("line {}: malformed (no `=`): {line}", i + 1)
            })?;
            cur.as_mut().ok_or_else(|| {
                format!("line {}: key/value outside [[cells]]: {line}", i + 1)
            })?.insert(k.trim().to_string(), v.trim().to_string());
        }
        if let Some(c) = cur.take() {
            flush_cell(&mut table, c).map_err(|e| format!("final cell: {e}"))?;
        }
        Ok(table)
    }
}

fn flush_cell(table: &mut CostTable, cur: HashMap<String, String>) -> Result<(), String> {
    let op = unquote(cur.get("op").ok_or("missing `op`")?);
    let backend = unquote(cur.get("backend").ok_or("missing `backend`")?);
    let a = parse_f64(&cur, "a_pj_per_flop")?;
    let b = parse_f64(&cur, "b_pj_per_byte")?;
    let c = parse_f64(&cur, "c_pj_overhead")?;
    let basis = if let Some(raw) = cur.get("basis") {
        EnergyBasis::parse(raw).ok_or_else(|| format!("basis `{raw}` is not est, reported or measured"))?
    } else if let Some(raw) = cur.get("measured") {
        // Files written before `basis` stored a bool. `true` named a counter
        // or an observation, which D.11 calls `reported`.
        EnergyBasis::from_legacy_flag(raw.trim() == "true")
    } else {
        EnergyBasis::Est
    };
    table.insert(
        op,
        &BackendId::new(backend),
        LinearCoeffs { a_pj_per_flop: a, b_pj_per_byte: b, c_pj_overhead: c },
        basis,
    );
    Ok(())
}

fn parse_f64(cur: &HashMap<String, String>, key: &str) -> Result<f64, String> {
    cur.get(key)
        .ok_or_else(|| format!("missing `{key}`"))?
        .parse::<f64>()
        .map_err(|e| format!("{key}: {e}"))
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

// ── Backend impl: a priced matmul leaf ────────────────────────────

/// A `Backend` impl that prices a single-matmul slice via a `CostTable`.
/// `dispatch` is a stub (Stage 5) — the artifact this prototype proves is the
/// cost-driven *routing decision*, which only needs `compile` + `cost`.
pub struct MatmulBackend {
    pub id: BackendId,
    pub caps: Capabilities,
    pub table: Arc<CostTable>,
    pub dtype_bytes: u64,
}

impl MatmulBackend {
    pub fn new(id: BackendId, kind: BackendKind, table: Arc<CostTable>) -> Self {
        Self {
            caps: Capabilities {
                kind: Some(kind),
                supports_f32: true,
                ..Default::default()
            },
            id,
            table,
            dtype_bytes: 4,
        }
    }
}

impl Backend for MatmulBackend {
    fn id(&self) -> &BackendId { &self.id }
    fn capabilities(&self) -> Capabilities { self.caps.clone() }

    fn compile(&self, slice: &GraphSlice) -> Result<Plan, DispatchError> {
        // The prototype handles single-matmul slices only; reject everything
        // else explicitly rather than silently passing.
        let _ = unpack_matmul(&slice.graph_cbor).ok_or_else(|| {
            DispatchError::CapabilityDenied(
                "placement prototype handles only matmul slices".into(),
            )
        })?;
        Ok(Plan {
            backend: self.id.clone(),
            artifact: slice.graph_cbor.clone(),
            placements: vec![],
            determinism_strict: slice.strict_deterministic,
        })
    }

    fn dispatch(
        &self,
        _plan: &Plan,
        _inputs: &[TensorHandle],
    ) -> Result<Vec<Value>, DispatchError> {
        Err(DispatchError::Driver(
            "placement prototype: dispatch is unwired (Stage 5); cost-driven selection is the artifact"
                .into(),
        ))
    }

    fn cost(&self, plan: &Plan) -> Cost {
        let (m, k, n) = unpack_matmul(&plan.artifact).unwrap_or((0, 0, 0));
        let (flops, bytes) = matmul_flops_bytes(m, k, n, self.dtype_bytes);
        Cost {
            energy: self.table.estimate("matmul", &plan.backend, flops, bytes),
            time_ns: 0,
            peak_bytes: 0,
        }
    }
}

// ── Backend impl: a generic priced single-op leaf ─────────────────

/// `Backend` impl that prices any single-op slice (matmul / silu / gelu /
/// rms_norm / layer_norm) via a `CostTable` + the `shape_to_flops_bytes`
/// shape model. Decodes the tagged op-slice format from `pack_op_slice`.
///
/// `handles` is the per-instance op allowlist — backends advertise which ops
/// they're prepared to execute (e.g. `["matmul"]` for op-blas, `["silu",
/// "rms_norm", "matmul"]` for an MPS+MetalLeaf bundle). `compile` rejects
/// any op outside the allowlist with `CapabilityDenied`, matching the per-op
/// claim semantics of `PricedBackend::handles()`.
pub struct OpBackend {
    pub id: BackendId,
    pub caps: Capabilities,
    pub table: Arc<CostTable>,
    pub handles: Vec<String>,
    pub dtype_bytes: u64,
}

impl OpBackend {
    pub fn new(
        id: BackendId,
        kind: BackendKind,
        table: Arc<CostTable>,
        handles: Vec<String>,
    ) -> Self {
        Self {
            caps: Capabilities {
                kind: Some(kind),
                supports_f32: true,
                ..Default::default()
            },
            id,
            table,
            handles,
            dtype_bytes: 4,
        }
    }
}

impl Backend for OpBackend {
    fn id(&self) -> &BackendId { &self.id }
    fn capabilities(&self) -> Capabilities { self.caps.clone() }

    fn compile(&self, slice: &GraphSlice) -> Result<Plan, DispatchError> {
        let (op, _shapes, _dtypes) = unpack_op_slice(&slice.graph_cbor).ok_or_else(|| {
            DispatchError::CapabilityDenied(
                "OpBackend expects an op-slice (use op_slice/pack_op_slice)".into(),
            )
        })?;
        if !self.handles.iter().any(|h| h == &op) {
            return Err(DispatchError::CapabilityDenied(format!(
                "{}: op {op:?} not in handles {:?}", self.id.0, self.handles
            )));
        }
        Ok(Plan {
            backend: self.id.clone(),
            artifact: slice.graph_cbor.clone(),
            placements: vec![],
            determinism_strict: slice.strict_deterministic,
        })
    }

    fn dispatch(
        &self,
        _plan: &Plan,
        _inputs: &[TensorHandle],
    ) -> Result<Vec<Value>, DispatchError> {
        Err(DispatchError::Driver(
            "OpBackend prototype: dispatch via the scheduler's PricedBackend path; \
             this Backend trait surface is for compile/cost reasoning"
                .into(),
        ))
    }

    fn cost(&self, plan: &Plan) -> Cost {
        let pj = unpack_op_slice(&plan.artifact)
            .and_then(|(op, shapes, _dtypes)| {
                let (flops, bytes) = shape_to_flops_bytes(&op, &shapes, self.dtype_bytes)?;
                Some(self.table.estimate(&op, &plan.backend, flops, bytes))
            })
            .unwrap_or(EnergyEstimate { picojoules: f64::INFINITY, basis: EnergyBasis::Est });
        Cost { energy: pj, time_ns: 0, peak_bytes: 0 }
    }
}

// ── CALM I-confluence classification (free vs bound fraction) ─────

/// Mesh-core's free-vs-bound split, applied to graph ops. Determines
/// what a fusion/reorder pass is allowed to do safely.
///
/// **Free**: order-independent per its inputs. Same inputs → same
/// outputs regardless of when the op runs relative to peer ops on
/// independent inputs. Element-wise activations (silu, gelu, rms_norm,
/// layer_norm, swiglu) and pure matmul/qmm/conv2d ops are all free —
/// they can be reordered, batched, fused, or run in parallel with peer
/// free ops without changing the program's observable output.
///
/// **Bound**: order-dependent. The op's effect depends on the
/// sequence of prior ops on the *same* state (KV-cache appends, RNN
/// state updates, RNG sampling, argmax over a stream). Bound ops must
/// preserve a total order with respect to other bound ops on the same
/// state; reordering or fusing them is unsafe without a CALM proof.
///
/// Same rationale as mesh-core's lattice/coordinate split: free
/// fraction rides any medium with zero coordination; bound fraction
/// is the only place the system pays a latency tax. In graph terms:
/// fusion is safe across free ops; bound ops force a barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpFraction {
    /// I-confluent — reorder/fuse/parallelize safely.
    Free,
    /// Non-I-confluent — preserve order.
    Bound,
}

/// Map a known op name to its `OpFraction`. Returns `None` for ops
/// we don't have a CALM judgement on yet — callers should default to
/// `Bound` (the safer choice) when they need a verdict.
///
/// The catalogue is curated rather than exhaustive: only ops with a
/// principled CALM classification are listed. Adding an op means
/// thinking about its I-confluence first, not just stamping it.
pub fn op_fraction(op: &str) -> Option<OpFraction> {
    match op {
        // ── Free: element-wise pure functions on independent inputs ──
        //
        // silu / gelu / swiglu / rms_norm / layer_norm act per
        // position (or per group); a chain `silu → rms_norm → gelu`
        // applied to a tensor yields the same output as applying each
        // to the corresponding positions independently. Free fraction.
        "silu" | "gelu" | "swiglu" | "rms_norm" | "layer_norm"
        // matmul / qmm / conv2d are pure functions of their inputs
        // (no internal state across calls). Independent matmuls on
        // independent (a, w) inputs commute. Free.
        | "matmul" | "qmm" | "conv2d"
        // softmax / argmax over a fixed input are pure; ordering them
        // with peer ops on disjoint inputs is fine. (Note: argmax
        // *over a stream* is Bound — see below — but per-call argmax
        // is Free.)
        | "softmax"
        // RoPE / embedding_lookup: pure functions of (input, table).
        | "rope" | "embedding_lookup" => Some(OpFraction::Free),

        // ── Bound: stateful or sequence-dependent ──
        //
        // KV-cache appends are the canonical bound op in transformer
        // inference: position N's append depends on positions 0..N-1
        // having appended in order. Reordering breaks the cache.
        "kv_cache_append" | "kv_cache_write"
        // Sampling reads the RNG; sequencing matters or two replicas
        // emitting the same token order diverge.
        | "sample" | "rng_next"
        // Argmax-over-stream (incremental top-1 over a growing
        // sequence) is bound — distinguish from per-call argmax above.
        | "argmax_stream" => Some(OpFraction::Bound),

        _ => None,
    }
}

/// True iff `op` is classified `Free`. Returns false for unknown ops
/// (the safer default — don't assume an op is reorderable without
/// CALM justification).
pub fn is_free(op: &str) -> bool {
    matches!(op_fraction(op), Some(OpFraction::Free))
}

/// True iff *every* op in `ops` is in the Free fraction. The
/// fusion-pass precondition: a chain of all-Free ops can be reordered,
/// parallelized, or fused into a single kernel without changing the
/// program's output. As soon as one Bound op appears in the chain,
/// fusion is unsafe — the pass must stop at that barrier.
///
/// Unknown ops are treated as Bound for safety. A fusion pass that
/// wants to handle a new op must register it with a CALM judgement in
/// `op_fraction` first.
pub fn fusion_eligible(ops: &[&str]) -> bool {
    ops.iter().all(|op| is_free(op))
}

// ── Placement pass (Stage 4) ──────────────────────────────────────

/// Pick the lowest-joule backend that can compile the slice.
///
/// Backends that don't support the slice return `CapabilityDenied` / `UnsupportedOp`
/// from `compile`; those are silently skipped. The remaining are sorted by
/// `cost(plan).energy.picojoules`; the cheapest wins. CPU does *not* get
/// special treatment — it falls out naturally because its overhead is small
/// (so it wins below the GPU's dispatch-overhead crossover) and its per-FLOP
/// cost is moderate (so it loses to GPU and silicon above the crossover).
pub fn pick_backend<'a>(
    slice: &GraphSlice,
    backends: &'a [Arc<dyn Backend>],
) -> Option<(&'a BackendId, EnergyEstimate)> {
    let mut best: Option<(&BackendId, EnergyEstimate)> = None;
    for be in backends {
        let Ok(plan) = be.compile(slice) else { continue; };
        let c = be.cost(&plan);
        let cheaper = best
            .as_ref()
            .map_or(true, |(_, prev)| c.energy.picojoules < prev.picojoules);
        if cheaper {
            best = Some((be.id(), c.energy));
        }
    }
    best
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Coefficients derived from the actual M3 Ultra measurement (commit
    // 8f1e69e63: matmul-2048 = 86 mJ on CPU, ~5 mJ on GPU).

    /// CPU (op-blas AMX sgemm): ~5 pJ/FLOP, small overhead.
    fn cpu_coeffs() -> LinearCoeffs {
        LinearCoeffs { a_pj_per_flop: 5.0, b_pj_per_byte: 0.0, c_pj_overhead: 1_000_000.0 } // 1µJ
    }
    /// GPU (op-mps MPSMatrixMultiplication): ~0.29 pJ/FLOP, ~5 mJ dispatch overhead.
    fn gpu_coeffs() -> LinearCoeffs {
        LinearCoeffs { a_pj_per_flop: 0.29, b_pj_per_byte: 0.0, c_pj_overhead: 5_000_000_000.0 }
    }
    /// Silicon Tier-A unit (silicon-spec target): two orders of magnitude
    /// cheaper per FLOP than the GPU; tiny overhead. Design target (modelled),
    /// not a reading, so the cell basis is `est`.
    fn silicon_coeffs() -> LinearCoeffs {
        LinearCoeffs { a_pj_per_flop: 0.05, b_pj_per_byte: 0.0, c_pj_overhead: 100_000.0 }
    }

    fn make_table() -> Arc<CostTable> {
        let mut t = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        });
        t.insert("matmul", &BackendId::new("cpu:op-blas"), cpu_coeffs(), EnergyBasis::Reported);
        t.insert("matmul", &BackendId::new("gpu:op-mps"), gpu_coeffs(), EnergyBasis::Reported);
        t.insert("matmul", &BackendId::new("silicon:tier-a:MatMul"), silicon_coeffs(), EnergyBasis::Est);
        Arc::new(t)
    }

    fn make_backends_cpu_gpu_silicon() -> Vec<Arc<dyn Backend>> {
        let t = make_table();
        vec![
            Arc::new(MatmulBackend::new(BackendId::new("cpu:op-blas"), BackendKind::Cpu, t.clone())),
            Arc::new(MatmulBackend::new(BackendId::new("gpu:op-mps"), BackendKind::Metal, t.clone())),
            Arc::new(MatmulBackend::new(BackendId::new("silicon:tier-a:MatMul"), BackendKind::Silicon, t.clone())),
        ]
    }

    #[test]
    fn small_matmul_routes_to_cpu_because_gpu_dispatch_overhead_dominates() {
        let backends = make_backends_cpu_gpu_silicon();
        let two = &backends[..2]; // CPU + GPU, no silicon
        let slice = matmul_slice(64, 64, 64, false);
        let (id, _) = pick_backend(&slice, two).unwrap();
        assert_eq!(id.0, "cpu:op-blas",
            "small matmul should route to CPU; the GPU's ~5 mJ dispatch overhead dominates");
    }

    #[test]
    fn large_matmul_routes_to_gpu_because_flop_cost_amortizes_overhead() {
        let backends = make_backends_cpu_gpu_silicon();
        let two = &backends[..2]; // CPU + GPU, no silicon
        let slice = matmul_slice(2048, 2048, 2048, false);
        let (id, gpu_est) = pick_backend(&slice, two).unwrap();
        assert_eq!(id.0, "gpu:op-mps",
            "large matmul should route to GPU; FLOP cost amortizes the dispatch overhead");
        // The routing-relevant fact: GPU should be a real fraction of CPU at
        // this shape. Exact pJ depends on the coefficient calibration; the
        // ratio is what placement consumes.
        let cpu_plan = backends[0].compile(&slice).unwrap();
        let cpu_pj = backends[0].cost(&cpu_plan).energy.picojoules;
        assert!(gpu_est.picojoules < cpu_pj * 0.5,
            "GPU should be <50% of CPU at 2048: GPU={:.2e} pJ, CPU={:.2e} pJ",
            gpu_est.picojoules, cpu_pj);
        assert_eq!(gpu_est.basis, EnergyBasis::Reported, "GPU coefficients are a counter basis, not a wall meter");
    }

    #[test]
    fn silicon_wins_both_extremes_when_available_because_orders_of_magnitude_cheaper() {
        // The hypothetical Tier-A silicon unit is ~100× cheaper per FLOP than
        // GPU; with tiny overhead it wins both small and large. The routing
        // decision is automatic — no special case, just min-joules.
        let backends = make_backends_cpu_gpu_silicon();
        for &n in &[64u64, 2048u64] {
            let slice = matmul_slice(n, n, n, false);
            let (id, est) = pick_backend(&slice, &backends).unwrap();
            assert_eq!(id.0, "silicon:tier-a:MatMul",
                "silicon should win at n={n}");
            assert_eq!(est.basis, EnergyBasis::Est, "silicon coefficients are a design target (est)");
        }
    }

    #[test]
    fn cpu_to_gpu_crossover_is_detectable_in_the_sweep() {
        let backends = make_backends_cpu_gpu_silicon();
        let two = &backends[..2];
        let mut winners = Vec::new();
        for &n in &[32u64, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048] {
            let slice = matmul_slice(n, n, n, false);
            let (id, _) = pick_backend(&slice, two).unwrap();
            winners.push((n, id.0.clone()));
        }
        // Must contain at least one CPU win and one GPU win, with CPU at the
        // small end and GPU at the large end (no oscillation in a monotone sweep).
        assert!(winners.iter().any(|(_, w)| w == "cpu:op-blas"));
        assert!(winners.iter().any(|(_, w)| w == "gpu:op-mps"));
        assert_eq!(winners.first().unwrap().1, "cpu:op-blas", "CPU at smallest n");
        assert_eq!(winners.last().unwrap().1, "gpu:op-mps",  "GPU at largest n");
    }

    #[test]
    fn dispatch_stub_returns_clear_stage5_error() {
        let backends = make_backends_cpu_gpu_silicon();
        let slice = matmul_slice(64, 64, 64, false);
        let be = &backends[0];
        let plan = be.compile(&slice).unwrap();
        match be.dispatch(&plan, &[]) {
            Err(DispatchError::Driver(msg)) => {
                assert!(msg.contains("Stage 5"), "stub should name the missing stage; got: {msg}");
            }
            other => panic!("expected Driver(...) stub, got {other:?}"),
        }
    }

    #[test]
    fn non_matmul_slice_rejected_with_capability_denied() {
        let backends = make_backends_cpu_gpu_silicon();
        let be = &backends[0];
        let bad = GraphSlice {
            graph_cbor: vec![0u8; 12], // wrong length
            entry_node: 0,
            strict_deterministic: false,
        };
        match be.compile(&bad) {
            Err(DispatchError::CapabilityDenied(_)) => {}
            other => panic!("expected CapabilityDenied, got {other:?}"),
        }
    }

    #[test]
    fn fallback_prior_used_for_unknown_op_or_backend() {
        // A CostTable miss falls back to the analytical prior, basis `est`.
        let t = make_table();
        let est = t.estimate("conv2d", &BackendId::new("unknown"), 1_000_000, 16_384);
        // Fallback coeffs above: a=1.0 pJ/FLOP, b=0, c=0 ⇒ 1_000_000 pJ.
        assert_eq!(est.picojoules, 1_000_000.0);
        assert_eq!(est.basis, EnergyBasis::Est);
    }

    // ── TOML round-trip ───────────────────────────────────────────────

    #[test]
    fn toml_loader_parses_harness_emit_format() {
        // Verbatim copy of `rce-energy-xover`'s emit, including header/footer
        // comments + blank-line separators. Loader must accept this unchanged.
        let s = r#"
# ----- cost_table.toml (paste into a file; load at scheduler init) -----
[[cells]]
op = "matmul"
backend = "cpu:op-blas"
a_pj_per_flop = 5.060000
b_pj_per_byte = 0.000000
c_pj_overhead = 1024.700
basis = "reported"

[[cells]]
op = "matmul"
backend = "gpu:op-mps"
a_pj_per_flop = 0.290000
b_pj_per_byte = 0.000000
c_pj_overhead = 5000000000.000
basis = "reported"
# ----- end cost_table.toml -----
"#;
        let fb = LinearCoeffs { a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 };
        let t = CostTable::from_toml(s, fb).expect("parse");

        let (cpu, b_cpu) = t.lookup("matmul", &BackendId::new("cpu:op-blas"));
        assert!((cpu.a_pj_per_flop - 5.06).abs() < 1e-9);
        assert!((cpu.c_pj_overhead - 1024.7).abs() < 1e-3);
        assert_eq!(b_cpu, EnergyBasis::Reported);

        let (gpu, b_gpu) = t.lookup("matmul", &BackendId::new("gpu:op-mps"));
        assert!((gpu.a_pj_per_flop - 0.29).abs() < 1e-9);
        assert!((gpu.c_pj_overhead - 5_000_000_000.0).abs() < 1e-3);
        assert_eq!(b_gpu, EnergyBasis::Reported);

        // Unknown cells fall back.
        let (fb_c, fb_b) = t.lookup("conv2d", &BackendId::new("nowhere"));
        assert_eq!(fb_c, fb);
        assert_eq!(fb_b, EnergyBasis::Est);

        // A file from before `basis` stored `measured = true` for a counter.
        let legacy = "[[cells]]\nop = \"matmul\"\nbackend = \"cpu:op-blas\"\na_pj_per_flop = 1.0\nb_pj_per_byte = 0.0\nc_pj_overhead = 0.0\nmeasured = true\n";
        let old = CostTable::from_toml(legacy, fb).expect("legacy");
        assert_eq!(old.lookup("matmul", &BackendId::new("cpu:op-blas")).1, EnergyBasis::Reported);
        let legacy_est = legacy.replace("measured = true", "measured = false");
        let old_est = CostTable::from_toml(&legacy_est, fb).expect("legacy est");
        assert_eq!(old_est.lookup("matmul", &BackendId::new("cpu:op-blas")).1, EnergyBasis::Est);
    }

    #[test]
    fn toml_loader_rejects_malformed_input() {
        let fb = LinearCoeffs { a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 };
        // Missing required key.
        let bad = "[[cells]]\nop = \"matmul\"\nbackend = \"cpu\"\n# missing coeffs\n";
        let err = CostTable::from_toml(bad, fb).unwrap_err();
        assert!(err.contains("missing"), "expected `missing` in error; got: {err}");
        // Key/value outside any [[cells]].
        let bad2 = "op = \"matmul\"\n";
        let err2 = CostTable::from_toml(bad2, fb).unwrap_err();
        assert!(err2.contains("outside"), "got: {err2}");
    }

    // ── Op-slice round-trip ───────────────────────────────────────────

    #[test]
    fn op_slice_roundtrip_matmul() {
        let s = op_slice(
            "matmul",
            &[&[64u64, 128], &[128, 256]],
            &[SliceDType::F32, SliceDType::F32],
            true,
        );
        let (op, shapes, dtypes) = unpack_op_slice(&s.graph_cbor).expect("decode");
        assert_eq!(op, "matmul");
        assert_eq!(shapes, vec![vec![64u64, 128], vec![128, 256]]);
        assert_eq!(dtypes, vec![SliceDType::F32, SliceDType::F32]);
        assert!(s.strict_deterministic);
    }

    #[test]
    fn op_slice_roundtrip_silu() {
        let s = op_slice("silu", &[&[1u64, 2048, 4096]], &[SliceDType::F16], false);
        let (op, shapes, dtypes) = unpack_op_slice(&s.graph_cbor).expect("decode");
        assert_eq!(op, "silu");
        assert_eq!(shapes, vec![vec![1u64, 2048, 4096]]);
        assert_eq!(dtypes, vec![SliceDType::F16]);
    }

    #[test]
    fn op_slice_rejects_bad_magic() {
        assert!(unpack_op_slice(&[0x00, 0x00, 5, b'h', b'i']).is_none());
        // Legacy 24-byte matmul payload has no magic, so it must NOT be
        // accepted by unpack_op_slice (the encodings are disjoint).
        let legacy = pack_matmul(64, 64, 64);
        assert!(unpack_op_slice(&legacy).is_none());
    }

    // ── OpBackend (generic priced single-op leaf) ─────────────────────

    fn silu_coeffs_cpu() -> LinearCoeffs {
        // ~2 pJ/FLOP CPU (4 FLOPs/elem; element-wise is bandwidth-bound), small overhead.
        LinearCoeffs { a_pj_per_flop: 2.0, b_pj_per_byte: 0.5, c_pj_overhead: 100_000.0 }
    }
    fn silu_coeffs_gpu() -> LinearCoeffs {
        // ~0.1 pJ/FLOP GPU, lower overhead than matmul kernel.
        LinearCoeffs { a_pj_per_flop: 0.1, b_pj_per_byte: 0.05, c_pj_overhead: 500_000.0 }
    }

    fn multi_op_table() -> Arc<CostTable> {
        let mut t = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        });
        t.insert("matmul", &BackendId::new("cpu:op-blas"), cpu_coeffs(), EnergyBasis::Reported);
        t.insert("matmul", &BackendId::new("gpu:op-mps"), gpu_coeffs(), EnergyBasis::Reported);
        t.insert("silu", &BackendId::new("cpu:op-nn-prims"), silu_coeffs_cpu(), EnergyBasis::Reported);
        t.insert("silu", &BackendId::new("gpu:op-metal"), silu_coeffs_gpu(), EnergyBasis::Reported);
        Arc::new(t)
    }

    #[test]
    fn op_backend_routes_matmul_per_size_via_pick_backend() {
        let t = multi_op_table();
        let backends: Vec<Arc<dyn Backend>> = vec![
            Arc::new(OpBackend::new(BackendId::new("cpu:op-blas"), BackendKind::Cpu,
                t.clone(), vec!["matmul".into()])),
            Arc::new(OpBackend::new(BackendId::new("gpu:op-mps"), BackendKind::Metal,
                t.clone(), vec!["matmul".into()])),
        ];
        let small = op_slice("matmul", &[&[64u64, 64], &[64, 64]],
            &[SliceDType::F32, SliceDType::F32], false);
        let big = op_slice("matmul", &[&[2048u64, 2048], &[2048, 2048]],
            &[SliceDType::F32, SliceDType::F32], false);
        assert_eq!(pick_backend(&small, &backends).unwrap().0.0, "cpu:op-blas");
        assert_eq!(pick_backend(&big, &backends).unwrap().0.0, "gpu:op-mps");
    }

    #[test]
    fn op_backend_routes_silu_to_gpu_when_n_large() {
        let t = multi_op_table();
        let backends: Vec<Arc<dyn Backend>> = vec![
            Arc::new(OpBackend::new(BackendId::new("cpu:op-nn-prims"), BackendKind::Cpu,
                t.clone(), vec!["silu".into()])),
            Arc::new(OpBackend::new(BackendId::new("gpu:op-metal"), BackendKind::Metal,
                t.clone(), vec!["silu".into()])),
        ];
        // Big tensor: GPU's lower per-FLOP cost should beat CPU even after
        // overhead amortizes.
        let big = op_slice("silu", &[&[1024u64, 4096]], &[SliceDType::F32], false);
        let winner = pick_backend(&big, &backends).unwrap().0.0.clone();
        assert_eq!(winner, "gpu:op-metal");
    }

    #[test]
    fn op_backend_rejects_op_outside_handles() {
        let t = multi_op_table();
        let only_matmul = OpBackend::new(
            BackendId::new("cpu:op-blas"), BackendKind::Cpu, t, vec!["matmul".into()]);
        let silu = op_slice("silu", &[&[64u64]], &[SliceDType::F32], false);
        match only_matmul.compile(&silu) {
            Err(DispatchError::CapabilityDenied(msg)) => {
                assert!(msg.contains("silu"), "expected op name in error; got: {msg}");
            }
            other => panic!("expected CapabilityDenied, got {other:?}"),
        }
    }

    #[test]
    fn pick_backend_routes_per_op_in_mixed_backend_set() {
        // Mixed backends: one handles matmul on CPU, one handles matmul on GPU,
        // one handles silu on GPU. pick_backend must route per-op-per-slice.
        let t = multi_op_table();
        let backends: Vec<Arc<dyn Backend>> = vec![
            Arc::new(OpBackend::new(BackendId::new("cpu:op-blas"), BackendKind::Cpu,
                t.clone(), vec!["matmul".into()])),
            Arc::new(OpBackend::new(BackendId::new("gpu:op-mps"), BackendKind::Metal,
                t.clone(), vec!["matmul".into()])),
            Arc::new(OpBackend::new(BackendId::new("gpu:op-metal"), BackendKind::Metal,
                t.clone(), vec!["silu".into(), "rms_norm".into()])),
        ];
        // Large matmul → gpu:op-mps.
        let mm = op_slice("matmul", &[&[2048u64, 2048], &[2048, 2048]],
            &[SliceDType::F32, SliceDType::F32], false);
        assert_eq!(pick_backend(&mm, &backends).unwrap().0.0, "gpu:op-mps");
        // Silu (handled only by op-metal) → gpu:op-metal.
        let silu = op_slice("silu", &[&[2048u64, 2048]], &[SliceDType::F32], false);
        assert_eq!(pick_backend(&silu, &backends).unwrap().0.0, "gpu:op-metal");
        // Conv2d → nobody handles it; pick_backend returns None.
        let conv = op_slice("conv2d", &[&[1u64, 3, 224, 224]], &[SliceDType::F32], false);
        assert!(pick_backend(&conv, &backends).is_none());
    }

    // ── observe() — self-calibration via runtime energy readings ─────

    fn priors() -> Arc<CostTable> {
        let mut t = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        });
        // Start with a clean a=10, b=0, c=0 cell so observations have
        // an unambiguous direction to drift toward.
        t.insert("matmul", &BackendId::new("cpu:test"),
            LinearCoeffs { a_pj_per_flop: 10.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 },
            EnergyBasis::Est);
        Arc::new(t)
    }

    #[test]
    fn observe_with_matching_estimate_is_a_noop_on_coefficients() {
        let mut t = (*priors()).clone();
        let be = BackendId::new("cpu:test");
        // estimated = 10 * 1000 = 10_000 pj. Observe exactly 10_000 ⇒
        // ratio = 1, scale = 1, no drift.
        for _ in 0..5 {
            t.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Reported);
        }
        let (c, basis) = t.lookup("matmul", &be);
        assert!((c.a_pj_per_flop - 10.0).abs() < 1e-9,
            "ratio=1 observation must leave coefficients unchanged; got a={}",
            c.a_pj_per_flop);
        assert_eq!(basis, EnergyBasis::Est, "5 observations is below the default threshold (20), so the prior basis stands");
        assert_eq!(t.observations_for("matmul", &be), 5);
    }

    #[test]
    fn observe_drifts_coefficients_toward_observation() {
        let mut t = (*priors()).clone();
        let be = BackendId::new("cpu:test");
        // estimated = 10 * 1000 = 10_000. Observed = 20_000 ⇒ ratio = 2,
        // scale = 1.125 per observation. After 5 observations the
        // coefficient should be moving steadily toward 20.
        for _ in 0..5 {
            // Re-estimate before each observe so the math is dynamic
            // (mirrors what a live dispatch path does).
            t.observe("matmul", &be, 1000, 0, 20_000.0, EnergyBasis::Reported);
        }
        let (c_after, _) = t.lookup("matmul", &be);
        // After 5 EWMA steps at α=0.125, coefficient should be
        // strictly between 10 and 20.
        assert!(c_after.a_pj_per_flop > 10.0,
            "must have moved up; got {}", c_after.a_pj_per_flop);
        assert!(c_after.a_pj_per_flop < 20.0,
            "EWMA shouldn't overshoot; got {}", c_after.a_pj_per_flop);
    }

    #[test]
    fn observe_inherits_meter_basis_after_threshold() {
        let mut t = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        }).with_observed_after(5);
        let be = BackendId::new("cpu:test");
        t.insert("matmul", &be,
            LinearCoeffs { a_pj_per_flop: 10.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 },
            EnergyBasis::Est);

        // 4 observations: the prior basis still stands.
        for _ in 0..4 { t.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Reported); }
        let (_, b) = t.lookup("matmul", &be);
        assert_eq!(b, EnergyBasis::Est, "4 obs below threshold 5 keeps the prior basis");

        // 5th observation takes the meter's basis.
        t.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Reported);
        let (_, b2) = t.lookup("matmul", &be);
        assert_eq!(b2, EnergyBasis::Reported, "a counter meter's samples are reported, not measured");

        // A fallback meter cannot be labelled reported or measured.
        let mut est = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        }).with_observed_after(2);
        est.insert("matmul", &be,
            LinearCoeffs { a_pj_per_flop: 10.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 },
            EnergyBasis::Est);
        est.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Est);
        est.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Est);
        assert_eq!(est.lookup("matmul", &be).1, EnergyBasis::Est);

        // A mix takes the weaker basis.
        let mut mix = CostTable::new(LinearCoeffs {
            a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
        }).with_observed_after(2);
        mix.insert("matmul", &be,
            LinearCoeffs { a_pj_per_flop: 10.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0 },
            EnergyBasis::Est);
        mix.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Reported);
        mix.observe("matmul", &be, 1000, 0, 10_000.0, EnergyBasis::Est);
        assert_eq!(mix.lookup("matmul", &be).1, EnergyBasis::Est, "one est sample keeps the cell at est");
    }

    #[test]
    fn observe_ignores_bad_observations_safely() {
        // observe() is called from a hot loop; bad values (NaN, infinity,
        // zero, negative) must be silently dropped without panic.
        let mut t = (*priors()).clone();
        let be = BackendId::new("cpu:test");
        let before = t.lookup("matmul", &be).0.a_pj_per_flop;
        t.observe("matmul", &be, 1000, 0, f64::NAN, EnergyBasis::Reported);
        t.observe("matmul", &be, 1000, 0, f64::INFINITY, EnergyBasis::Reported);
        t.observe("matmul", &be, 1000, 0, -1.0, EnergyBasis::Reported);
        t.observe("matmul", &be, 1000, 0, 0.0, EnergyBasis::Reported);
        t.observe("matmul", &be, 0, 0, 100.0, EnergyBasis::Reported); // zero estimate is ignored
        let after = t.lookup("matmul", &be).0.a_pj_per_flop;
        assert_eq!(before, after, "bad observations must not perturb the cell");
        assert_eq!(t.observations_for("matmul", &be), 0,
            "bad observations must not count toward the measured threshold");
    }

    // ── OpFraction / fusion_eligible ────────────────────────────────

    #[test]
    fn element_wise_activations_are_free() {
        for op in ["silu", "gelu", "swiglu", "rms_norm", "layer_norm"] {
            assert_eq!(op_fraction(op), Some(OpFraction::Free),
                "{op} should be in the free fraction (element-wise pure)");
            assert!(is_free(op));
        }
    }

    #[test]
    fn pure_compute_ops_are_free() {
        for op in ["matmul", "qmm", "conv2d", "softmax", "rope", "embedding_lookup"] {
            assert_eq!(op_fraction(op), Some(OpFraction::Free),
                "{op} should be free (pure function of inputs)");
        }
    }

    #[test]
    fn kv_cache_and_sampling_are_bound() {
        for op in ["kv_cache_append", "kv_cache_write", "sample", "rng_next",
                   "argmax_stream"] {
            assert_eq!(op_fraction(op), Some(OpFraction::Bound),
                "{op} should be bound (stateful / sequence-dependent)");
            assert!(!is_free(op));
        }
    }

    #[test]
    fn unknown_ops_have_no_fraction_and_default_to_unsafe() {
        // `op_fraction` returns None so callers know we have no
        // judgement; `is_free` returns false so the unsafe default is
        // "don't reorder/fuse this." Adding a new op = add it to
        // `op_fraction` with a CALM justification.
        assert_eq!(op_fraction("attention"), None,
            "attention isn't classified yet (KV cache write is bound; reads are free)");
        assert!(!is_free("attention"));
        assert_eq!(op_fraction("__nonexistent_op__"), None);
        assert!(!is_free("__nonexistent_op__"));
    }

    #[test]
    fn fusion_eligible_requires_all_free_ops() {
        // The classic transformer FFN: rms_norm → silu → matmul →
        // rms_norm. All free — fully fusable into one Metal kernel
        // without changing semantics.
        assert!(fusion_eligible(&["rms_norm", "silu", "matmul", "rms_norm"]));
        // Element-wise chain — the case the demo already exploits.
        assert!(fusion_eligible(&["silu", "gelu"]));
        // Single free op is trivially eligible.
        assert!(fusion_eligible(&["matmul"]));
        // Empty is trivially eligible (vacuous truth — no fusion to do).
        assert!(fusion_eligible(&[]));
    }

    #[test]
    fn fusion_blocked_by_one_bound_op() {
        // Attention output → KV cache append: the append is the
        // fusion barrier. A pass walking this chain must stop at it.
        assert!(!fusion_eligible(&["matmul", "softmax", "kv_cache_append"]));
        // RNG sampling at the end of a generation step is a barrier.
        assert!(!fusion_eligible(&["matmul", "softmax", "sample"]));
        // Unknown ops are treated as bound — fusion blocked.
        assert!(!fusion_eligible(&["matmul", "mystery_op"]));
    }

    #[test]
    fn observe_converges_toward_observation_over_many_samples() {
        // The classical EWMA convergence property: feeding the same
        // observed value indefinitely drives the coefficient arbitrarily
        // close to the value-implied a (with c=b=0 in the prior, observed
        // pj / flops = implied a).
        let mut t = (*priors()).clone();
        let be = BackendId::new("cpu:test");
        for _ in 0..200 {
            t.observe("matmul", &be, 1000, 0, 50_000.0, EnergyBasis::Reported); // implies a = 50
        }
        let (c, b) = t.lookup("matmul", &be);
        assert!((c.a_pj_per_flop - 50.0).abs() < 0.5,
            "after 200 observations a should converge to ~50; got {}", c.a_pj_per_flop);
        assert_eq!(b, EnergyBasis::Reported, "200 counter samples take the meter's basis");
    }
}
