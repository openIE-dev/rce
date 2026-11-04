//! `rce-op-metal` — `OpDispatch` leaves wrapping Metal Shading
//! Language compute kernels, plus the public `dispatch_*` API that
//! `rce-fusion`'s codegen-emitted kernel names dispatch through.
//!
//! ## Convergence with efficient-genai
//!
//! The hand-written MSL shader sources (silu, rms_norm, silu_rms_norm)
//! are **vendored from `efficient-genai/src/hal/metal/shader.rs`** (the
//! working in-house runtime that already runs every model on the user's
//! drive). Vendoring rather than path-importing the whole efficient-genai
//! crate keeps the dependency graph clean while preserving exact
//! byte-for-byte kernel behaviour. The codegen-emitted MSL (matmul
//! variants, FFN-swiglu) comes from `rce-fusion::codegen_*` at runtime
//! — `MetalContext::execute_named` recognises the canonical name and
//! lazily compiles + caches the pipeline.
//!
//! ## Architecture
//!
//! - `MetalContext`: shared `MTLDevice` + `MTLCommandQueue` + a
//!   per-pipeline cache (`HashMap<String, ComputePipelineState>`) +
//!   the Phase S weight upload cache (`HashMap<u64, Buffer>`). One
//!   context per process; reused across calls.
//! - `MetalLeaf`: the public `OpDispatch` type. Claims an op name iff
//!   it's in `handles()` (the hand-written set: `silu`, `rms_norm`,
//!   `silu_rms_norm`) or matches any of the six codegen-name parsers
//!   (each accepting f32 / f16 / bf16 suffixes from Phase O + U):
//!   `codegen` (per-element chains, Phase A),
//!   `codegen_matmul_epilogue` (Phase F), `codegen_matmul_matmul`
//!   (Phase H), `codegen_matmul_tiled` (Phase I), `codegen_ffn_swiglu`
//!   (Phase Q), `codegen_ffn_swiglu_tiled` (Phase R). Phase X wires
//!   `fused_ffn_swiglu_*` and `fused_ffn_swiglu_tiled_*` into
//!   `execute_named`'s prefix-routing match so OpDispatch callers
//!   (graph rewriter, FunctionGraph interpreter, manual `execute(...)`)
//!   reach the same kernels as the direct `dispatch_*` methods.
//! - `HostTensor` I/O covers f32, f16, and bf16 (Phase O). The kernel
//!   library picks `_f32` / `_f16` / `_bf16` variants by suffix on the
//!   canonical name. `Value::DeviceTensor` flow (skip the host
//!   roundtrip when input is already on GPU) is supported on the
//!   input side via `promote_to_device` + `metal_buffer_of`; the
//!   output side still rounds through host bytes.
//!
//! ## Public dispatch_* API (escape hatch around OpDispatch)
//!
//! For callers that already have HostTensors and want a single Metal
//! kernel call without going through the FunctionGraph interpreter:
//!
//! - `dispatch_fused_chain` / `_f16` / `_bf16` — per-element Free chain
//!   (Phase A codegen consumer).
//! - `dispatch_matmul_epilogue` / `_f16` / `_bf16` — Phase F naive
//!   matmul + register epilogue.
//! - `dispatch_matmul_epilogue_cached` / `_f16` / `_bf16` — Phase AA
//!   (f32) + Phase BB (narrow). Same kernel as Phase F / M but weight
//!   comes from a `u64` cache handle (`cache_weight_{f32,f16,bf16}`)
//!   instead of a `&Value`. Drops the per-call weight upload — the
//!   load-once / dispatch-many shape that dominates transformer
//!   inference (down_proj, lm_head, etc.). Handle dtype tag is
//!   validated against the kernel's element dtype.
//! - `dispatch_matmul_tiled` / `_f16` / `_bf16` — Phase I threadgroup-
//!   tiled matmul + epilogue. Best for medium/large shapes.
//! - `dispatch_matmul_tiled_cached` / `_f16` / `_bf16` — Phase CC.
//!   Same tiled kernel, weight via cache handle. The canonical
//!   inference-path kernel for transformer projections: tiled
//!   threadgroup-shared dot products + load-once weight + narrow
//!   storage all in one dispatch.
//! - `dispatch_matmul_auto` — routes a matmul call to Naive or Tiled
//!   via `route_matmul`; the [`MatmulRouting`] enum exposes the choice
//!   so cost models / placement layers can mirror it.
//! - `dispatch_matmul_matmul` / `_f16` / `_bf16` — Phase H chained
//!   `A @ B @ C` + epilogue in one dispatch.
//! - `dispatch_matmul_matmul_cached` / `_f16` / `_bf16` — Phase DD.
//!   Same Phase H kernel; B + C are `u64` handles from cache. A
//!   stays as per-call input. The "two-of-three uploads avoided"
//!   form — useful when both B and C are load-once weights.
//! - `dispatch_ffn_swiglu` / `_f16` / `_bf16` — Phase Q naive
//!   double-matmul FFN swiglu (Phase V adds the narrow variants;
//!   host f32 ↔ narrow on upload + download, accumulators stay f32).
//! - `dispatch_ffn_swiglu_cached` / `_f16` / `_bf16` — Phase EE.
//!   Same naive Phase Q kernel; gate + up come from cache handles.
//!   Completes the cached coverage matrix: every shipped chain
//!   shape × dtype now has both an uncached and a cached form.
//! - `dispatch_ffn_swiglu_tiled` / `_f16` / `_bf16` — Phase R tiled
//!   variant, shared input tile across both matmuls. Threadgroup
//!   tiles stay f32 even when storage is half/bfloat.
//! - `dispatch_ffn_swiglu_tiled_cached` / `_f16` / `_bf16` — weights
//!   addressed by `u64` cache handle instead of `&[f32]` (no per-call
//!   weight upload). f32 form is Phase S; Phase Y adds narrow
//!   variants. The cache tags each handle with its element dtype so
//!   dispatchers refuse cross-dtype use with a clear error.
//!
//! ## Weight upload cache (Phase S + Y)
//!
//! - `cache_weight_f32(&[f32]) → u64` — uploads as f32 storage,
//!   returns a handle.
//! - `cache_weight_f16(&[f32]) → u64` — converts f32 → f16 bytes,
//!   uploads as half storage. Phase Y.
//! - `cache_weight_bf16(&[f32]) → u64` — same shape, bfloat storage.
//!   Phase Y.
//! - `release_weight(u64) → bool` — explicit drop. Works for handles
//!   from any of the three `cache_weight_*` paths.
//! - `cached_weight_count() → usize` — total across dtypes; for
//!   tests / telemetry.
//!
//! Cache entries are tagged with their element dtype, so the
//! `dispatch_*_cached_{f16,bf16}` variants validate the handle was
//! cached at the matching dtype and surface a clear error otherwise.
//!
//! Lifetime is caller-managed (no GC, no LRU). Sized for the
//! "load-once, dispatch many times" pattern that dominates inference.
//!
//! ## Cross-device interop (zero-copy on the input side)
//!
//! - `download_via_host(&DeviceTensor) → Option<HostTensor>` —
//!   universal-host migration path. Any other backend can read a Metal
//!   tensor by going Metal → host bytes → its own upload.
//! - `metal_buffer_of(&DeviceTensor) → Option<&metal::Buffer>` — the
//!   raw `MTLBuffer` for consumers (e.g. `rce-op-mps`) that want to
//!   memcpy / blit / wrap without a host roundtrip. This and
//!   `wrap_metal_buffer` exist only on Apple targets, where `metal` does.
//!
//! ## Cross-platform
//!
//! Non-macOS / non-iOS builds compile to a stub leaf that always
//! returns `Unsupported`, so the dispatch chain falls through to the
//! CPU-reference leaves (e.g. `rce-op-nn-prims`) and the workspace
//! builds cleanly on Linux/Windows. The `dispatch_*` and weight-cache
//! API surface still exists on those targets — every method returns an
//! `Unsupported` error / `0` handle / `0` count so callers don't need
//! `#[cfg]` at every site.

use rce_vm_backend_cpu::{OpDispatch, Result, Value};
#[cfg(not(rce_metal))]
use rce_vm_backend_cpu::InterpretError;

/// Combined Metal-backed `OpDispatch` leaf. Currently claims
/// `Custom("silu")` and `Custom("rms_norm")`. As more kernels get
/// vendored from efficient-genai, register them in
/// [`imp::MetalContext::new`] and add a match arm in
/// [`imp::MetalContext::execute_named`].
pub struct MetalLeaf {
    #[cfg(rce_metal)]
    inner: imp::MetalContext,
}

/// Which matmul template `dispatch_matmul_auto` (and the
/// graph-rewriter `execute_named` path for `fused_matmul_<...>_f32`
/// names) selects for a given shape. See `MetalContext::route_matmul`
/// for the heuristic. Exposed publicly so callers can mirror the
/// routing decision in cost models / placement layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulRouting {
    /// Phase F naive per-output-element kernel. Minimal threading
    /// overhead; best for small shapes.
    Naive,
    /// Phase I threadgroup-tiled kernel with shared `A_tile` +
    /// `B_tile`. Best for medium-to-large shapes where the K-stream
    /// reads dominate.
    Tiled,
}

impl MetalLeaf {
    /// Build a Metal leaf bound to the system default device. Returns
    /// `None` if Metal is unavailable (non-Apple target, headless CI,
    /// sandbox without GPU, MSL compilation failure).
    pub fn new() -> Option<Self> {
        #[cfg(rce_metal)]
        {
            imp::MetalContext::new().map(|inner| Self { inner })
        }
        #[cfg(not(rce_metal))]
        {
            None
        }
    }

    /// Upload a `HostTensor` to GPU memory and return a
    /// `Value::DeviceTensor`. Subsequent kernel dispatches that take
    /// this DeviceTensor as input run **without a host roundtrip** —
    /// the data stays on the GPU across consecutive Metal Apply nodes.
    ///
    /// This is the latency lever: a graph with N consecutive Metal
    /// kernels saves (N-1)·(host→device + device→host) memcpy passes.
    /// For Mamba2 a single mixer block has 5+ consecutive ops that
    /// benefit (silu → ssm_scan → rms_norm → silu → mul), each running
    /// on a 5KB–5MB tensor.
    #[cfg(rce_metal)]
    pub fn promote_to_device(&self, t: &rce_vm_backend_cpu::HostTensor) -> Option<Value> {
        self.inner.promote_to_device(t)
    }

    #[cfg(not(rce_metal))]
    pub fn promote_to_device(&self, _t: &rce_vm_backend_cpu::HostTensor) -> Option<Value> {
        None
    }

    /// Pull a `DeviceTensor` back to host. Used at chain boundaries
    /// (e.g., when the next op is on CPU, or when handing the result
    /// to the caller). Bytes are copied from the shared-storage
    /// MTLBuffer; cheap on Apple silicon (unified memory).
    #[cfg(rce_metal)]
    pub fn demote_to_host(&self, t: &rce_vm_backend_cpu::DeviceTensor) -> Option<Value> {
        self.inner.demote_to_host(t)
    }

    #[cfg(not(rce_metal))]
    pub fn demote_to_host(&self, _t: &rce_vm_backend_cpu::DeviceTensor) -> Option<Value> {
        None
    }

    /// Dispatch a codegen-emitted fused chain. Delegates to
    /// `MetalContext::dispatch_fused_chain`; see its docs for the
    /// buffer-slot contract.
    ///
    /// Returns `Err(Unsupported)` on non-Apple targets (no Metal context).
    #[cfg(rce_metal)]
    pub fn dispatch_fused_chain(
        &self,
        chain: &[&str],
        input: &Value,
        uniforms: &[&Value],
        scalars: &[f32],
        ngroups: u32,
    ) -> Result<Value> {
        self.inner.dispatch_fused_chain(chain, input, uniforms, scalars, ngroups)
    }

    /// Dispatch the f16 variant of a fused chain. Phase K of the
    /// codegen rollout. See `MetalContext::dispatch_fused_chain_f16`
    /// for the conversion + buffer-slot details.
    #[cfg(rce_metal)]
    pub fn dispatch_fused_chain_f16(
        &self,
        chain: &[&str],
        input: &Value,
        uniforms: &[&Value],
        scalars: &[f32],
        ngroups: u32,
    ) -> Result<Value> {
        self.inner.dispatch_fused_chain_f16(chain, input, uniforms, scalars, ngroups)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_fused_chain_f16(
        &self,
        _chain: &[&str], _input: &Value, _uniforms: &[&Value],
        _scalars: &[f32], _ngroups: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal fused-chain f16 dispatch: no Metal device on this target".into(),
        ))
    }

    /// Dispatch the bf16 variant of a fused chain. Phase L of the
    /// codegen rollout. Requires Apple silicon M2 or later (Metal
    /// 3.1+) for the `bfloat` MSL type; older targets surface the
    /// pipeline compile error as `Unsupported`.
    #[cfg(rce_metal)]
    pub fn dispatch_fused_chain_bf16(
        &self,
        chain: &[&str],
        input: &Value,
        uniforms: &[&Value],
        scalars: &[f32],
        ngroups: u32,
    ) -> Result<Value> {
        self.inner.dispatch_fused_chain_bf16(chain, input, uniforms, scalars, ngroups)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_fused_chain_bf16(
        &self,
        _chain: &[&str], _input: &Value, _uniforms: &[&Value],
        _scalars: &[f32], _ngroups: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal fused-chain bf16 dispatch: no Metal device on this target".into(),
        ))
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_fused_chain(
        &self,
        _chain: &[&str],
        _input: &Value,
        _uniforms: &[&Value],
        _scalars: &[f32],
        _ngroups: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal fused-chain dispatch: no Metal device on this target".into(),
        ))
    }

    /// Distinct fused-chain pipelines this leaf has compiled. Test
    /// affordance — proves the codegen pipeline cache absorbs repeats.
    #[cfg(rce_metal)]
    pub fn fused_pipeline_count(&self) -> usize {
        self.inner.fused_pipeline_count()
    }

    #[cfg(not(rce_metal))]
    pub fn fused_pipeline_count(&self) -> usize { 0 }

    /// Dispatch a codegen-emitted `output = epilogue(input @ weight)`
    /// matmul-epilogue kernel. See `MetalContext::dispatch_matmul_epilogue`
    /// for the buffer-slot + arg contract. Phase F of the codegen rollout.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_epilogue(
        &self,
        input: &Value,
        weight: &Value,
        m: u32,
        k: u32,
        n: u32,
        epilogue: &[&str],
        uniforms: &[&Value],
        scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_epilogue(input, weight, m, k, n, epilogue, uniforms, scalars)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_epilogue(
        &self,
        _input: &Value,
        _weight: &Value,
        _m: u32,
        _k: u32,
        _n: u32,
        _epilogue: &[&str],
        _uniforms: &[&Value],
        _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-epilogue dispatch: no Metal device on this target".into(),
        ))
    }

    /// Phase AA: matmul-epilogue dispatch with a cached weight handle.
    /// Same kernel + epilogue contract as `dispatch_matmul_epilogue`;
    /// only the `weight` argument changes from `&Value` to a `u64`
    /// handle from `cache_weight_f32`. Drops the per-call weight
    /// upload — useful for the `load-once / dispatch-many` pattern
    /// that dominates transformer inference (down_proj, lm_head, etc.).
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_epilogue_cached(
        &self,
        input: &Value,
        weight_handle: u64,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_epilogue_cached(
            input, weight_handle, m, k, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_epilogue_cached(
        &self,
        _input: &Value,
        _weight_handle: u64,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-epilogue-cached: no Metal device on this target".into(),
        ))
    }

    /// Phase BB: matmul-epilogue cached + f16 storage. Weight handle
    /// must come from `cache_weight_f16`.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_epilogue_cached_f16(
        &self,
        input: &Value,
        weight_handle: u64,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_epilogue_cached_f16(
            input, weight_handle, m, k, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_epilogue_cached_f16(
        &self,
        _input: &Value,
        _weight_handle: u64,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-epilogue-cached f16: no Metal device on this target".into(),
        ))
    }

    /// Phase BB: matmul-epilogue cached + bf16 storage. Weight handle
    /// must come from `cache_weight_bf16`.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_epilogue_cached_bf16(
        &self,
        input: &Value,
        weight_handle: u64,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_epilogue_cached_bf16(
            input, weight_handle, m, k, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_epilogue_cached_bf16(
        &self,
        _input: &Value,
        _weight_handle: u64,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-epilogue-cached bf16: no Metal device on this target".into(),
        ))
    }

    /// Phase CC: tiled matmul-epilogue with a cached weight handle.
    /// Same threadgroup-tiled kernel as `dispatch_matmul_tiled`; only
    /// the `weight` argument changes from `&Value` to a `u64` handle.
    /// For medium/large K (where the tiled kernel beats naive) and
    /// load-once / dispatch-many weights — the bulk of transformer
    /// projection work.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_tiled_cached(
        &self,
        input: &Value,
        weight_handle: u64,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_tiled_cached(
            input, weight_handle, m, k, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_tiled_cached(
        &self,
        _input: &Value,
        _weight_handle: u64,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-tiled-cached: no Metal device on this target".into(),
        ))
    }

    /// Phase CC: tiled matmul-epilogue cached + f16 storage.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_tiled_cached_f16(
        &self,
        input: &Value,
        weight_handle: u64,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_tiled_cached_f16(
            input, weight_handle, m, k, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_tiled_cached_f16(
        &self,
        _input: &Value,
        _weight_handle: u64,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-tiled-cached f16: no Metal device on this target".into(),
        ))
    }

    /// Phase CC: tiled matmul-epilogue cached + bf16 storage.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_tiled_cached_bf16(
        &self,
        input: &Value,
        weight_handle: u64,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_tiled_cached_bf16(
            input, weight_handle, m, k, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_tiled_cached_bf16(
        &self,
        _input: &Value,
        _weight_handle: u64,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-tiled-cached bf16: no Metal device on this target".into(),
        ))
    }

    /// Dispatch the Phase Q FFN-swiglu kernel: one Metal dispatch
    /// computes `output = silu(input @ gate_W) * (input @ up_W)`.
    /// See `MetalContext::dispatch_ffn_swiglu` for the buffer-slot
    /// contract.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu(
        &self,
        input: &Value, gate_w: &Value, up_w: &Value,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu(input, gate_w, up_w, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu(
        &self,
        _input: &Value, _gate_w: &Value, _up_w: &Value,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu dispatch: no Metal device on this target".into(),
        ))
    }

    /// Phase S: cache an f32 weight buffer on the GPU. Returns a
    /// handle the caller stores and passes to
    /// `dispatch_ffn_swiglu_tiled_cached` (and future
    /// `*_cached` variants). One upload, many reuses.
    #[cfg(rce_metal)]
    pub fn cache_weight_f32(&self, data: &[f32]) -> u64 {
        self.inner.cache_weight_f32(data)
    }

    #[cfg(not(rce_metal))]
    pub fn cache_weight_f32(&self, _data: &[f32]) -> u64 { 0 }

    /// Phase Y: cache an f32 weight as f16 storage. Same handle
    /// lifecycle as `cache_weight_f32`; the returned handle is only
    /// valid for the `_cached_f16` dispatch variants. Conversion is
    /// done once at cache time.
    #[cfg(rce_metal)]
    pub fn cache_weight_f16(&self, data: &[f32]) -> u64 {
        self.inner.cache_weight_f16(data)
    }

    #[cfg(not(rce_metal))]
    pub fn cache_weight_f16(&self, _data: &[f32]) -> u64 { 0 }

    /// Phase Y: cache an f32 weight as bf16 storage. Requires Apple
    /// silicon M2+ for the actual narrow dispatch (the cache itself
    /// works everywhere).
    #[cfg(rce_metal)]
    pub fn cache_weight_bf16(&self, data: &[f32]) -> u64 {
        self.inner.cache_weight_bf16(data)
    }

    #[cfg(not(rce_metal))]
    pub fn cache_weight_bf16(&self, _data: &[f32]) -> u64 { 0 }

    /// Phase S: release a cached weight. Returns `true` if the handle
    /// existed, `false` otherwise.
    #[cfg(rce_metal)]
    pub fn release_weight(&self, handle: u64) -> bool {
        self.inner.release_weight(handle)
    }

    #[cfg(not(rce_metal))]
    pub fn release_weight(&self, _handle: u64) -> bool { false }

    /// Number of cached weights. Test affordance for the cache-hit /
    /// cache-grow invariants.
    #[cfg(rce_metal)]
    pub fn cached_weight_count(&self) -> usize {
        self.inner.cached_weight_count()
    }

    #[cfg(not(rce_metal))]
    pub fn cached_weight_count(&self) -> usize { 0 }

    /// Phase EE: naive FFN-swiglu (Phase Q) with cached weights.
    /// Gate + up handles must come from `cache_weight_f32`.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_cached(
        &self,
        input: &Value,
        gate_handle: u64, up_handle: u64,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_cached(input, gate_handle, up_handle, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_cached(
        &self,
        _input: &Value,
        _gate_handle: u64, _up_handle: u64,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-cached: no Metal device on this target".into(),
        ))
    }

    /// Phase EE: naive FFN-swiglu cached + f16 storage.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_cached_f16(
        &self,
        input: &Value,
        gate_handle: u64, up_handle: u64,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_cached_f16(input, gate_handle, up_handle, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_cached_f16(
        &self,
        _input: &Value,
        _gate_handle: u64, _up_handle: u64,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-cached f16: no Metal device on this target".into(),
        ))
    }

    /// Phase EE: naive FFN-swiglu cached + bf16 storage.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_cached_bf16(
        &self,
        input: &Value,
        gate_handle: u64, up_handle: u64,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_cached_bf16(input, gate_handle, up_handle, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_cached_bf16(
        &self,
        _input: &Value,
        _gate_handle: u64, _up_handle: u64,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-cached bf16: no Metal device on this target".into(),
        ))
    }

    /// Phase S: tiled FFN-swiglu with cached weight buffers. See
    /// `MetalContext::dispatch_ffn_swiglu_tiled_cached`.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_tiled_cached(
        &self,
        input: &Value,
        gate_handle: u64, up_handle: u64,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_tiled_cached(input, gate_handle, up_handle, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_tiled_cached(
        &self,
        _input: &Value,
        _gate_handle: u64, _up_handle: u64,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-tiled-cached: no Metal device on this target".into(),
        ))
    }

    /// Phase Y: tiled FFN-swiglu with f16-cached weights.
    /// Handles must come from `cache_weight_f16`.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_tiled_cached_f16(
        &self,
        input: &Value,
        gate_handle: u64, up_handle: u64,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_tiled_cached_f16(input, gate_handle, up_handle, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_tiled_cached_f16(
        &self,
        _input: &Value,
        _gate_handle: u64, _up_handle: u64,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-tiled-cached f16: no Metal device on this target".into(),
        ))
    }

    /// Phase Y: tiled FFN-swiglu with bf16-cached weights.
    /// Handles must come from `cache_weight_bf16`.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_tiled_cached_bf16(
        &self,
        input: &Value,
        gate_handle: u64, up_handle: u64,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_tiled_cached_bf16(input, gate_handle, up_handle, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_tiled_cached_bf16(
        &self,
        _input: &Value,
        _gate_handle: u64, _up_handle: u64,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-tiled-cached bf16: no Metal device on this target".into(),
        ))
    }

    /// Phase R tiled FFN-swiglu dispatch. See
    /// `MetalContext::dispatch_ffn_swiglu_tiled` for the shape +
    /// threadgroup contract.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_tiled(
        &self,
        input: &Value, gate_w: &Value, up_w: &Value,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_tiled(input, gate_w, up_w, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_tiled(
        &self,
        _input: &Value, _gate_w: &Value, _up_w: &Value,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-tiled dispatch: no Metal device on this target".into(),
        ))
    }

    /// Phase V f16 naive FFN-swiglu. See
    /// `MetalContext::dispatch_ffn_swiglu_f16` for the conversion
    /// contract.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_f16(
        &self,
        input: &Value, gate_w: &Value, up_w: &Value,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_f16(input, gate_w, up_w, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_f16(
        &self,
        _input: &Value, _gate_w: &Value, _up_w: &Value,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu f16 dispatch: no Metal device on this target".into(),
        ))
    }

    /// Phase V bf16 naive FFN-swiglu. Requires Apple silicon M2+.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_bf16(
        &self,
        input: &Value, gate_w: &Value, up_w: &Value,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_bf16(input, gate_w, up_w, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_bf16(
        &self,
        _input: &Value, _gate_w: &Value, _up_w: &Value,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu bf16 dispatch: no Metal device on this target".into(),
        ))
    }

    /// Phase V f16 tiled FFN-swiglu.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_tiled_f16(
        &self,
        input: &Value, gate_w: &Value, up_w: &Value,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_tiled_f16(input, gate_w, up_w, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_tiled_f16(
        &self,
        _input: &Value, _gate_w: &Value, _up_w: &Value,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-tiled f16 dispatch: no Metal device on this target".into(),
        ))
    }

    /// Phase V bf16 tiled FFN-swiglu.
    #[cfg(rce_metal)]
    pub fn dispatch_ffn_swiglu_tiled_bf16(
        &self,
        input: &Value, gate_w: &Value, up_w: &Value,
        m: u32, k: u32, n: u32,
    ) -> Result<Value> {
        self.inner.dispatch_ffn_swiglu_tiled_bf16(input, gate_w, up_w, m, k, n)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_ffn_swiglu_tiled_bf16(
        &self,
        _input: &Value, _gate_w: &Value, _up_w: &Value,
        _m: u32, _k: u32, _n: u32,
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal ffn-swiglu-tiled bf16 dispatch: no Metal device on this target".into(),
        ))
    }

    /// f16 matmul-epilogue dispatch. Phase M of the codegen rollout.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_epilogue_f16(
        &self,
        input: &Value, weight: &Value,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_epilogue_f16(input, weight, m, k, n, epilogue, uniforms, scalars)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_epilogue_f16(
        &self,
        _input: &Value, _weight: &Value,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-epilogue f16: no Metal device on this target".into(),
        ))
    }

    /// bf16 matmul-epilogue dispatch. Phase M.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_epilogue_bf16(
        &self,
        input: &Value, weight: &Value,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_epilogue_bf16(input, weight, m, k, n, epilogue, uniforms, scalars)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_epilogue_bf16(
        &self,
        _input: &Value, _weight: &Value,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-epilogue bf16: no Metal device on this target".into(),
        ))
    }

    /// Dispatch a codegen-emitted `output = epilogue(A @ B @ C)` kernel.
    /// Phase H of the codegen rollout. See
    /// `MetalContext::dispatch_matmul_matmul` for the buffer-slot
    /// contract; this is the public `MetalLeaf` delegate.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_matmul(
        &self,
        a: &Value,
        b: &Value,
        c: &Value,
        m: u32,
        k1: u32,
        k2: u32,
        n: u32,
        epilogue: &[&str],
        uniforms: &[&Value],
        scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_matmul(a, b, c, m, k1, k2, n, epilogue, uniforms, scalars)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_matmul(
        &self,
        _a: &Value, _b: &Value, _c: &Value,
        _m: u32, _k1: u32, _k2: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-matmul dispatch: no Metal device on this target".into(),
        ))
    }

    /// f16 double-matmul dispatch. Phase N.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_matmul_f16(
        &self, a: &Value, b: &Value, c: &Value,
        m: u32, k1: u32, k2: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_matmul_f16(a, b, c, m, k1, k2, n, epilogue, uniforms, scalars)
    }
    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_matmul_f16(
        &self, _a: &Value, _b: &Value, _c: &Value,
        _m: u32, _k1: u32, _k2: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul² f16: no Metal device".into()))
    }

    /// bf16 double-matmul dispatch. Phase N.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_matmul_bf16(
        &self, a: &Value, b: &Value, c: &Value,
        m: u32, k1: u32, k2: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_matmul_bf16(a, b, c, m, k1, k2, n, epilogue, uniforms, scalars)
    }
    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_matmul_bf16(
        &self, _a: &Value, _b: &Value, _c: &Value,
        _m: u32, _k1: u32, _k2: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul² bf16: no Metal device".into()))
    }

    /// Phase DD: double-matmul (Phase H) with cached B + C handles.
    /// A is per-call input; B + C come from `cache_weight_f32`.
    /// Same `output = epilogue(A @ B @ C)` kernel; drops two of the
    /// three per-call uploads.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_matmul_cached(
        &self,
        a: &Value,
        b_handle: u64, c_handle: u64,
        m: u32, k1: u32, k2: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_matmul_cached(
            a, b_handle, c_handle, m, k1, k2, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_matmul_cached(
        &self,
        _a: &Value,
        _b_handle: u64, _c_handle: u64,
        _m: u32, _k1: u32, _k2: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul-matmul-cached: no Metal device".into()))
    }

    /// Phase DD: double-matmul cached + f16 storage.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_matmul_cached_f16(
        &self,
        a: &Value,
        b_handle: u64, c_handle: u64,
        m: u32, k1: u32, k2: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_matmul_cached_f16(
            a, b_handle, c_handle, m, k1, k2, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_matmul_cached_f16(
        &self,
        _a: &Value,
        _b_handle: u64, _c_handle: u64,
        _m: u32, _k1: u32, _k2: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul-matmul-cached f16: no Metal device".into()))
    }

    /// Phase DD: double-matmul cached + bf16 storage.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_matmul_cached_bf16(
        &self,
        a: &Value,
        b_handle: u64, c_handle: u64,
        m: u32, k1: u32, k2: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_matmul_cached_bf16(
            a, b_handle, c_handle, m, k1, k2, n, epilogue, uniforms, scalars,
        )
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_matmul_cached_bf16(
        &self,
        _a: &Value,
        _b_handle: u64, _c_handle: u64,
        _m: u32, _k1: u32, _k2: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul-matmul-cached bf16: no Metal device".into()))
    }

    /// Auto-route matmul + epilogue dispatch. Picks naive vs tiled
    /// via `route_matmul` and calls the corresponding low-level
    /// dispatcher. This is the entry point the graph-rewriter
    /// Phase G path uses via `execute_named`.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_auto(
        &self,
        input: &Value,
        weight: &Value,
        m: u32,
        k: u32,
        n: u32,
        epilogue: &[&str],
        uniforms: &[&Value],
        scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_auto(input, weight, m, k, n, epilogue, uniforms, scalars)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_auto(
        &self,
        _input: &Value, _weight: &Value,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-auto dispatch: no Metal device on this target".into(),
        ))
    }

    /// Which matmul template the auto-router would pick for a given
    /// shape. Useful for tests + cost models that need to mirror the
    /// routing decision without actually dispatching.
    pub fn route_matmul(m: u32, k: u32, n: u32) -> MatmulRouting {
        #[cfg(rce_metal)]
        { imp::MetalContext::route_matmul(m, k, n) }
        #[cfg(not(rce_metal))]
        {
            // On non-Metal targets the routing decision is academic;
            // return Naive as a default that matches the Phase F
            // baseline.
            let _ = (m, k, n);
            MatmulRouting::Naive
        }
    }

    /// Dispatch a codegen-emitted tiled matmul + epilogue kernel.
    /// Phase I of the codegen rollout. See
    /// `MetalContext::dispatch_matmul_tiled` for the threadgroup
    /// layout + safety contract.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_tiled(
        &self,
        input: &Value,
        weight: &Value,
        m: u32,
        k: u32,
        n: u32,
        epilogue: &[&str],
        uniforms: &[&Value],
        scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_tiled(input, weight, m, k, n, epilogue, uniforms, scalars)
    }

    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_tiled(
        &self,
        _input: &Value, _weight: &Value,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported(
            "metal matmul-tiled dispatch: no Metal device on this target".into(),
        ))
    }

    /// f16 tiled-matmul dispatch. Phase N.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_tiled_f16(
        &self, input: &Value, weight: &Value,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_tiled_f16(input, weight, m, k, n, epilogue, uniforms, scalars)
    }
    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_tiled_f16(
        &self, _input: &Value, _weight: &Value,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul-tiled f16: no Metal device".into()))
    }

    /// bf16 tiled-matmul dispatch. Phase N.
    #[cfg(rce_metal)]
    pub fn dispatch_matmul_tiled_bf16(
        &self, input: &Value, weight: &Value,
        m: u32, k: u32, n: u32,
        epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
    ) -> Result<Value> {
        self.inner.dispatch_matmul_tiled_bf16(input, weight, m, k, n, epilogue, uniforms, scalars)
    }
    #[cfg(not(rce_metal))]
    pub fn dispatch_matmul_tiled_bf16(
        &self, _input: &Value, _weight: &Value,
        _m: u32, _k: u32, _n: u32,
        _epilogue: &[&str], _uniforms: &[&Value], _scalars: &[f32],
    ) -> Result<Value> {
        Err(InterpretError::Unsupported("metal matmul-tiled bf16: no Metal device".into()))
    }
}

impl OpDispatch for MetalLeaf {
    fn handles(&self) -> &[&'static str] {
        &[
            "silu",
            "rms_norm",
            // The first fused-Free-chain op: rewrites
            // `silu → rms_norm` into one Metal dispatch. Same args
            // shape as `rms_norm` (x, weight, ngroups, eps). See
            // `rce-placement::fusion_eligible` for the safety
            // predicate that justifies this rewrite.
            "silu_rms_norm",
            "gelu",
            "swiglu",
            "layer_norm",
            "attention",
            "embedding_lookup",
            "moe_combine",
            "group_norm",
            "topk_mask",
            "argmax",
            "conv2d",
        ]
    }

    /// In addition to the static `handles()` list, this leaf claims any
    /// codegen-emitted fused-chain name (`fused_<...>_f32`) whose body
    /// parses back to a chain via `rce_fusion::codegen::parse_function_name`.
    /// That set is unbounded in principle (grows with the fragment table)
    /// but bounded in practice by the fragments actually registered.
    fn claims(&self, name: &str) -> bool {
        if self.handles().iter().any(|h| *h == name) { return true; }
        if rce_fusion::codegen::parse_function_name(name).is_some() { return true; }
        // Phase F: matmul-epilogue codegen names are a separate family.
        if rce_fusion::codegen_matmul_epilogue::parse_function_name(name).is_some() { return true; }
        // Phase H: double-matmul codegen names are yet another family.
        if rce_fusion::codegen_matmul_matmul::parse_function_name(name).is_some() { return true; }
        // Phase I: tiled matmul + epilogue codegen names.
        if rce_fusion::codegen_matmul_tiled::parse_function_name(name).is_some() { return true; }
        // Phase Q + U: FFN-swiglu dual-matmul kernel — accept f32 /
        // f16 / bf16 canonical names now that execute_named routes
        // them through `dispatch_ffn_swiglu_by_name` (Phase X).
        if rce_fusion::codegen_ffn_swiglu::parse_function_name_with_dtype(name).is_some() { return true; }
        // Phase R + U: tiled FFN-swiglu variant — same story.
        rce_fusion::codegen_ffn_swiglu_tiled::parse_function_name_with_dtype(name).is_some()
    }

    fn execute(&self, _node_id: usize, name: &str, args: &[Value]) -> Result<Value> {
        #[cfg(rce_metal)]
        {
            self.inner.execute_named(name, args)
        }
        #[cfg(not(rce_metal))]
        {
            let _ = (name, args);
            Err(InterpretError::Unsupported(
                "rce-op-metal: not built with Metal support".into(),
            ))
        }
    }
}

/// Backward-compatible alias: the original `MetalSiluLeaf` from the
/// first iteration is now a synonym for `MetalLeaf`. Keeps any
/// downstream code that imported it compiling.
pub type MetalSiluLeaf = MetalLeaf;

/// Cross-device transfer entry point: download a Metal `DeviceTensor`'s
/// bytes to host. Returns `None` if the input isn't a Metal tensor or
/// wasn't produced by `MetalLeaf`. Pairs with
/// `MetalLeaf::promote_to_device` (the upload direction).
///
/// **Cross-device transfer pattern**: a CUDA→Metal migration is
/// `rce_op_cuda::download_via_host(&cuda_dt) → bytes → metal_leaf
/// .promote_to_device(&host_tensor)`. The host is the universal
/// interconnect; per-device-pair direct paths can be added as
/// optimisations later (CUDA P2P, multi-GPU Metal IOSurface, etc.).
#[cfg(rce_metal)]
pub fn download_via_host(
    src: &rce_vm_backend_cpu::DeviceTensor,
) -> Option<rce_vm_backend_cpu::HostTensor> {
    use rce_vm_backend_cpu::{DeviceId, DownloadToHost, HostTensor};
    if src.device != DeviceId::Metal {
        return None;
    }
    let metal_mem = src.mem.as_any().downcast_ref::<imp::MetalDeviceMem>()?;
    let bytes = metal_mem.download_bytes()?;
    HostTensor::new(src.shape.clone(), src.dtype, bytes).ok()
}

#[cfg(not(rce_metal))]
pub fn download_via_host(
    _src: &rce_vm_backend_cpu::DeviceTensor,
) -> Option<rce_vm_backend_cpu::HostTensor> {
    None
}

/// Borrow the shared-storage Metal `Buffer` underlying a Metal
/// `DeviceTensor`. Returns `None` if the tensor isn't on Metal or wasn't
/// produced by `MetalLeaf`.
///
/// This is the **zero-copy interop surface** for other Metal-using crates
/// (e.g. `rce-op-mps`): the returned buffer is the *exact* `MTLBuffer`
/// MetalLeaf wrote into, so consumers can `contents()`-memcpy it into
/// their own buffers, issue a Metal blit, or wrap it in an MPS dispatch
/// — no host roundtrip on the input side. Pairs with
/// `MetalLeaf::promote_to_device` (the input-side upload).
#[cfg(rce_metal)]
pub fn metal_buffer_of(
    src: &rce_vm_backend_cpu::DeviceTensor,
) -> Option<&metal::Buffer> {
    use rce_vm_backend_cpu::DeviceId;
    if src.device != DeviceId::Metal {
        return None;
    }
    let metal_mem = src.mem.as_any().downcast_ref::<imp::MetalDeviceMem>()?;
    Some(&metal_mem.buffer)
}

/// Wrap an existing Metal `Buffer` as a `Value::DeviceTensor` on
/// `DeviceId::Metal`. The reverse of `metal_buffer_of`: a producer
/// (e.g. rce-op-mps's matmul) finishes writing its output buffer, then
/// hands it to the wire as a device-resident `Value` instead of
/// downloading to a `HostTensor`. The next consumer (`MetalLeaf`, MPS,
/// any Metal-aware backend) borrows the buffer back via
/// `metal_buffer_of` — zero host roundtrip.
#[cfg(rce_metal)]
pub fn wrap_metal_buffer(
    buffer: metal::Buffer,
    shape: Vec<usize>,
    dtype: rce_vm_backend_cpu::DType,
) -> rce_vm_backend_cpu::Value {
    use std::sync::Arc;
    use rce_vm_backend_cpu::{DeviceId, DeviceTensor, Value};
    let size_bytes: usize = shape.iter().product::<usize>() * dtype.byte_size();
    let mem = Arc::new(imp::MetalDeviceMem { buffer, size_bytes });
    Value::DeviceTensor(DeviceTensor::new(DeviceId::Metal, dtype, shape, mem))
}

// ── macOS / iOS implementation ─────────────────────────────────────

#[cfg(rce_metal)]
mod imp {
    use crate::MatmulRouting;
    use rce_vm_backend_cpu::{
        DType, DeviceId, DeviceMem, DeviceTensor, HostTensor, InterpretError, Result,
        ScalarValue, Value,
    };
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device,
        MTLResourceOptions, MTLSize,
    };
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use tracing::trace;

    /// Owned `metal::Buffer` wrapper that implements
    /// [`rce_vm_backend_cpu::DeviceMem`]. Created when a Metal kernel
    /// is asked to return a `Value::DeviceTensor` instead of copying
    /// bytes back to host. `Drop` releases the underlying `MTLBuffer`
    /// when the last `Arc` clone goes out of scope (metal-rs's
    /// `Buffer` impls `Drop`-via-`release` already).
    pub struct MetalDeviceMem {
        pub buffer: Buffer,
        pub size_bytes: usize,
    }

    // SAFETY: Metal `Buffer` is documented thread-safe per Apple's
    // contract. We never mutate the buffer post-creation outside of
    // the kernel encoder — which itself is bound to a single command
    // buffer, never shared across threads mid-encode.
    unsafe impl Send for MetalDeviceMem {}
    unsafe impl Sync for MetalDeviceMem {}

    impl DeviceMem for MetalDeviceMem {
        fn ptr(&self) -> u64 {
            // foreign_types::ForeignType raw pointer = MTLBuffer object.
            use metal::foreign_types::ForeignType;
            self.buffer.as_ptr() as u64
        }
        fn size_bytes(&self) -> usize {
            self.size_bytes
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl rce_vm_backend_cpu::DownloadToHost for MetalDeviceMem {
        fn download_bytes(&self) -> Option<Vec<u8>> {
            if self.size_bytes == 0 {
                return Some(Vec::new());
            }
            let ptr = self.buffer.contents() as *const u8;
            if ptr.is_null() {
                return None;
            }
            // SAFETY: shared-storage MTLBuffer guarantees the contents
            // pointer is valid for the lifetime of the Buffer; we
            // hold &self so the Buffer is live.
            Some(unsafe { std::slice::from_raw_parts(ptr, self.size_bytes).to_vec() })
        }
    }

    /// Vendored MSL sources, name → source. Each kernel function
    /// inside the source is registered separately; multiple kernels
    /// sharing one MSL source string (like SiLU's f16+f32) are fine.
    const SILU_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    uint gid                   [[thread_position_in_grid]]
) {
    float x = input[gid];
    output[gid] = x / (1.0f + exp(-x));
}
"#;

    /// `gelu_tanh_f32` — tanh-approximation GeLU. Vendored from
    /// `efficient-genai/src/hal/metal/shader.rs::sources::GELU` lines
    /// 597-611. Used by Gemma, BERT, GPT-2 and other archs that
    /// matched PyTorch's `gelu_pytorch_tanh`. Single tensor in/out.
    const GELU_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void gelu_tanh_f32(
    device const float* input  [[buffer(0)]],
    device       float* output [[buffer(1)]],
    uint gid                   [[thread_position_in_grid]]
) {
    float x = input[gid];
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    inner = clamp(inner, -10.0f, 10.0f);
    output[gid] = 0.5f * x * (1.0f + tanh(inner));
}
"#;

    /// `swiglu_f32` — SiLU(gate) * up. Vendored & adapted (f16→f32)
    /// from `efficient-genai/src/hal/metal/shader.rs::sources::SWIGLU`
    /// lines 1626-1637. Used by Llama 2/3, Mistral, Mixtral. Two input
    /// tensors (gate, up), one output tensor.
    const SWIGLU_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void swiglu_f32(
    device const float* gate   [[buffer(0)]],
    device const float* up     [[buffer(1)]],
    device       float* output [[buffer(2)]],
    uint gid                   [[thread_position_in_grid]]
) {
    float g = gate[gid];
    float u = up[gid];
    float silu_g = g / (1.0f + exp(-g));
    output[gid] = silu_g * u;
}
"#;

    /// `layer_norm_f32` — standard LayerNorm (mean + variance, scale + shift).
    /// Vendored & adapted (f16→f32) from
    /// `efficient-genai/src/hal/metal/shader.rs::sources::LAYER_NORM`
    /// lines 330-365. Used by older transformers (BERT, GPT-2) and by
    /// Stable Diffusion's UNet. Whole-vector by default; if `ngroups>1`
    /// the kernel normalises each group independently (analogous to
    /// `rms_norm_f32_grouped` but with mean+variance instead of RMS).
    const LAYER_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void layer_norm_f32_grouped(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device       float* output [[buffer(3)]],
    constant uint&  ngroups    [[buffer(4)]],
    constant uint&  group_size [[buffer(5)]],
    constant float& eps        [[buffer(6)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= ngroups) return;

    uint base = gid * group_size;

    float sum = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        sum += input[base + i];
    }
    float mean = sum / float(group_size);

    float var_sum = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        float d = input[base + i] - mean;
        var_sum += d * d;
    }
    float inv_std = rsqrt(var_sum / float(group_size) + eps);

    for (uint i = 0; i < group_size; i++) {
        float n = (input[base + i] - mean) * inv_std;
        output[base + i] = n * weight[base + i] + bias[base + i];
    }
}
"#;

    /// `argmax_f32` — find the index of the maximum element in a 1-D
    /// vector. Used for greedy LLM token sampling. Adapted from
    /// `efficient-genai/src/hal/metal/shader.rs::sources::ARGMAX`
    /// `argmax_f16` (lines 1653-1700) into an f32 single-thread linear
    /// scan; the parallel-reduction variant is 5-10× faster on large
    /// vocab sizes but adds threadgroup-memory complexity. For sampling
    /// at vocab=128K this dispatches a single thread; the work is ~120K
    /// FP32 compares, ≤200 µs on M3 Ultra. Promote to parallel-reduction
    /// when sampling becomes the hot path.
    const ARGMAX_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void argmax_f32(
    device const float* input [[buffer(0)]],
    device       uint*  output [[buffer(1)]],
    constant uint&  size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid != 0) return;
    float max_val = -INFINITY;
    uint  max_idx = 0;
    for (uint i = 0; i < size; i++) {
        float v = input[i];
        if (v > max_val) {
            max_val = v;
            max_idx = i;
        }
    }
    output[0] = max_idx;
}
"#;

    /// `conv2d_naive_f32` — 2-D convolution with NHWC input + OIHW
    /// weight layout. Adapted from
    /// `efficient-genai/src/hal/metal/shader.rs::sources::CONV2D`
    /// `conv2d_naive_f16` into f32. Naive shape-agnostic kernel: each
    /// thread computes one output element. Slow vs the 3x3 tiled +
    /// 1x1 simdgroup variants for hot UNet kernels; sufficient for the
    /// demo and for any conv2d shape efficient-genai's specialised
    /// kernels don't cover.
    ///
    /// Layouts:
    /// - input:   [N, H, W, C_in]    (NHWC, contiguous on C_in)
    /// - weight:  [C_out, K_h, K_w, C_in] (OHWI per-output-channel)
    /// - bias:    [C_out]            (optional — pass zero if unused)
    /// - output:  [N, H_out, W_out, C_out]
    /// - stride:  (s_h, s_w)
    /// - padding: (p_h, p_w)
    /// - H_out = (H + 2 p_h - K_h) / s_h + 1, similarly W_out.
    const CONV2D_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void conv2d_naive_f32(
    device const float* input    [[buffer(0)]],
    device const float* weight   [[buffer(1)]],
    device const float* bias     [[buffer(2)]],
    device       float* output   [[buffer(3)]],
    constant uint&  N            [[buffer(4)]],
    constant uint&  H            [[buffer(5)]],
    constant uint&  W            [[buffer(6)]],
    constant uint&  C_in         [[buffer(7)]],
    constant uint&  C_out        [[buffer(8)]],
    constant uint&  K_h          [[buffer(9)]],
    constant uint&  K_w          [[buffer(10)]],
    constant uint&  S_h          [[buffer(11)]],
    constant uint&  S_w          [[buffer(12)]],
    constant uint&  P_h          [[buffer(13)]],
    constant uint&  P_w          [[buffer(14)]],
    constant uint&  H_out        [[buffer(15)]],
    constant uint&  W_out        [[buffer(16)]],
    uint3 gid                    [[thread_position_in_grid]]
) {
    uint c_out = gid.x;
    uint w_out = gid.y;
    uint hn    = gid.z;
    if (c_out >= C_out || w_out >= W_out) return;
    uint n     = hn / H_out;
    uint h_out = hn - n * H_out;
    if (n >= N) return;

    float acc = bias[c_out];
    for (uint kh = 0; kh < K_h; kh++) {
        int h_in = int(h_out * S_h) + int(kh) - int(P_h);
        if (h_in < 0 || (uint)h_in >= H) continue;
        for (uint kw = 0; kw < K_w; kw++) {
            int w_in = int(w_out * S_w) + int(kw) - int(P_w);
            if (w_in < 0 || (uint)w_in >= W) continue;
            for (uint cin = 0; cin < C_in; cin++) {
                uint in_idx = ((n * H + (uint)h_in) * W + (uint)w_in) * C_in + cin;
                uint w_idx  = ((c_out * K_h + kh) * K_w + kw) * C_in + cin;
                acc += input[in_idx] * weight[w_idx];
            }
        }
    }
    uint out_idx = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out;
    output[out_idx] = acc;
}
"#;

    /// `group_norm_f32` — channel-grouped image normalization. Used
    /// by Stable Diffusion / SDXL UNet + VAE. Adapted from
    /// `efficient-genai/src/hal/metal/shader.rs::sources::GROUP_NORM`
    /// `group_norm_stats_f16` + `group_norm_apply_f16` (two-pass) into
    /// a single-pass f32 kernel. Each thread handles one `(n, g)`
    /// pair: it reads the `(C/G) * HW` elements of its group, computes
    /// mean+variance, then writes all `(C/G) * HW` outputs scaled by
    /// gamma and shifted by bias.
    ///
    /// One thread per group is fine for SDXL-class shapes (N=1, G=32,
    /// HW≤64²). For very large HW the two-pass pattern is faster; ship
    /// when SDXL/Flux end-to-end exercises it.
    ///
    /// Args (host side, expanded into kernel constants):
    ///   x      f32 [N, C, HW]
    ///   gamma  f32 [C]   (per-channel scale)
    ///   bias   f32 [C]   (per-channel shift)
    ///   N, G, C, HW: i32 scalars
    ///   eps:    f32 scalar
    /// Output: f32 [N, C, HW].
    const GROUP_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void group_norm_f32(
    device const float* input  [[buffer(0)]],
    device const float* gamma  [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device       float* output [[buffer(3)]],
    constant uint&  N          [[buffer(4)]],
    constant uint&  G          [[buffer(5)]],
    constant uint&  C          [[buffer(6)]],
    constant uint&  HW         [[buffer(7)]],
    constant float& eps        [[buffer(8)]],
    uint2 gid                  [[thread_position_in_grid]]
) {
    uint g = gid.x;
    uint n = gid.y;
    if (g >= G || n >= N) return;

    uint c_per_g = C / G;
    uint num_elems = c_per_g * HW;
    uint base_c = g * c_per_g;
    uint nchw_base = n * C * HW;

    // First pass: mean and variance over the group.
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (uint cl = 0; cl < c_per_g; cl++) {
        uint c = base_c + cl;
        uint c_off = nchw_base + c * HW;
        for (uint hw = 0; hw < HW; hw++) {
            float v = input[c_off + hw];
            sum += v;
            sum_sq += v * v;
        }
    }
    float mean = sum / float(num_elems);
    float var = (sum_sq / float(num_elems)) - mean * mean;
    float inv_std = rsqrt(max(var, 0.0f) + eps);

    // Second pass: apply.
    for (uint cl = 0; cl < c_per_g; cl++) {
        uint c = base_c + cl;
        uint c_off = nchw_base + c * HW;
        float gm = gamma[c];
        float bt = bias[c];
        for (uint hw = 0; hw < HW; hw++) {
            float v = input[c_off + hw];
            output[c_off + hw] = (v - mean) * inv_std * gm + bt;
        }
    }
}
"#;

    /// `topk_mask_f32` — given logits `[N]` and `top_k`, write a binary
    /// mask `[N]` with 1.0 at the top-K positions and 0.0 elsewhere.
    /// The math primitive for sparse-MoE evaluation at the IR level —
    /// composed with a multiply node, the mask zero-weights non-routed
    /// experts, and an IR-level optimizer can later detect "expert
    /// times zero" and skip its computation entirely.
    ///
    /// Top-K selection is computed host-side (small N, requires
    /// sort-like behavior); the GPU only does the buffer write.
    /// Currently the host-side selection happens in the dispatch
    /// fn, so the GPU pipeline is just `output[i] = (i in topk_set) ? 1 : 0`.
    /// We pass the topk_indices array as a buffer.
    const TOPK_MASK_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void topk_mask_f32(
    device const uint*  topk_indices [[buffer(0)]],
    device       float* output       [[buffer(1)]],
    constant uint&  n                [[buffer(2)]],
    constant uint&  top_k            [[buffer(3)]],
    uint gid                         [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    // Linear scan over top_k (small).
    bool in_topk = false;
    for (uint k = 0; k < top_k; k++) {
        if (topk_indices[k] == gid) { in_topk = true; break; }
    }
    output[gid] = in_topk ? 1.0f : 0.0f;
}
"#;

    /// `embedding_lookup_f32` — gather rows from the embedding table.
    /// Adapted from `efficient-genai/src/hal/metal/shader.rs::sources::EMBEDDING`
    /// `embedding_lookup_f16` (lines 1710-1731) → f32. Defensive OOV
    /// (token_id ≥ vocab_size or negative) writes a zero row.
    const EMBEDDING_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void embedding_lookup_f32(
    device const float* embed_table [[buffer(0)]],
    device const int*   token_ids   [[buffer(1)]],
    device       float* output      [[buffer(2)]],
    constant uint&  vocab_size      [[buffer(3)]],
    constant uint&  hidden_size     [[buffer(4)]],
    constant uint&  seq_len         [[buffer(5)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= seq_len) return;
    int tok = token_ids[gid];
    device float* dst = output + gid * hidden_size;
    if (tok < 0 || (uint)tok >= vocab_size) {
        for (uint i = 0; i < hidden_size; i++) {
            dst[i] = 0.0f;
        }
        return;
    }
    device const float* src = embed_table + (uint)tok * hidden_size;
    for (uint i = 0; i < hidden_size; i++) {
        dst[i] = src[i];
    }
}
"#;

    /// `moe_combine_f32` — top-K weighted combination of expert outputs.
    /// Original (no efficient-genai source — MoE combine math is
    /// arch-level, not in efficient-genai's per-kernel HAL).
    /// Each thread handles one output element `output[d]`. The thread
    /// scans the gate_logits to find top-K, computes softmax weights,
    /// and accumulates the weighted sum. Wasteful (top-K and softmax
    /// duplicated D times) but correct; threadgroup-shared variant is
    /// a follow-up.
    ///
    /// For the dispatch we pass the top-K indices and weights through
    /// constant buffers (computed host-side) so the GPU only does the
    /// reduce. This is the more standard MoE Metal kernel pattern and
    /// keeps the kernel dispatch-grid-shaped.
    const MOE_COMBINE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void moe_combine_f32(
    device const float* expert_outputs [[buffer(0)]],   // [E, D]
    device const uint*  topk_indices   [[buffer(1)]],   // [K]
    device const float* topk_weights   [[buffer(2)]],   // [K]
    device       float* output         [[buffer(3)]],   // [D]
    constant uint&  d_model            [[buffer(4)]],
    constant uint&  top_k              [[buffer(5)]],
    uint gid                           [[thread_position_in_grid]]
) {
    if (gid >= d_model) return;
    float acc = 0.0f;
    for (uint k = 0; k < top_k; k++) {
        uint expert_id = topk_indices[k];
        float w = topk_weights[k];
        acc += w * expert_outputs[expert_id * d_model + gid];
    }
    output[gid] = acc;
}
"#;

    /// `attention_f32` — scaled-dot-product attention matching the
    /// `rce-op-attention::AttentionLeaf` op contract:
    ///
    ///   inputs:  Q[B, D]  K[S, D]  V[S, D]  + scale (f32) + kv_start, active_len (u32)
    ///   output:  O[B, D] = softmax((Q · K_active^T) * scale) · V_active
    ///   K_active = K[kv_start..kv_start+active_len], V_active analogous.
    ///
    /// Architecturally simpler than efficient-genai's
    /// `attention_f16` (which carries 4-D batch+head+seq+dim strides
    /// for monolithic transformer forward passes): rce's leaf takes
    /// 2-D Q/K/V because the per-head split is done in the IR layer
    /// (one Apply node per head, like inv-ssi-inference's
    /// `multi_head_attention`). One thread per output element
    /// `(b, d)`; each thread runs the full per-row softmax internally.
    /// Inefficient at long contexts (duplicates softmax D times per
    /// row) but correct and validates the dispatch chain. A
    /// threadgroup-shared scores-buffer kernel (matching efficient-
    /// genai's pattern) is the next perf step.
    const ATTENTION_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void attention_f32(
    device const float* Q       [[buffer(0)]],   // [B, D]
    device const float* K       [[buffer(1)]],   // [S, D] (full sequence)
    device const float* V       [[buffer(2)]],   // [S, D]
    device       float* O       [[buffer(3)]],   // [B, D]
    constant uint&  D           [[buffer(4)]],
    constant uint&  S           [[buffer(5)]],   // K/V seq_len (unused except for bounds)
    constant uint&  kv_start    [[buffer(6)]],
    constant uint&  active_len  [[buffer(7)]],
    constant float& scale       [[buffer(8)]],
    uint2 gid                   [[thread_position_in_grid]]
) {
    uint b = gid.y;
    uint d = gid.x;
    if (d >= D) return;
    if (active_len == 0) {
        O[b * D + d] = 0.0f;
        return;
    }
    // Each thread does the full softmax for its (b, d). Wasteful but
    // correct; threadgroup-shared variant is a follow-up.

    // Pass 1: max(score) for stability.
    float max_score = -INFINITY;
    for (uint i = 0; i < active_len; i++) {
        uint k_pos = kv_start + i;
        float dot = 0.0f;
        for (uint j = 0; j < D; j++) {
            dot += Q[b * D + j] * K[k_pos * D + j];
        }
        float s = dot * scale;
        if (s > max_score) max_score = s;
    }

    // Pass 2: sum(exp(s - max)) and accumulate output for this d.
    float sum_exp = 0.0f;
    float acc = 0.0f;
    for (uint i = 0; i < active_len; i++) {
        uint k_pos = kv_start + i;
        float dot = 0.0f;
        for (uint j = 0; j < D; j++) {
            dot += Q[b * D + j] * K[k_pos * D + j];
        }
        float e = exp(dot * scale - max_score);
        sum_exp += e;
        acc += e * V[k_pos * D + d];
    }

    O[b * D + d] = acc / sum_exp;
}
"#;

    /// `rms_norm_f32_grouped` — group-aware RMSNorm.
    ///
    /// Vendored & adapted from
    /// `efficient-genai/src/hal/metal/shader.rs::sources::RMS_NORM`'s
    /// `rms_norm_f16` (lines 413-439). Two changes:
    ///   1. f32 throughout (matches rce's f32-first surface).
    ///   2. `group_size` and `ngroups` parameters so Mamba2 / Jamba
    ///      group-norm (where multiple groups share one weight tensor
    ///      but normalise independently) can use the same kernel.
    ///      `ngroups=1` recovers standard whole-vector RMSNorm.
    ///
    /// Each thread handles one group, computing the group's RMS over
    /// `group_size` elements and writing all `group_size` outputs.
    /// Total dispatch threads = `ngroups`.
    const RMS_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_f32_grouped(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&  ngroups    [[buffer(3)]],
    constant uint&  group_size [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= ngroups) return;

    uint base = gid * group_size;
    float sum_sq = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        float v = input[base + i];
        sum_sq += v * v;
    }
    float inv_rms = rsqrt(sum_sq / float(group_size) + eps);

    for (uint i = 0; i < group_size; i++) {
        output[base + i] = input[base + i] * inv_rms * weight[base + i];
    }
}
"#;

    /// `silu_rms_norm_f32_grouped` — **the first fused kernel landed by
    /// `rce-placement::fusion_eligible`-class work**. Computes
    /// `rms_norm(silu(x), weight, eps)` in one Metal dispatch instead of
    /// two: silu intermediate is never materialised, the per-block group
    /// values get a single read of `input` per pass (silu re-evaluated
    /// per pass because intermediate doesn't escape the thread). Args
    /// match `Custom("rms_norm")` exactly so a graph rewriter can
    /// substitute `Custom("silu") → Custom("rms_norm")` with
    /// `Custom("silu_rms_norm")` at the same node site without changing
    /// arg shapes.
    ///
    /// Math, exactly: y_i = silu(x_i) = x_i / (1 + exp(-x_i));
    ///                rms = sqrt(mean(y_i^2) + eps);
    ///                out_i = (y_i / rms) * weight_i.
    /// Equivalent to applying SiluLeaf then RmsNormLeaf up to f32
    /// reduction order (parity test below pins this to abs < 1e-4).
    const SILU_RMS_NORM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void silu_rms_norm_f32_grouped(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device       float* output [[buffer(2)]],
    constant uint&  ngroups    [[buffer(3)]],
    constant uint&  group_size [[buffer(4)]],
    constant float& eps        [[buffer(5)]],
    uint gid                   [[thread_position_in_grid]]
) {
    if (gid >= ngroups) return;

    uint base = gid * group_size;

    // Pass 1: silu(x) sum-of-squares for the RMS reduction. The silu
    // intermediate doesn't escape the thread — recomputing in pass 2
    // is cheaper than going through threadgroup memory at typical
    // group_size, and avoids any TG sync.
    float sum_sq = 0.0f;
    for (uint i = 0; i < group_size; i++) {
        float xi = input[base + i];
        float sx = xi / (1.0f + exp(-xi));  // silu
        sum_sq += sx * sx;
    }
    float inv_rms = rsqrt(sum_sq / float(group_size) + eps);

    // Pass 2: write silu(x) * inv_rms * weight.
    for (uint i = 0; i < group_size; i++) {
        float xi = input[base + i];
        float sx = xi / (1.0f + exp(-xi));
        output[base + i] = sx * inv_rms * weight[base + i];
    }
}
"#;

    pub struct MetalContext {
        device: Device,
        queue: CommandQueue,
        pipelines: Mutex<HashMap<&'static str, ComputePipelineState>>,
        /// Pipelines for runtime-emitted fused chains (Phase B of the
        /// `rce_fusion::codegen` rollout). Keyed by canonical chain
        /// function name (`fused_silu_rms_norm_f32`, etc.). Separate map
        /// because the keys aren't `&'static` — they're computed per
        /// chain at dispatch time and cached on first miss.
        fused_pipelines: Mutex<HashMap<String, ComputePipelineState>>,
        /// Phase S: weight upload cache. Callers register f32 data
        /// (typically static model weights) once and get a handle;
        /// subsequent dispatches reuse the uploaded buffer. The
        /// alternative (re-uploading 80MB of gate_W per call at full
        /// Jamba scale) is the per-call cost the cache eliminates.
        // Phase Y: tag every cached weight buffer with its element
        // dtype so dispatchers can validate the handle's storage
        // matches the kernel they're about to compile. f32 / f16 /
        // bf16 buffers all live in the same map, distinguished by
        // tag, not by separate sub-caches.
        weight_cache: Mutex<HashMap<u64, (Buffer, rce_fusion::codegen::Dtype)>>,
        next_weight_id: std::sync::atomic::AtomicU64,
    }

    // SAFETY: Metal Device / CommandQueue / ComputePipelineState are
    // thread-safe per Apple docs. Encoders are built per-call on the
    // current thread; never shared.
    unsafe impl Send for MetalContext {}
    unsafe impl Sync for MetalContext {}

    impl MetalContext {
        pub fn new() -> Option<Self> {
            let device = Device::system_default()?;
            let queue = device.new_command_queue();
            let mut pipelines = HashMap::new();
            register(&device, &mut pipelines, "silu_f32", SILU_MSL)?;
            register(
                &device,
                &mut pipelines,
                "rms_norm_f32_grouped",
                RMS_NORM_MSL,
            )?;
            register(
                &device,
                &mut pipelines,
                "silu_rms_norm_f32_grouped",
                SILU_RMS_NORM_MSL,
            )?;
            register(&device, &mut pipelines, "gelu_tanh_f32", GELU_MSL)?;
            register(&device, &mut pipelines, "swiglu_f32", SWIGLU_MSL)?;
            register(
                &device,
                &mut pipelines,
                "layer_norm_f32_grouped",
                LAYER_NORM_MSL,
            )?;
            register(&device, &mut pipelines, "attention_f32", ATTENTION_MSL)?;
            register(&device, &mut pipelines, "embedding_lookup_f32", EMBEDDING_MSL)?;
            register(&device, &mut pipelines, "moe_combine_f32", MOE_COMBINE_MSL)?;
            register(&device, &mut pipelines, "group_norm_f32", GROUP_NORM_MSL)?;
            register(&device, &mut pipelines, "topk_mask_f32", TOPK_MASK_MSL)?;
            register(&device, &mut pipelines, "argmax_f32", ARGMAX_MSL)?;
            register(&device, &mut pipelines, "conv2d_naive_f32", CONV2D_MSL)?;
            Some(Self {
                device,
                queue,
                pipelines: Mutex::new(pipelines),
                fused_pipelines: Mutex::new(HashMap::new()),
                weight_cache: Mutex::new(HashMap::new()),
                next_weight_id: std::sync::atomic::AtomicU64::new(1),
            })
        }

        // ── Weight upload cache (Phase S) ────────────────────────
        //
        // Each `cache_weight_f32` upload returns an opaque `u64`
        // handle. Subsequent `dispatch_*_cached` calls take the
        // handle and look up the buffer; the data is uploaded
        // exactly once. `release_weight` drops the cached buffer.

        /// Upload `data` (f32 little-endian bytes) as a cached
        /// weight buffer and return a handle for reuse.
        pub fn cache_weight_f32(&self, data: &[f32]) -> u64 {
            let bytes: &[u8] = bytemuck::cast_slice(data);
            self.cache_weight_bytes(bytes, rce_fusion::codegen::Dtype::F32)
        }

        /// Phase Y: upload `data` to the GPU as f16 storage, return
        /// the cache handle. One conversion at cache time; every
        /// later dispatch reads the narrow bytes directly.
        pub fn cache_weight_f16(&self, data: &[f32]) -> u64 {
            let bytes = f32_slice_to_f16_bytes(data);
            self.cache_weight_bytes(&bytes, rce_fusion::codegen::Dtype::F16)
        }

        /// Phase Y: upload `data` to the GPU as bf16 storage. Same
        /// shape as `cache_weight_f16` — narrow once, dispatch many.
        pub fn cache_weight_bf16(&self, data: &[f32]) -> u64 {
            let bytes = f32_slice_to_bf16_bytes(data);
            self.cache_weight_bytes(&bytes, rce_fusion::codegen::Dtype::BF16)
        }

        fn cache_weight_bytes(
            &self,
            bytes: &[u8],
            dtype: rce_fusion::codegen::Dtype,
        ) -> u64 {
            let buf = self.device.new_buffer_with_data(
                bytes.as_ptr() as *const _,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let id = self.next_weight_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            self.weight_cache
                .lock()
                .expect("metal weight cache mutex")
                .insert(id, (buf, dtype));
            id
        }

        /// Release a cached weight. Returns `true` if the handle was
        /// present, `false` otherwise.
        pub fn release_weight(&self, handle: u64) -> bool {
            self.weight_cache
                .lock()
                .expect("metal weight cache mutex")
                .remove(&handle)
                .is_some()
        }

        /// Number of cached weights — test affordance.
        pub fn cached_weight_count(&self) -> usize {
            self.weight_cache
                .lock()
                .expect("metal weight cache mutex")
                .len()
        }

        /// Get a clone of a cached weight buffer by handle. Returns
        /// `None` if the handle isn't known. Drops the dtype tag;
        /// dispatchers that need it should use `get_cached_weight_tagged`.
        fn get_cached_weight(&self, handle: u64) -> Option<Buffer> {
            self.weight_cache
                .lock()
                .expect("metal weight cache mutex")
                .get(&handle)
                .map(|(buf, _)| buf.clone())
        }

        /// Phase Y: tagged lookup — returns `(Buffer, Dtype)` so
        /// narrow dispatchers can validate the handle matches the
        /// kernel's element dtype before encoding.
        fn get_cached_weight_tagged(&self, handle: u64) -> Option<(Buffer, rce_fusion::codegen::Dtype)> {
            self.weight_cache
                .lock()
                .expect("metal weight cache mutex")
                .get(&handle)
                .map(|(buf, d)| (buf.clone(), *d))
        }

        // ── Host ↔ Device transfers ────────────────────────
        //
        // These let the caller move tensors between host and GPU at
        // chain boundaries. Inside a chain of Metal ops the
        // `DeviceTensor` input → `DeviceTensor` output flow keeps the
        // data resident — no host roundtrip per op.
        pub fn promote_to_device(&self, t: &HostTensor) -> Option<Value> {
            if t.dtype != DType::F32 {
                return None;
            }
            let buf = self.device.new_buffer_with_data(
                t.data.as_ptr() as *const _,
                t.data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let mem = MetalDeviceMem {
                buffer: buf,
                size_bytes: t.data.len(),
            };
            Some(Value::DeviceTensor(DeviceTensor::new(
                DeviceId::Metal,
                t.dtype,
                t.shape.clone(),
                Arc::new(mem),
            )))
        }

        pub fn demote_to_host(&self, t: &DeviceTensor) -> Option<Value> {
            if t.device != DeviceId::Metal || t.dtype != DType::F32 {
                return None;
            }
            let metal_mem = t.mem.as_any().downcast_ref::<MetalDeviceMem>()?;
            let n_bytes = metal_mem.size_bytes;
            let ptr = metal_mem.buffer.contents() as *const u8;
            let bytes = unsafe { std::slice::from_raw_parts(ptr, n_bytes).to_vec() };
            HostTensor::new(t.shape.clone(), DType::F32, bytes)
                .map(Value::Tensor)
                .ok()
        }

        /// Dispatch a codegen-emitted fused chain. The chain is resolved
        /// via `rce_fusion::codegen::emit_chain_msl`; the resulting MSL
        /// is compiled into a `ComputePipelineState` on first call and
        /// cached in `fused_pipelines` for every later dispatch of the
        /// same chain.
        ///
        /// Buffer-table contract (matches `EmittedKernel.buffers`):
        ///   slot 0:   input             (`&[f32]` device buffer)
        ///   slot 1:   output            (allocated here)
        ///   slot 2..: each fragment's uniforms in chain order, deduped
        ///             by name. Caller supplies them via `uniforms` in
        ///             the SAME order `EmittedKernel.buffers[2..]`
        ///             declares (the only practical lookup today: the
        ///             callsite knows its chain).
        ///   then:     `constant uint& ngroups`, `constant uint& group_size`
        ///   then:     each fragment's scalars in chain order, deduped
        ///             by name. Caller supplies them via `scalars` in
        ///             matching order.
        ///
        /// `input.numel()` must equal `ngroups * group_size` where
        /// `group_size = input.numel() / ngroups` — the chain operates
        /// in groups (a reduction op like `rms_norm` is per-group).
        pub fn dispatch_fused_chain(
            &self,
            chain: &[&str],
            input: &Value,
            uniforms: &[&Value],
            scalars: &[f32],
            ngroups: u32,
        ) -> Result<Value> {
            // Emit MSL up front so an unknown op fails fast.
            let emitted = rce_fusion::codegen::emit_chain_msl(chain).map_err(|e| {
                InterpretError::Unsupported(format!("metal fused-chain codegen: {e}"))
            })?;

            // Slot-count contract.
            let want_uniforms = emitted.buffers.len() - 2; // minus input + output
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if ngroups == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain `{}`: ngroups must be positive",
                    emitted.function_name,
                )));
            }

            // Resolve input + uniform Metal buffers.
            let (in_buf, in_shape, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total % (ngroups as usize) != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain `{}`: numel {in_total} not divisible by ngroups {ngroups}",
                    emitted.function_name,
                )));
            }
            let group_size: u32 = (in_total / ngroups as usize) as u32;

            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = in_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[2..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != in_total {
                    return Err(InterpretError::Unsupported(format!(
                        "metal fused-chain `{}`: uniform `{}` numel {u_total} != input numel {in_total}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            // Output buffer.
            let out_size = (in_total * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            // Look up or build the pipeline. The cache is keyed on the
            // canonical function name so the same chain hits the cache on
            // every dispatch after the first.
            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self
                        .device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal fused-chain `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal fused-chain `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self
                        .device
                        .new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal fused-chain `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            // Encode the dispatch. Slot order MUST match the codegen's
            // `EmittedKernel.buffers` + `ngroups` + `group_size` +
            // `EmittedKernel.scalars` layout.
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((2 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (2 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &ngroups as *const u32 as *const _);
            slot += 1;
            encoder.set_bytes(slot, 4, &group_size as *const u32 as *const _);
            slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let dispatch_threads = ngroups as usize;
            let tg_width = max_threads.min(dispatch_threads).max(1);
            encoder.dispatch_threads(
                MTLSize { width: dispatch_threads as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                chain = ?chain,
                kernel = %emitted.function_name,
                total = in_total,
                ngroups,
                group_size,
                "metal fused-chain dispatched",
            );
            finalize_output(out_buf, out_size as usize, in_shape, any_dev)
        }

        /// Dispatch the f16 variant of a fused chain. Thin wrapper
        /// around `dispatch_fused_chain_narrow` with `Dtype::F16`.
        pub fn dispatch_fused_chain_f16(
            &self,
            chain: &[&str],
            input: &Value,
            uniforms: &[&Value],
            scalars: &[f32],
            ngroups: u32,
        ) -> Result<Value> {
            self.dispatch_fused_chain_narrow(
                chain, input, uniforms, scalars, ngroups,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Dispatch the bf16 variant of a fused chain. Thin wrapper
        /// around `dispatch_fused_chain_narrow` with `Dtype::BF16`.
        /// Phase L of the codegen rollout.
        ///
        /// Requires Metal 3.1+ on Apple silicon (M2 and later); older
        /// targets will surface a compile error from the MSL pipeline
        /// build as `InterpretError::Unsupported`.
        pub fn dispatch_fused_chain_bf16(
            &self,
            chain: &[&str],
            input: &Value,
            uniforms: &[&Value],
            scalars: &[f32],
            ngroups: u32,
        ) -> Result<Value> {
            self.dispatch_fused_chain_narrow(
                chain, input, uniforms, scalars, ngroups,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Shared implementation for the narrow-dtype dispatch paths.
        /// Same chain shape + args contract as `dispatch_fused_chain`,
        /// but reads / writes narrow-precision buffers internally and
        /// converts host f32 ↔ narrow on upload + download.
        ///
        /// Host-side: input + uniform tensors arrive as f32 (today's
        /// HostTensor doesn't carry narrow dtypes); this method
        /// converts them to the target dtype on upload and back to
        /// f32 on download. The bandwidth + register-pressure win is
        /// on-GPU only until narrow dtypes land as first-class
        /// HostTensor dtypes upstream.
        fn dispatch_fused_chain_narrow(
            &self,
            chain: &[&str],
            input: &Value,
            uniforms: &[&Value],
            scalars: &[f32],
            ngroups: u32,
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            // Caller sanity: this entry point is for narrow dtypes
            // only. F32 inputs go through the wider, no-conversion
            // `dispatch_fused_chain`.
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_fused_chain_narrow called with F32; use dispatch_fused_chain",
            );
            let emitted = rce_fusion::codegen::emit_chain_msl_dtype(
                chain, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal fused-chain {} codegen: {e}", dtype.name(),
            )))?;

            let want_uniforms = emitted.buffers.len() - 2;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain {} `{}`: expected {} scalars, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if ngroups == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain {} `{}`: ngroups must be positive",
                    dtype.name(), emitted.function_name,
                )));
            }

            // Phase O fast-path: if `input` is already a narrow
            // HostTensor of the target dtype, use its raw bytes — no
            // f32 widening, no f32→narrow conversion. Falls back to
            // f32 conversion otherwise. Same for each uniform.
            let input_shape = value_shape(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal fused-chain {} `{}`: input must be a Tensor / DeviceTensor",
                dtype.name(), emitted.function_name,
            )))?;
            // We need `in_total` (the numel) before we have the bytes;
            // shape provides it (numel = shape.product()).
            let in_total: usize = input_shape.iter().product();
            if in_total == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain {} `{}`: input has zero numel",
                    dtype.name(), emitted.function_name,
                )));
            }
            if in_total % (ngroups as usize) != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal fused-chain {} `{}`: numel {in_total} not divisible by ngroups {ngroups}",
                    dtype.name(), emitted.function_name,
                )));
            }
            let group_size = (in_total / ngroups as usize) as u32;

            let op_label = format!("fused-chain {} `{}`", dtype.name(), emitted.function_name);
            let input_bytes = value_as_narrow_bytes(input, dtype, in_total, "input", &op_label)?;
            let in_buf = self.device.new_buffer_with_data(
                input_bytes.as_ptr() as *const _,
                input_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[2..]) {
                let u_bytes = value_as_narrow_bytes(u, dtype, in_total, decl.name, &op_label)?;
                let u_buf = self.device.new_buffer_with_data(
                    u_bytes.as_ptr() as *const _,
                    u_bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                uniform_bufs.push(u_buf);
            }

            let out_size_bytes = (in_total * dtype.size_bytes()) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self
                        .device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal fused-chain {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal fused-chain {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self
                        .device
                        .new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal fused-chain {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((2 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (2 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &ngroups as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &group_size as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let dispatch_threads = ngroups as usize;
            let tg_width = max_threads.min(dispatch_threads).max(1);
            encoder.dispatch_threads(
                MTLSize { width: dispatch_threads as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            // Phase O: keep the narrow output bytes in a narrow
            // HostTensor — no f32 widening. Callers that want f32
            // call `t.as_f32_promoted()` afterwards.
            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            narrow_bytes_into_host_tensor(out_bytes, input_shape, dtype)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal fused-chain {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Dispatch a codegen-emitted matmul-epilogue kernel:
        /// `output = epilogue(input @ weight)` where `input: [M, K]`,
        /// `weight: [K, N]`, `output: [M, N]`. The `epilogue` chain is
        /// a sequence of Phase A fragment names (elementwise only —
        /// reduction-bearing fragments are rejected by the codegen).
        ///
        /// Same cache + lazy-build behaviour as `dispatch_fused_chain`:
        /// first call for a given epilogue compiles the MSL and stores
        /// the pipeline under the canonical function name; every later
        /// call hits the cache.
        ///
        /// Args:
        ///   - `input`:    `Value::Tensor` of shape `[M, K]` (or any
        ///                 numel == M*K — the kernel walks indices).
        ///   - `weight`:   `Value::Tensor` of shape `[K, N]`.
        /// Pick which matmul kernel template to dispatch for a given
        /// `(M, K, N)` shape. The naive template (Phase F) launches
        /// `M*N` independent threads with no shared memory — minimal
        /// overhead for tiny shapes. The tiled template (Phase I)
        /// pays a per-threadgroup memory + barrier cost in exchange
        /// for shared `A_tile` / `B_tile` reads across the
        /// `TILE_M × TILE_N` threads of a threadgroup.
        ///
        /// The threshold is a heuristic, not measurement-derived:
        ///
        ///   - `Naive` when the output fits in one tile
        ///     (`M*N <= TILE_M*TILE_N`) OR every dimension is small
        ///     (`M < TILE_M/2` or `N < TILE_N/2`). The tile loads
        ///     would mostly hit the bounds-guarded `0.0f` branch and
        ///     the threadgroup memory + barriers would dominate.
        ///   - `Tiled` otherwise.
        ///
        /// Refine per shape with profiling — these thresholds are an
        /// educated default, not a per-device-tuned routing table.
        pub fn route_matmul(m: u32, k: u32, n: u32) -> MatmulRouting {
            let _ = k; // K doesn't affect the routing decision in this
                       // first-cut heuristic — the K loop runs either
                       // way, only the per-element threading shape
                       // differs.
            let tile_m = rce_fusion::codegen_matmul_tiled::TILE_M;
            let tile_n = rce_fusion::codegen_matmul_tiled::TILE_N;
            if (m as u64) * (n as u64) <= (tile_m as u64) * (tile_n as u64) {
                return MatmulRouting::Naive;
            }
            if m < tile_m / 2 || n < tile_n / 2 {
                return MatmulRouting::Naive;
            }
            MatmulRouting::Tiled
        }

        /// Auto-routing matmul + epilogue dispatch. Picks naive vs
        /// tiled via `route_matmul` and calls the corresponding
        /// low-level dispatcher. Same args contract as both — same
        /// epilogue, same `(M, K, N)`, same uniforms + scalars.
        ///
        /// This is the entry point `dispatch_matmul_epilogue_by_name`
        /// and the graph-rewriter Phase G hand off to. Callers that
        /// want to force one template (e.g. for shape-specific
        /// benchmarking) keep calling `dispatch_matmul_epilogue` or
        /// `dispatch_matmul_tiled` directly.
        pub fn dispatch_matmul_auto(
            &self,
            input: &Value,
            weight: &Value,
            m: u32,
            k: u32,
            n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            match Self::route_matmul(m, k, n) {
                MatmulRouting::Naive => self.dispatch_matmul_epilogue(
                    input, weight, m, k, n, epilogue, uniforms, scalars,
                ),
                MatmulRouting::Tiled => self.dispatch_matmul_tiled(
                    input, weight, m, k, n, epilogue, uniforms, scalars,
                ),
            }
        }

        ///   - `m`, `k`, `n`: matrix dimensions. Passed as constants.
        ///   - `epilogue`: chain of fragment names applied per output
        ///                 element after the dot product.
        ///   - `uniforms`: one `Value::Tensor` per fragment uniform, in
        ///                 the order `EmittedMatmulKernel.buffers[3..]`
        ///                 declares. Each must have numel == M*N
        ///                 (indexed at `i * N + j`).
        ///   - `scalars`:  one `f32` per fragment scalar, in chain order.

        /// Phase EE: naive FFN-swiglu (Phase Q) with cached weights.
        /// Same kernel + shape contract as `dispatch_ffn_swiglu`; gate
        /// and up weights come from `cache_weight_f32` handles. Drops
        /// the per-call gate + up uploads. For small/medium shapes
        /// where the naive kernel beats tiled, or for parity with
        /// Phase Y's tiled cached path.
        pub fn dispatch_ffn_swiglu_cached(
            &self,
            input: &Value,
            gate_handle: u64,
            up_handle: u64,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_ffn_swiglu::emit_ffn_swiglu_msl()
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached codegen: {e}",
                )))?;
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (gate_buf, gate_dtype) = self.get_cached_weight_tagged(gate_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: gate handle {gate_handle} not in cache",
                    emitted.function_name,
                )))?;
            if gate_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: gate handle {gate_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, gate_dtype.name(),
                )));
            }
            let (up_buf, up_dtype) = self.get_cached_weight_tagged(up_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: up handle {up_handle} not in cache",
                    emitted.function_name,
                )))?;
            if up_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: up handle {up_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, up_dtype.name(),
                )));
            }
            let gate_len = gate_buf.length() as usize / 4;
            if gate_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: cached gate numel {gate_len} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let up_len = up_buf.length() as usize / 4;
            if up_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: cached up numel {up_len} != K*N = {kn}",
                    emitted.function_name,
                )));
            }

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-cached `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-cached `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-cached `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&gate_buf), 0);
            encoder.set_buffer(2, Some(&up_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            // Naive Phase Q kernel uses dispatch_threads (one work-item
            // per output element), unlike the tiled cousin which needs
            // full-threadgroup dispatch.
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                kernel = %emitted.function_name,
                m, k, n, gate_handle, up_handle,
                "metal ffn-swiglu-cached dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], in_dev)
        }

        /// Phase EE: naive FFN-swiglu cached + narrow dtype. Same
        /// kernel as Phase V's `dispatch_ffn_swiglu_f16` / `_bf16` but
        /// gate + up come from `cache_weight_{f16,bf16}` handles.
        fn dispatch_ffn_swiglu_cached_narrow(
            &self,
            input: &Value,
            gate_handle: u64,
            up_handle: u64,
            m: u32, k: u32, n: u32,
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_ffn_swiglu_cached_narrow called with F32; use dispatch_ffn_swiglu_cached",
            );
            let emitted = rce_fusion::codegen_ffn_swiglu::emit_ffn_swiglu_msl_dtype(dtype)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} codegen: {e}", dtype.name(),
                )))?;
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);
            let elem_bytes = dtype.size_bytes();

            let (gate_buf, gate_dtype) = self.get_cached_weight_tagged(gate_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: gate handle {gate_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if gate_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: gate handle {gate_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, gate_dtype.name(), dtype.name(),
                )));
            }
            let (up_buf, up_dtype) = self.get_cached_weight_tagged(up_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: up handle {up_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if up_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: up handle {up_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, up_dtype.name(), dtype.name(),
                )));
            }
            let gate_len = gate_buf.length() as usize / elem_bytes;
            if gate_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: cached gate numel {gate_len} != K*N = {kn}",
                    dtype.name(), emitted.function_name,
                )));
            }
            let up_len = up_buf.length() as usize / elem_bytes;
            if up_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: cached up numel {up_len} != K*N = {kn}",
                    dtype.name(), emitted.function_name,
                )));
            }

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let input_f32 = value_as_f32(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal ffn-swiglu-cached {} `{}`: input must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if input_f32.len() != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: input numel {} != M*K={mk}",
                    dtype.name(), emitted.function_name, input_f32.len(),
                )));
            }
            let in_bytes = f32_to_bytes(&input_f32);
            let in_buf = self.device.new_buffer_with_data(
                in_bytes.as_ptr() as *const _, in_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let out_size_bytes = (mn * elem_bytes) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-cached {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-cached {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-cached {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&gate_buf), 0);
            encoder.set_buffer(2, Some(&up_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-cached {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Phase EE f16 cached naive FFN-swiglu.
        pub fn dispatch_ffn_swiglu_cached_f16(
            &self,
            input: &Value,
            gate_handle: u64, up_handle: u64,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_cached_narrow(
                input, gate_handle, up_handle, m, k, n,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase EE bf16 cached naive FFN-swiglu.
        pub fn dispatch_ffn_swiglu_cached_bf16(
            &self,
            input: &Value,
            gate_handle: u64, up_handle: u64,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_cached_narrow(
                input, gate_handle, up_handle, m, k, n,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Phase S: tiled FFN-swiglu with cached weights. Same kernel
        /// + shape contract as `dispatch_ffn_swiglu_tiled` but takes
        /// pre-uploaded weight handles via `cache_weight_f32`. The
        /// per-call cost drops from "upload input + 2× weight upload"
        /// to "upload input + look up two cached buffers" — the
        /// bandwidth win for the call-after-the-first case.
        pub fn dispatch_ffn_swiglu_tiled_cached(
            &self,
            input: &Value,
            gate_handle: u64,
            up_handle: u64,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_ffn_swiglu_tiled::emit_ffn_swiglu_tiled_msl()
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached codegen: {e}",
                )))?;
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (gate_buf, gate_dtype) = self.get_cached_weight_tagged(gate_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: gate handle {gate_handle} not in cache",
                    emitted.function_name,
                )))?;
            if gate_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: gate handle {gate_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, gate_dtype.name(),
                )));
            }
            let (up_buf, up_dtype) = self.get_cached_weight_tagged(up_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: up handle {up_handle} not in cache",
                    emitted.function_name,
                )))?;
            if up_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: up handle {up_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, up_dtype.name(),
                )));
            }
            // Validate cached buffer sizes match K*N.
            let gate_len = gate_buf.length() as usize / 4;
            if gate_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: cached gate numel {gate_len} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let up_len = up_buf.length() as usize / 4;
            if up_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: cached up numel {up_len} != K*N = {kn}",
                    emitted.function_name,
                )));
            }

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled-cached `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled-cached `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled-cached `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&gate_buf), 0);
            encoder.set_buffer(2, Some(&up_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            let tile_m = rce_fusion::codegen_ffn_swiglu_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_ffn_swiglu_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                kernel = %emitted.function_name,
                m, k, n, gate_handle, up_handle,
                "metal ffn-swiglu-tiled-cached dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], in_dev)
        }

        /// Phase Y: narrow tiled FFN-swiglu with cached weights.
        /// Shared implementation for f16 + bf16. Validates each cached
        /// handle's stored dtype matches `dtype`; uploads input
        /// f32 → narrow bytes per call; reads narrow output bytes
        /// back as f32. Same kernel + dispatch shape as Phase R, just
        /// with narrow storage on input + weights + output.
        fn dispatch_ffn_swiglu_tiled_cached_narrow(
            &self,
            input: &Value,
            gate_handle: u64,
            up_handle: u64,
            m: u32, k: u32, n: u32,
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_ffn_swiglu_tiled_cached_narrow called with F32; use the f32 entry point",
            );
            let emitted = rce_fusion::codegen_ffn_swiglu_tiled::emit_ffn_swiglu_tiled_msl_dtype(dtype)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} codegen: {e}", dtype.name(),
                )))?;
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);
            let elem_bytes = dtype.size_bytes();

            let (gate_buf, gate_dtype) = self.get_cached_weight_tagged(gate_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: gate handle {gate_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if gate_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: gate handle {gate_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, gate_dtype.name(), dtype.name(),
                )));
            }
            let (up_buf, up_dtype) = self.get_cached_weight_tagged(up_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: up handle {up_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if up_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: up handle {up_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, up_dtype.name(), dtype.name(),
                )));
            }
            let gate_len = gate_buf.length() as usize / elem_bytes;
            if gate_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: cached gate numel {gate_len} != K*N = {kn}",
                    dtype.name(), emitted.function_name,
                )));
            }
            let up_len = up_buf.length() as usize / elem_bytes;
            if up_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: cached up numel {up_len} != K*N = {kn}",
                    dtype.name(), emitted.function_name,
                )));
            }

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let input_f32 = value_as_f32(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal ffn-swiglu-tiled-cached {} `{}`: input must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if input_f32.len() != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: input numel {} != M*K={mk}",
                    dtype.name(), emitted.function_name, input_f32.len(),
                )));
            }
            let in_bytes = f32_to_bytes(&input_f32);
            let in_buf = self.device.new_buffer_with_data(
                in_bytes.as_ptr() as *const _, in_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let out_size_bytes = (mn * elem_bytes) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled-cached {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled-cached {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled-cached {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&gate_buf), 0);
            encoder.set_buffer(2, Some(&up_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            let tile_m = rce_fusion::codegen_ffn_swiglu_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_ffn_swiglu_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                kernel = %emitted.function_name,
                m, k, n, gate_handle, up_handle, dtype = dtype.name(),
                "metal ffn-swiglu-tiled-cached {} dispatched", dtype.name(),
            );

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled-cached {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Phase Y f16 cached tiled FFN-swiglu. Cached weights must
        /// have been registered with `cache_weight_f16`.
        pub fn dispatch_ffn_swiglu_tiled_cached_f16(
            &self,
            input: &Value,
            gate_handle: u64, up_handle: u64,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_tiled_cached_narrow(
                input, gate_handle, up_handle, m, k, n,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase Y bf16 cached tiled FFN-swiglu. Cached weights must
        /// have been registered with `cache_weight_bf16`.
        pub fn dispatch_ffn_swiglu_tiled_cached_bf16(
            &self,
            input: &Value,
            gate_handle: u64, up_handle: u64,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_tiled_cached_narrow(
                input, gate_handle, up_handle, m, k, n,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Phase R: tiled FFN-swiglu dispatch. Same shape contract as
        /// `dispatch_ffn_swiglu` but uses the threadgroup-tiled kernel
        /// — the input tile is loaded once into shared memory and
        /// reused across the gate AND up dot products (the
        /// bandwidth amplifier the naive Phase Q kernel doesn't have).
        ///
        /// Uses `dispatch_threadgroups` (not `_threads`) so every
        /// threadgroup gets a full `TILE_N × TILE_M` complement —
        /// the cooperative tile load requires it.
        pub fn dispatch_ffn_swiglu_tiled(
            &self,
            input: &Value, gate_w: &Value, up_w: &Value,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_ffn_swiglu_tiled::emit_ffn_swiglu_tiled_msl()
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled codegen: {e}",
                )))?;
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let (gate_buf, _, gate_total, gate_dev) =
                buffer_for_f32_input(&self.device, gate_w, &emitted.function_name, "gate_W")?;
            if gate_total != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled `{}`: gate_W numel {gate_total} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let (up_buf, _, up_total, up_dev) =
                buffer_for_f32_input(&self.device, up_w, &emitted.function_name, "up_W")?;
            if up_total != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu-tiled `{}`: up_W numel {up_total} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu-tiled `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(gate_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(up_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            // Same full-threadgroup dispatch convention as Phase I.
            let tile_m = rce_fusion::codegen_ffn_swiglu_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_ffn_swiglu_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                kernel = %emitted.function_name,
                m, k, n,
                "metal ffn-swiglu-tiled dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], in_dev | gate_dev | up_dev)
        }

        /// Dispatch the codegen-emitted FFN-swiglu kernel (Phase Q).
        /// One Metal dispatch computes
        ///   `output = silu(input @ gate_W) * (input @ up_W)`
        /// with `input` read once per K iteration and used in two
        /// in-register accumulators. Replaces the
        /// `dispatch_matmul_epilogue(matmul + swiglu uniform)` +
        /// host-side up matmul combo with a single kernel.
        pub fn dispatch_ffn_swiglu(
            &self,
            input: &Value,
            gate_w: &Value,
            up_w: &Value,
            m: u32,
            k: u32,
            n: u32,
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_ffn_swiglu::emit_ffn_swiglu_msl()
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal ffn-swiglu codegen: {e}",
                )))?;
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let (gate_buf, _, gate_total, gate_dev) =
                buffer_for_f32_input(&self.device, gate_w, &emitted.function_name, "gate_W")?;
            if gate_total != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu `{}`: gate_W numel {gate_total} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let (up_buf, _, up_total, up_dev) =
                buffer_for_f32_input(&self.device, up_w, &emitted.function_name, "up_W")?;
            if up_total != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal ffn-swiglu `{}`: up_W numel {up_total} != K*N = {kn}",
                    emitted.function_name,
                )));
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal ffn-swiglu `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal ffn-swiglu `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(gate_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(up_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                kernel = %emitted.function_name,
                m, k, n,
                "metal ffn-swiglu dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], in_dev | gate_dev | up_dev)
        }

        /// Phase V — shared narrow-dtype path for both naive (Phase Q)
        /// and tiled (Phase R) FFN-swiglu kernels. Converts host f32 ↔
        /// narrow on upload + download, emits the right narrow MSL
        /// variant, and selects `dispatch_threads` (naive) vs
        /// `dispatch_thread_groups` (tiled) based on `tiled`. Output
        /// is materialised as an f32 `HostTensor` — host narrow
        /// dtypes are still upstream work.
        fn dispatch_ffn_swiglu_narrow(
            &self,
            input: &Value, gate_w: &Value, up_w: &Value,
            m: u32, k: u32, n: u32,
            tiled: bool,
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_ffn_swiglu_narrow called with F32; use the f32 entry points",
            );
            let label = if tiled { "ffn-swiglu-tiled" } else { "ffn-swiglu" };
            let emitted_fn_name: String;
            let emitted_source: String;
            if tiled {
                let e = rce_fusion::codegen_ffn_swiglu_tiled::emit_ffn_swiglu_tiled_msl_dtype(dtype)
                    .map_err(|e| InterpretError::Unsupported(format!(
                        "metal {label} {} codegen: {e}", dtype.name(),
                    )))?;
                emitted_fn_name = e.function_name;
                emitted_source  = e.source;
            } else {
                let e = rce_fusion::codegen_ffn_swiglu::emit_ffn_swiglu_msl_dtype(dtype)
                    .map_err(|e| InterpretError::Unsupported(format!(
                        "metal {label} {} codegen: {e}", dtype.name(),
                    )))?;
                emitted_fn_name = e.function_name;
                emitted_source  = e.source;
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal {label} {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted_fn_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let input_f32 = value_as_f32(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal {label} {} `{}`: input must be f32",
                dtype.name(), emitted_fn_name,
            )))?;
            if input_f32.len() != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal {label} {} `{}`: input numel {} != M*K={mk}",
                    dtype.name(), emitted_fn_name, input_f32.len(),
                )));
            }
            let gate_f32 = value_as_f32(gate_w).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal {label} {} `{}`: gate_W must be f32",
                dtype.name(), emitted_fn_name,
            )))?;
            if gate_f32.len() != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal {label} {} `{}`: gate_W numel {} != K*N={kn}",
                    dtype.name(), emitted_fn_name, gate_f32.len(),
                )));
            }
            let up_f32 = value_as_f32(up_w).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal {label} {} `{}`: up_W must be f32",
                dtype.name(), emitted_fn_name,
            )))?;
            if up_f32.len() != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal {label} {} `{}`: up_W numel {} != K*N={kn}",
                    dtype.name(), emitted_fn_name, up_f32.len(),
                )));
            }

            let in_bytes = f32_to_bytes(&input_f32);
            let in_buf = self.device.new_buffer_with_data(
                in_bytes.as_ptr() as *const _, in_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let gate_bytes = f32_to_bytes(&gate_f32);
            let gate_buf = self.device.new_buffer_with_data(
                gate_bytes.as_ptr() as *const _, gate_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let up_bytes = f32_to_bytes(&up_f32);
            let up_buf = self.device.new_buffer_with_data(
                up_bytes.as_ptr() as *const _, up_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let out_size_bytes = (mn * dtype.size_bytes()) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted_fn_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted_source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal {label} {} `{}`: compile failed: {e}",
                            dtype.name(), emitted_fn_name,
                        )))?;
                    let func = lib.get_function(&emitted_fn_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal {label} {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted_fn_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal {label} {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted_fn_name,
                        )))?;
                    fp.insert(emitted_fn_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&gate_buf), 0);
            encoder.set_buffer(2, Some(&up_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &m as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k as *const u32 as *const _);
            encoder.set_bytes(6, 4, &n as *const u32 as *const _);

            if tiled {
                let tile_m = rce_fusion::codegen_ffn_swiglu_tiled::TILE_M as u64;
                let tile_n = rce_fusion::codegen_ffn_swiglu_tiled::TILE_N as u64;
                let groups_x = (n as u64 + tile_n - 1) / tile_n;
                let groups_y = (m as u64 + tile_m - 1) / tile_m;
                encoder.dispatch_thread_groups(
                    MTLSize { width: groups_x, height: groups_y, depth: 1 },
                    MTLSize { width: tile_n, height: tile_m, depth: 1 },
                );
            } else {
                let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
                let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
                encoder.dispatch_threads(
                    MTLSize { width: n as u64, height: m as u64, depth: 1 },
                    MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
                );
            }
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                kernel = %emitted_fn_name,
                m, k, n, dtype = dtype.name(),
                "metal {label} {} dispatched", dtype.name(),
            );

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal {label} {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted_fn_name,
                )))
        }

        /// Phase V f16 naive FFN-swiglu. Host f32 input + weights →
        /// half storage on the GPU → f32 host output. Bandwidth-only
        /// win; accumulators stay f32 internally.
        pub fn dispatch_ffn_swiglu_f16(
            &self,
            input: &Value, gate_w: &Value, up_w: &Value,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_narrow(
                input, gate_w, up_w, m, k, n,
                /*tiled*/ false, rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase V bf16 naive FFN-swiglu. Requires Apple silicon M2+
        /// (Metal 3.1+); older targets surface the pipeline compile
        /// error as `Unsupported`.
        pub fn dispatch_ffn_swiglu_bf16(
            &self,
            input: &Value, gate_w: &Value, up_w: &Value,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_narrow(
                input, gate_w, up_w, m, k, n,
                /*tiled*/ false, rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Phase V f16 tiled FFN-swiglu. Threadgroup tiles stay f32
        /// even when storage is half — narrowing only on the output
        /// store.
        pub fn dispatch_ffn_swiglu_tiled_f16(
            &self,
            input: &Value, gate_w: &Value, up_w: &Value,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_narrow(
                input, gate_w, up_w, m, k, n,
                /*tiled*/ true, rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase V bf16 tiled FFN-swiglu.
        pub fn dispatch_ffn_swiglu_tiled_bf16(
            &self,
            input: &Value, gate_w: &Value, up_w: &Value,
            m: u32, k: u32, n: u32,
        ) -> Result<Value> {
            self.dispatch_ffn_swiglu_narrow(
                input, gate_w, up_w, m, k, n,
                /*tiled*/ true, rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// f16 variant of the matmul-epilogue dispatch. Phase M of
        /// the codegen rollout. Thin wrapper around
        /// `dispatch_matmul_epilogue_narrow` with `Dtype::F16`.
        pub fn dispatch_matmul_epilogue_f16(
            &self,
            input: &Value, weight: &Value,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_epilogue_narrow(
                input, weight, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// bf16 variant of the matmul-epilogue dispatch. Phase M.
        pub fn dispatch_matmul_epilogue_bf16(
            &self,
            input: &Value, weight: &Value,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_epilogue_narrow(
                input, weight, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Shared implementation for f16 / bf16 matmul-epilogue
        /// dispatch. Same flow as the f32 path, but converts host
        /// f32 ↔ narrow on upload + download and emits the narrow
        /// kernel variant. The output is materialised as an f32
        /// `HostTensor` (host narrow dtypes are still upstream work).
        fn dispatch_matmul_epilogue_narrow(
            &self,
            input: &Value, weight: &Value,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value], scalars: &[f32],
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_matmul_epilogue_narrow called with F32; use dispatch_matmul_epilogue",
            );
            let emitted = rce_fusion::codegen_matmul_epilogue::emit_matmul_epilogue_msl_dtype(
                epilogue, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal matmul-epilogue {} codegen: {e}", dtype.name(),
            )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: expected {} scalar values, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }

            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            // Input + weight uploads.
            let input_f32 = value_as_f32(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal matmul-epilogue {} `{}`: input must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if input_f32.len() != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: input numel {} != M*K={mk}",
                    dtype.name(), emitted.function_name, input_f32.len(),
                )));
            }
            let weight_f32 = value_as_f32(weight).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal matmul-epilogue {} `{}`: weight must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if weight_f32.len() != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: weight numel {} != K*N={kn}",
                    dtype.name(), emitted.function_name, weight_f32.len(),
                )));
            }
            let in_bytes = f32_to_bytes(&input_f32);
            let in_buf = self.device.new_buffer_with_data(
                in_bytes.as_ptr() as *const _, in_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let w_bytes = f32_to_bytes(&weight_f32);
            let w_buf = self.device.new_buffer_with_data(
                w_bytes.as_ptr() as *const _, w_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let u_f32 = value_as_f32(u).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: uniform `{}` must be f32",
                    dtype.name(), emitted.function_name, decl.name,
                )))?;
                if u_f32.len() != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-epilogue {} `{}`: uniform `{}` numel {} != M*N={mn}",
                        dtype.name(), emitted.function_name, decl.name, u_f32.len(),
                    )));
                }
                let u_bytes = f32_to_bytes(&u_f32);
                let u_buf = self.device.new_buffer_with_data(
                    u_bytes.as_ptr() as *const _, u_bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                uniform_bufs.push(u_buf);
            }

            let out_size_bytes = (mn * dtype.size_bytes()) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-epilogue {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&w_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _); slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Phase BB: matmul-epilogue cached + narrow dtype. Same
        /// kernel as Phase M (`dispatch_matmul_epilogue_narrow`) but
        /// `weight` is a `u64` handle from `cache_weight_f16` /
        /// `cache_weight_bf16`. The cached weight is validated to
        /// have a matching dtype tag.
        ///
        /// Input + uniforms still upload per-call (those vary across
        /// dispatches). Only the weight matrix avoids per-call upload.
        /// Output materialises as an f32 `HostTensor`.
        fn dispatch_matmul_epilogue_cached_narrow(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value], scalars: &[f32],
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_matmul_epilogue_cached_narrow called with F32; use dispatch_matmul_epilogue_cached",
            );
            let emitted = rce_fusion::codegen_matmul_epilogue::emit_matmul_epilogue_msl_dtype(
                epilogue, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal matmul-epilogue-cached {} codegen: {e}", dtype.name(),
            )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: expected {} scalar values, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);
            let elem_bytes = dtype.size_bytes();

            let (w_buf, w_dtype) = self.get_cached_weight_tagged(weight_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: weight handle {weight_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if w_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: weight handle {weight_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, w_dtype.name(), dtype.name(),
                )));
            }
            let w_len = w_buf.length() as usize / elem_bytes;
            if w_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: cached weight numel {w_len} != K*N = {kn}",
                    dtype.name(), emitted.function_name,
                )));
            }

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let input_f32 = value_as_f32(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal matmul-epilogue-cached {} `{}`: input must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if input_f32.len() != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: input numel {} != M*K={mk}",
                    dtype.name(), emitted.function_name, input_f32.len(),
                )));
            }
            let in_bytes = f32_to_bytes(&input_f32);
            let in_buf = self.device.new_buffer_with_data(
                in_bytes.as_ptr() as *const _, in_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let u_f32 = value_as_f32(u).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: uniform `{}` must be f32",
                    dtype.name(), emitted.function_name, decl.name,
                )))?;
                if u_f32.len() != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-epilogue-cached {} `{}`: uniform `{}` numel {} != M*N={mn}",
                        dtype.name(), emitted.function_name, decl.name, u_f32.len(),
                    )));
                }
                let u_bytes = f32_to_bytes(&u_f32);
                let u_buf = self.device.new_buffer_with_data(
                    u_bytes.as_ptr() as *const _, u_bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                uniform_bufs.push(u_buf);
            }

            let out_size_bytes = (mn * elem_bytes) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue-cached {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-epilogue-cached {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue-cached {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&w_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _); slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Phase BB f16 cached matmul-epilogue. Weight handle must
        /// come from `cache_weight_f16`.
        pub fn dispatch_matmul_epilogue_cached_f16(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_epilogue_cached_narrow(
                input, weight_handle, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase BB bf16 cached matmul-epilogue. Requires Apple
        /// silicon M2+ (Metal 3.1+); older targets surface the
        /// pipeline compile error as `Unsupported`.
        pub fn dispatch_matmul_epilogue_cached_bf16(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_epilogue_cached_narrow(
                input, weight_handle, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        pub fn dispatch_matmul_epilogue(
            &self,
            input: &Value,
            weight: &Value,
            m: u32,
            k: u32,
            n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_matmul_epilogue::emit_matmul_epilogue_msl(epilogue)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue codegen: {e}",
                )))?;
            let want_uniforms = emitted.buffers.len() - 3; // minus input + weight + output
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }

            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let (w_buf, _, w_total, w_dev) =
                buffer_for_f32_input(&self.device, weight, &emitted.function_name, "weight")?;
            if w_total != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue `{}`: weight numel {w_total} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = in_dev | w_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-epilogue `{}`: uniform `{}` numel {u_total} != M*N = {mn}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self
                        .device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-epilogue `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self
                        .device
                        .new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(w_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            // 2D dispatch: width = N (x), height = M (y). Threadgroup
            // sizing: split the device's max-threads budget into a 2D
            // tile that doesn't exceed N x M.
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                epilogue = ?epilogue,
                kernel = %emitted.function_name,
                m, k, n,
                "metal matmul-epilogue dispatched",
            );
            // Output shape: [M, N]. We carry the shape forward as-is
            // (callers that care about the 2D shape can reshape).
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], any_dev)
        }

        /// Phase AA: matmul-epilogue dispatch with **cached** weight.
        /// Identical kernel and shape contract to
        /// `dispatch_matmul_epilogue` — only the `weight` argument
        /// changes from `&Value` to a `u64` handle from
        /// `cache_weight_f32`. The cached buffer's dtype tag is
        /// validated as Dtype::F32; mismatch surfaces a clean dtype
        /// error rather than a buffer-size mismatch.
        ///
        /// Per-call uniforms / scalars / input are unchanged from the
        /// uncached path (uniforms vary across dispatches; only the
        /// weight matrix is the "load once, dispatch many" candidate).
        pub fn dispatch_matmul_epilogue_cached(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_matmul_epilogue::emit_matmul_epilogue_msl(epilogue)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached codegen: {e}",
                )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (w_buf, w_dtype) = self.get_cached_weight_tagged(weight_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: weight handle {weight_handle} not in cache",
                    emitted.function_name,
                )))?;
            if w_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: weight handle {weight_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, w_dtype.name(),
                )));
            }
            let w_len = w_buf.length() as usize / 4;
            if w_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: cached weight numel {w_len} != K*N = {kn}",
                    emitted.function_name,
                )));
            }

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-epilogue-cached `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = in_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-epilogue-cached `{}`: uniform `{}` numel {u_total} != M*N = {mn}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue-cached `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-epilogue-cached `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-epilogue-cached `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&w_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                epilogue = ?epilogue,
                kernel = %emitted.function_name,
                m, k, n, weight_handle,
                "metal matmul-epilogue-cached dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], any_dev)
        }

        /// Dispatch a codegen-emitted double-matmul kernel:
        /// `output = epilogue(A @ B @ C)` where `A: [M, K1]`, `B: [K1, K2]`,
        /// `C: [K2, N]`, `output: [M, N]`. The intermediate `[M, K2]`
        /// matrix lives entirely in registers — no allocation, no
        /// host/device round-trip, no second kernel launch.
        ///
        /// f16 variant of the double-matmul dispatch. Phase N.
        pub fn dispatch_matmul_matmul_f16(
            &self,
            a: &Value, b: &Value, c: &Value,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_matmul_narrow(
                a, b, c, m, k1, k2, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// bf16 variant of the double-matmul dispatch. Phase N.
        pub fn dispatch_matmul_matmul_bf16(
            &self,
            a: &Value, b: &Value, c: &Value,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_matmul_narrow(
                a, b, c, m, k1, k2, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Shared implementation for narrow-dtype double-matmul
        /// dispatch. Converts host f32 ↔ narrow on upload + download
        /// and emits the narrow kernel variant.
        fn dispatch_matmul_matmul_narrow(
            &self,
            a: &Value, b: &Value, c: &Value,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(dtype != rce_fusion::codegen::Dtype::F32);
            let emitted = rce_fusion::codegen_matmul_matmul::emit_matmul_matmul_msl_dtype(
                epilogue, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal matmul² {} codegen: {e}", dtype.name(),
            )))?;
            let want_uniforms = emitted.buffers.len() - 4;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul² {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul² {} `{}`: expected {} scalars, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k1 == 0 || k2 == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul² {} `{}`: M={m} K1={k1} K2={k2} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }

            let mk1 = (m as usize) * (k1 as usize);
            let k1k2 = (k1 as usize) * (k2 as usize);
            let k2n = (k2 as usize) * (n as usize);
            let mn  = (m as usize) * (n as usize);

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let upload_matrix = |val: &Value, label: &str, want_numel: usize| -> Result<Buffer> {
                let f = value_as_f32(val).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul² {} `{}`: {label} must be f32",
                    dtype.name(), emitted.function_name,
                )))?;
                if f.len() != want_numel {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul² {} `{}`: {label} numel {} != expected {want_numel}",
                        dtype.name(), emitted.function_name, f.len(),
                    )));
                }
                let bytes = f32_to_bytes(&f);
                Ok(self.device.new_buffer_with_data(
                    bytes.as_ptr() as *const _, bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            };
            let a_buf = upload_matrix(a, "A", mk1)?;
            let b_buf = upload_matrix(b, "B", k1k2)?;
            let c_buf = upload_matrix(c, "C", k2n)?;

            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[4..]) {
                uniform_bufs.push(upload_matrix(u, decl.name, mn)?);
            }

            let out_size_bytes = (mn * dtype.size_bytes()) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul² {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul² {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul² {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buf), 0);
            encoder.set_buffer(1, Some(&b_buf), 0);
            encoder.set_buffer(2, Some(&c_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((4 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (4 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m  as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k1 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k2 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n  as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _); slot += 1;
            }
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul² {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Same caching + lazy-build behaviour as
        /// `dispatch_matmul_epilogue` and `dispatch_fused_chain`:
        /// first call for a given epilogue compiles the MSL and stores
        /// the pipeline under the canonical function name.
        pub fn dispatch_matmul_matmul(
            &self,
            a: &Value,
            b: &Value,
            c: &Value,
            m: u32,
            k1: u32,
            k2: u32,
            n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_matmul_matmul::emit_matmul_matmul_msl(epilogue)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-matmul codegen: {e}",
                )))?;
            let want_uniforms = emitted.buffers.len() - 4; // minus A + B + C + output
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k1 == 0 || k2 == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul `{}`: M={m} K1={k1} K2={k2} N={n} must all be positive",
                    emitted.function_name,
                )));
            }

            let mk1 = (m as usize)  * (k1 as usize);
            let k1k2 = (k1 as usize) * (k2 as usize);
            let k2n = (k2 as usize) * (n as usize);
            let mn  = (m as usize)  * (n as usize);

            let (a_buf, _, a_total, a_dev) =
                buffer_for_f32_input(&self.device, a, &emitted.function_name, "A")?;
            if a_total != mk1 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul `{}`: A numel {a_total} != M*K1 = {mk1}",
                    emitted.function_name,
                )));
            }
            let (b_buf, _, b_total, b_dev) =
                buffer_for_f32_input(&self.device, b, &emitted.function_name, "B")?;
            if b_total != k1k2 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul `{}`: B numel {b_total} != K1*K2 = {k1k2}",
                    emitted.function_name,
                )));
            }
            let (c_buf, _, c_total, c_dev) =
                buffer_for_f32_input(&self.device, c, &emitted.function_name, "C")?;
            if c_total != k2n {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul `{}`: C numel {c_total} != K2*N = {k2n}",
                    emitted.function_name,
                )));
            }
            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = a_dev | b_dev | c_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[4..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-matmul `{}`: uniform `{}` numel {u_total} != M*N = {mn}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self
                        .device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-matmul `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-matmul `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self
                        .device
                        .new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-matmul `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(a_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(b_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(c_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((4 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (4 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k1 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k2 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                epilogue = ?epilogue,
                kernel = %emitted.function_name,
                m, k1, k2, n,
                "metal matmul-matmul dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], any_dev)
        }

        /// Phase DD: double-matmul (`output = epilogue(A @ B @ C)`)
        /// with cached weight handles. `A` stays as a per-call input;
        /// `B` and `C` become `u64` handles from `cache_weight_f32`.
        /// Tag-validated. Drops two of three per-call uploads, the
        /// largest savings of any cached form so far.
        pub fn dispatch_matmul_matmul_cached(
            &self,
            a: &Value,
            b_handle: u64,
            c_handle: u64,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_matmul_matmul::emit_matmul_matmul_msl(epilogue)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached codegen: {e}",
                )))?;
            let want_uniforms = emitted.buffers.len() - 4;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k1 == 0 || k2 == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: M={m} K1={k1} K2={k2} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk1 = (m as usize)  * (k1 as usize);
            let k1k2 = (k1 as usize) * (k2 as usize);
            let k2n = (k2 as usize) * (n as usize);
            let mn  = (m as usize)  * (n as usize);

            let (b_buf, b_dtype) = self.get_cached_weight_tagged(b_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: B handle {b_handle} not in cache",
                    emitted.function_name,
                )))?;
            if b_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: B handle {b_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, b_dtype.name(),
                )));
            }
            let b_len = b_buf.length() as usize / 4;
            if b_len != k1k2 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: cached B numel {b_len} != K1*K2 = {k1k2}",
                    emitted.function_name,
                )));
            }
            let (c_buf, c_dtype) = self.get_cached_weight_tagged(c_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: C handle {c_handle} not in cache",
                    emitted.function_name,
                )))?;
            if c_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: C handle {c_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, c_dtype.name(),
                )));
            }
            let c_len = c_buf.length() as usize / 4;
            if c_len != k2n {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: cached C numel {c_len} != K2*N = {k2n}",
                    emitted.function_name,
                )));
            }

            let (a_buf, _, a_total, a_dev) =
                buffer_for_f32_input(&self.device, a, &emitted.function_name, "A")?;
            if a_total != mk1 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached `{}`: A numel {a_total} != M*K1 = {mk1}",
                    emitted.function_name,
                )));
            }
            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = a_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[4..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-matmul-cached `{}`: uniform `{}` numel {u_total} != M*N = {mn}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-matmul-cached `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-matmul-cached `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-matmul-cached `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(a_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&b_buf), 0);
            encoder.set_buffer(2, Some(&c_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((4 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (4 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k1 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k2 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                epilogue = ?epilogue,
                kernel = %emitted.function_name,
                m, k1, k2, n, b_handle, c_handle,
                "metal matmul-matmul-cached dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], any_dev)
        }

        /// Phase DD: double-matmul cached + narrow dtype. Same kernel
        /// as `dispatch_matmul_matmul_narrow` but B + C come from
        /// `cache_weight_{f16,bf16}` handles. Both handles must be
        /// tagged at `dtype`.
        fn dispatch_matmul_matmul_cached_narrow(
            &self,
            a: &Value,
            b_handle: u64,
            c_handle: u64,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value], scalars: &[f32],
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_matmul_matmul_cached_narrow called with F32; use dispatch_matmul_matmul_cached",
            );
            let emitted = rce_fusion::codegen_matmul_matmul::emit_matmul_matmul_msl_dtype(
                epilogue, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal matmul-matmul-cached {} codegen: {e}", dtype.name(),
            )))?;
            let want_uniforms = emitted.buffers.len() - 4;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: expected {} scalar values, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k1 == 0 || k2 == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: M={m} K1={k1} K2={k2} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }
            let mk1 = (m as usize)  * (k1 as usize);
            let k1k2 = (k1 as usize) * (k2 as usize);
            let k2n = (k2 as usize) * (n as usize);
            let mn  = (m as usize)  * (n as usize);
            let elem_bytes = dtype.size_bytes();

            let (b_buf, b_dtype) = self.get_cached_weight_tagged(b_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: B handle {b_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if b_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: B handle {b_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, b_dtype.name(), dtype.name(),
                )));
            }
            let b_len = b_buf.length() as usize / elem_bytes;
            if b_len != k1k2 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: cached B numel {b_len} != K1*K2 = {k1k2}",
                    dtype.name(), emitted.function_name,
                )));
            }
            let (c_buf, c_dtype) = self.get_cached_weight_tagged(c_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: C handle {c_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if c_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: C handle {c_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, c_dtype.name(), dtype.name(),
                )));
            }
            let c_len = c_buf.length() as usize / elem_bytes;
            if c_len != k2n {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: cached C numel {c_len} != K2*N = {k2n}",
                    dtype.name(), emitted.function_name,
                )));
            }

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let a_f32 = value_as_f32(a).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal matmul-matmul-cached {} `{}`: A must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if a_f32.len() != mk1 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: A numel {} != M*K1 = {mk1}",
                    dtype.name(), emitted.function_name, a_f32.len(),
                )));
            }
            let a_bytes = f32_to_bytes(&a_f32);
            let a_buf = self.device.new_buffer_with_data(
                a_bytes.as_ptr() as *const _, a_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[4..]) {
                let u_f32 = value_as_f32(u).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: uniform `{}` must be f32",
                    dtype.name(), emitted.function_name, decl.name,
                )))?;
                if u_f32.len() != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-matmul-cached {} `{}`: uniform `{}` numel {} != M*N={mn}",
                        dtype.name(), emitted.function_name, decl.name, u_f32.len(),
                    )));
                }
                let u_bytes = f32_to_bytes(&u_f32);
                let u_buf = self.device.new_buffer_with_data(
                    u_bytes.as_ptr() as *const _, u_bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                uniform_bufs.push(u_buf);
            }

            let out_size_bytes = (mn * elem_bytes) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-matmul-cached {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-matmul-cached {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-matmul-cached {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&a_buf), 0);
            encoder.set_buffer(1, Some(&b_buf), 0);
            encoder.set_buffer(2, Some(&c_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((4 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (4 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k1 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k2 as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _); slot += 1;
            }

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let (tg_w, tg_h) = pick_2d_threadgroup(max_threads, n as usize, m as usize);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: m as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: tg_h as u64, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-matmul-cached {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Phase DD f16 cached double-matmul. B + C handles must come
        /// from `cache_weight_f16`.
        pub fn dispatch_matmul_matmul_cached_f16(
            &self,
            a: &Value,
            b_handle: u64, c_handle: u64,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_matmul_cached_narrow(
                a, b_handle, c_handle, m, k1, k2, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase DD bf16 cached double-matmul.
        pub fn dispatch_matmul_matmul_cached_bf16(
            &self,
            a: &Value,
            b_handle: u64, c_handle: u64,
            m: u32, k1: u32, k2: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_matmul_cached_narrow(
                a, b_handle, c_handle, m, k1, k2, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Dispatch a codegen-emitted **tiled** matmul + epilogue
        /// kernel. Same chain shape as `dispatch_matmul_epilogue` —
        /// `output = epilogue(input @ weight)` — but using the
        /// threadgroup-memory tiled template emitted by
        /// `rce_fusion::codegen_matmul_tiled`. Each threadgroup
        /// owns a `TILE_M × TILE_N` output tile and walks the K
        /// dimension in `TILE_K`-sized blocks, sharing the input
        /// reads across all threads in the group.
        ///
        /// f16 variant of the tiled matmul dispatch. Phase N.
        pub fn dispatch_matmul_tiled_f16(
            &self,
            input: &Value, weight: &Value,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_tiled_narrow(
                input, weight, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// bf16 variant of the tiled matmul dispatch. Phase N.
        pub fn dispatch_matmul_tiled_bf16(
            &self,
            input: &Value, weight: &Value,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_tiled_narrow(
                input, weight, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Shared implementation for narrow-dtype tiled-matmul dispatch.
        fn dispatch_matmul_tiled_narrow(
            &self,
            input: &Value, weight: &Value,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(dtype != rce_fusion::codegen::Dtype::F32);
            let emitted = rce_fusion::codegen_matmul_tiled::emit_matmul_tiled_msl_dtype(
                epilogue, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal matmul-tiled {} codegen: {e}", dtype.name(),
            )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled {} `{}`: expected {} scalars, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }

            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let upload_matrix = |val: &Value, label: &str, want_numel: usize| -> Result<Buffer> {
                let f = value_as_f32(val).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-tiled {} `{}`: {label} must be f32",
                    dtype.name(), emitted.function_name,
                )))?;
                if f.len() != want_numel {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-tiled {} `{}`: {label} numel {} != expected {want_numel}",
                        dtype.name(), emitted.function_name, f.len(),
                    )));
                }
                let bytes = f32_to_bytes(&f);
                Ok(self.device.new_buffer_with_data(
                    bytes.as_ptr() as *const _, bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                ))
            };
            let in_buf = upload_matrix(input, "input", mk)?;
            let w_buf = upload_matrix(weight, "weight", kn)?;
            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                uniform_bufs.push(upload_matrix(u, decl.name, mn)?);
            }

            let out_size_bytes = (mn * dtype.size_bytes()) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-tiled {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&w_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _); slot += 1;
            }

            // Same full-threadgroup dispatch as the f32 tiled path —
            // cooperative shared-memory loads require it.
            let tile_m = rce_fusion::codegen_matmul_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_matmul_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-tiled {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Dispatch uses `dispatch_threadgroups` (not
        /// `dispatch_threads`) so every threadgroup gets a full
        /// `TILE_N × TILE_M` complement of threads — required for the
        /// cooperative tile load to leave every shared-memory slot
        /// populated. Boundary threadgroups (covering shapes that
        /// aren't tile multiples) are still full at the threadgroup
        /// level; the kernel's bounds checks gate the global memory
        /// reads + writes.
        pub fn dispatch_matmul_tiled(
            &self,
            input: &Value,
            weight: &Value,
            m: u32,
            k: u32,
            n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_matmul_tiled::emit_matmul_tiled_msl(epilogue)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-tiled codegen: {e}",
                )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }

            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let (w_buf, _, w_total, w_dev) =
                buffer_for_f32_input(&self.device, weight, &emitted.function_name, "weight")?;
            if w_total != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled `{}`: weight numel {w_total} != K*N = {kn}",
                    emitted.function_name,
                )));
            }
            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = in_dev | w_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-tiled `{}`: uniform `{}` numel {u_total} != M*N = {mn}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self
                        .device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-tiled `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self
                        .device
                        .new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(w_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            // dispatch_threadgroups (not _threads): each threadgroup
            // must be full TILE_N × TILE_M so the cooperative loads
            // populate every shared-memory slot.
            let tile_m = rce_fusion::codegen_matmul_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_matmul_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                epilogue = ?epilogue,
                kernel = %emitted.function_name,
                m, k, n,
                tile_m = tile_m as u32,
                tile_n = tile_n as u32,
                "metal matmul-tiled dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], any_dev)
        }

        /// Phase CC: tiled matmul-epilogue dispatch with a **cached**
        /// weight handle. Same kernel + per-element epilogue contract
        /// as `dispatch_matmul_tiled`; only the `weight` argument
        /// changes from `&Value` to a `u64` handle from
        /// `cache_weight_f32`. Drops the per-call weight upload.
        pub fn dispatch_matmul_tiled_cached(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value],
            scalars: &[f32],
        ) -> Result<Value> {
            let emitted = rce_fusion::codegen_matmul_tiled::emit_matmul_tiled_msl(epilogue)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached codegen: {e}",
                )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: expected {want_uniforms} uniform tensors, got {}",
                    emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: expected {} scalar values, got {}",
                    emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: M={m} K={k} N={n} must all be positive",
                    emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);

            let (w_buf, w_dtype) = self.get_cached_weight_tagged(weight_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: weight handle {weight_handle} not in cache",
                    emitted.function_name,
                )))?;
            if w_dtype != rce_fusion::codegen::Dtype::F32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: weight handle {weight_handle} cached as {} \
                     but kernel needs f32",
                    emitted.function_name, w_dtype.name(),
                )));
            }
            let w_len = w_buf.length() as usize / 4;
            if w_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: cached weight numel {w_len} != K*N = {kn}",
                    emitted.function_name,
                )));
            }

            let (in_buf, _, in_total, in_dev) =
                buffer_for_f32_input(&self.device, input, &emitted.function_name, "input")?;
            if in_total != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached `{}`: input numel {in_total} != M*K = {mk}",
                    emitted.function_name,
                )));
            }
            let mut uniform_bufs = Vec::with_capacity(uniforms.len());
            let mut any_dev = in_dev;
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let (u_buf, _, u_total, u_dev) =
                    buffer_for_f32_input(&self.device, u, &emitted.function_name, decl.name)?;
                if u_total != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-tiled-cached `{}`: uniform `{}` numel {u_total} != M*N = {mn}",
                        emitted.function_name, decl.name,
                    )));
                }
                uniform_bufs.push(u_buf);
                any_dev |= u_dev;
            }

            let out_size = (mn * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled-cached `{}`: compile failed: {e}",
                            emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-tiled-cached `{}`: get_function failed: {e}",
                            emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled-cached `{}`: pipeline state failed: {e}",
                            emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&w_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf.as_buf()), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _);
                slot += 1;
            }

            let tile_m = rce_fusion::codegen_matmul_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_matmul_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                epilogue = ?epilogue,
                kernel = %emitted.function_name,
                m, k, n, weight_handle,
                "metal matmul-tiled-cached dispatched",
            );
            finalize_output(out_buf, out_size as usize, vec![m as usize, n as usize], any_dev)
        }

        /// Phase CC: tiled matmul-epilogue cached + narrow dtype.
        /// Same kernel as `dispatch_matmul_tiled_narrow` (Phase N) but
        /// the weight is a `u64` handle from `cache_weight_f16` /
        /// `cache_weight_bf16`. Tag-validated against `dtype`.
        fn dispatch_matmul_tiled_cached_narrow(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str],
            uniforms: &[&Value], scalars: &[f32],
            dtype: rce_fusion::codegen::Dtype,
        ) -> Result<Value> {
            debug_assert!(
                dtype != rce_fusion::codegen::Dtype::F32,
                "dispatch_matmul_tiled_cached_narrow called with F32; use dispatch_matmul_tiled_cached",
            );
            let emitted = rce_fusion::codegen_matmul_tiled::emit_matmul_tiled_msl_dtype(
                epilogue, dtype,
            ).map_err(|e| InterpretError::Unsupported(format!(
                "metal matmul-tiled-cached {} codegen: {e}", dtype.name(),
            )))?;
            let want_uniforms = emitted.buffers.len() - 3;
            if uniforms.len() != want_uniforms {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: expected {want_uniforms} uniform tensors, got {}",
                    dtype.name(), emitted.function_name, uniforms.len(),
                )));
            }
            if scalars.len() != emitted.scalars.len() {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: expected {} scalar values, got {}",
                    dtype.name(), emitted.function_name, emitted.scalars.len(), scalars.len(),
                )));
            }
            if m == 0 || k == 0 || n == 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: M={m} K={k} N={n} must all be positive",
                    dtype.name(), emitted.function_name,
                )));
            }
            let mk = (m as usize) * (k as usize);
            let kn = (k as usize) * (n as usize);
            let mn = (m as usize) * (n as usize);
            let elem_bytes = dtype.size_bytes();

            let (w_buf, w_dtype) = self.get_cached_weight_tagged(weight_handle)
                .ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: weight handle {weight_handle} not in cache",
                    dtype.name(), emitted.function_name,
                )))?;
            if w_dtype != dtype {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: weight handle {weight_handle} cached as {} \
                     but kernel needs {}",
                    dtype.name(), emitted.function_name, w_dtype.name(), dtype.name(),
                )));
            }
            let w_len = w_buf.length() as usize / elem_bytes;
            if w_len != kn {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: cached weight numel {w_len} != K*N = {kn}",
                    dtype.name(), emitted.function_name,
                )));
            }

            let f32_to_bytes: fn(&[f32]) -> Vec<u8> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes,
                rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };
            let bytes_to_f32: fn(&[u8]) -> Vec<f32> = match dtype {
                rce_fusion::codegen::Dtype::F16 => f16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::BF16 => bf16_bytes_to_f32_vec,
                rce_fusion::codegen::Dtype::F32 => unreachable!(),
            };

            let input_f32 = value_as_f32(input).ok_or_else(|| InterpretError::Unsupported(format!(
                "metal matmul-tiled-cached {} `{}`: input must be f32",
                dtype.name(), emitted.function_name,
            )))?;
            if input_f32.len() != mk {
                return Err(InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: input numel {} != M*K={mk}",
                    dtype.name(), emitted.function_name, input_f32.len(),
                )));
            }
            let in_bytes = f32_to_bytes(&input_f32);
            let in_buf = self.device.new_buffer_with_data(
                in_bytes.as_ptr() as *const _, in_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let mut uniform_bufs: Vec<Buffer> = Vec::with_capacity(uniforms.len());
            for (u, decl) in uniforms.iter().zip(&emitted.buffers[3..]) {
                let u_f32 = value_as_f32(u).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: uniform `{}` must be f32",
                    dtype.name(), emitted.function_name, decl.name,
                )))?;
                if u_f32.len() != mn {
                    return Err(InterpretError::Unsupported(format!(
                        "metal matmul-tiled-cached {} `{}`: uniform `{}` numel {} != M*N={mn}",
                        dtype.name(), emitted.function_name, decl.name, u_f32.len(),
                    )));
                }
                let u_bytes = f32_to_bytes(&u_f32);
                let u_buf = self.device.new_buffer_with_data(
                    u_bytes.as_ptr() as *const _, u_bytes.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                uniform_bufs.push(u_buf);
            }

            let out_size_bytes = (mn * elem_bytes) as u64;
            let out_buf = self.device.new_buffer(out_size_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = {
                let mut fp = self.fused_pipelines.lock().expect("metal fused pipeline mutex");
                if let Some(p) = fp.get(&emitted.function_name) {
                    p.clone()
                } else {
                    let lib = self.device
                        .new_library_with_source(&emitted.source, &metal::CompileOptions::new())
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled-cached {} `{}`: compile failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    let func = lib.get_function(&emitted.function_name, None).map_err(|e| {
                        InterpretError::Unsupported(format!(
                            "metal matmul-tiled-cached {} `{}`: get_function failed: {e}",
                            dtype.name(), emitted.function_name,
                        ))
                    })?;
                    let p = self.device.new_compute_pipeline_state_with_function(&func)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal matmul-tiled-cached {} `{}`: pipeline state failed: {e}",
                            dtype.name(), emitted.function_name,
                        )))?;
                    fp.insert(emitted.function_name.clone(), p.clone());
                    p
                }
            };

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(&in_buf), 0);
            encoder.set_buffer(1, Some(&w_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            for (i, u_buf) in uniform_bufs.iter().enumerate() {
                encoder.set_buffer((3 + i) as u64, Some(u_buf), 0);
            }
            let mut slot = (3 + uniform_bufs.len()) as u64;
            encoder.set_bytes(slot, 4, &m as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &k as *const u32 as *const _); slot += 1;
            encoder.set_bytes(slot, 4, &n as *const u32 as *const _); slot += 1;
            for s in scalars {
                encoder.set_bytes(slot, 4, s as *const f32 as *const _); slot += 1;
            }

            let tile_m = rce_fusion::codegen_matmul_tiled::TILE_M as u64;
            let tile_n = rce_fusion::codegen_matmul_tiled::TILE_N as u64;
            let groups_x = (n as u64 + tile_n - 1) / tile_n;
            let groups_y = (m as u64 + tile_m - 1) / tile_m;
            encoder.dispatch_thread_groups(
                MTLSize { width: groups_x, height: groups_y, depth: 1 },
                MTLSize { width: tile_n, height: tile_m, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let out_bytes = read_buffer_bytes(&out_buf, out_size_bytes as usize);
            let out_f32 = bytes_to_f32(&out_bytes);
            HostTensor::from_f32(vec![m as usize, n as usize], &out_f32)
                .map(Value::Tensor)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal matmul-tiled-cached {} `{}`: output assembly failed: {e:?}",
                    dtype.name(), emitted.function_name,
                )))
        }

        /// Phase CC f16 cached tiled matmul-epilogue. Weight handle
        /// must come from `cache_weight_f16`.
        pub fn dispatch_matmul_tiled_cached_f16(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_tiled_cached_narrow(
                input, weight_handle, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::F16,
            )
        }

        /// Phase CC bf16 cached tiled matmul-epilogue.
        pub fn dispatch_matmul_tiled_cached_bf16(
            &self,
            input: &Value,
            weight_handle: u64,
            m: u32, k: u32, n: u32,
            epilogue: &[&str], uniforms: &[&Value], scalars: &[f32],
        ) -> Result<Value> {
            self.dispatch_matmul_tiled_cached_narrow(
                input, weight_handle, m, k, n, epilogue, uniforms, scalars,
                rce_fusion::codegen::Dtype::BF16,
            )
        }

        /// Number of distinct fused-chain pipelines compiled so far.
        /// Test affordance — confirms the cache absorbs repeats.
        pub fn fused_pipeline_count(&self) -> usize {
            self.fused_pipelines.lock().expect("metal fused pipeline mutex").len()
        }

        pub fn execute_named(&self, name: &str, args: &[Value]) -> Result<Value> {
            match name {
                "silu" => self.silu(args),
                "rms_norm" => self.rms_norm(args),
                "silu_rms_norm" => self.silu_rms_norm(args),
                "gelu" => self.elementwise_unary(args, "gelu_tanh_f32", "gelu"),
                "swiglu" => self.swiglu(args),
                "layer_norm" => self.layer_norm(args),
                "attention" => self.attention(args),
                "embedding_lookup" => self.embedding_lookup(args),
                "moe_combine" => self.moe_combine(args),
                "group_norm" => self.group_norm(args),
                "topk_mask" => self.topk_mask(args),
                "argmax" => self.argmax(args),
                "conv2d" => self.conv2d(args),
                // Codegen-emitted fused-chain dispatch. Names of shape
                // `fused_<op1>_<op2>_..._f32` come out of
                // `rce_fusion::fuse_with_codegen`. Parse the name back
                // to a chain via the codegen module (single source of
                // truth for which ops have fragments) and dispatch.
                //
                // Args contract for the parsed chain (matches what
                // `rce_fusion::apply_fusion` produces for codegen-driven
                // fusion of a chain with exactly one reduction op):
                //   args[0]              = input
                //   args[1..1+nu]        = uniforms (in chain order)
                //   args[1+nu]           = ngroups (i32 scalar)
                //   args[2+nu..2+nu+ns]  = scalars (in chain order)
                // where `nu` = number of unique fragment uniforms and
                // `ns` = number of unique fragment scalars for the chain.
                // Phase H double-matmul dispatch. Most-specific prefix
                // first: `fused_matmul2` IS a prefix of `fused_matmul`
                // textually, but `starts_with` order matters.
                other if other.starts_with("fused_matmul2") => {
                    self.dispatch_matmul_matmul_by_name(other, args)
                }
                // Phase I tiled matmul + epilogue dispatch. Same
                // matmul-shape contract as Phase F but uses the
                // threadgroup-memory tiled kernel. Tested before the
                // bare `fused_matmul` prefix because the tiled names
                // are a more-specific prefix.
                other if other.starts_with("fused_matmul_tiled") => {
                    self.dispatch_matmul_tiled_by_name(other, args)
                }
                // Phase F matmul-epilogue dispatch path. Tested
                // *before* the generic fused_ path because the matmul
                // names also start with `fused_` — the more specific
                // prefix wins.
                other if other.starts_with("fused_matmul") => {
                    self.dispatch_matmul_epilogue_by_name(other, args)
                }
                // Phase X: FFN-swiglu OpDispatch routing. Tiled prefix
                // first (it's a superset of the naive prefix
                // `fused_ffn_swiglu_`), then the naive form. Both
                // route through dtype-aware dispatchers so f32 / f16 /
                // bf16 canonical names all reach the right kernel.
                other if other.starts_with("fused_ffn_swiglu_tiled") => {
                    self.dispatch_ffn_swiglu_tiled_by_name(other, args)
                }
                other if other.starts_with("fused_ffn_swiglu") => {
                    self.dispatch_ffn_swiglu_by_name(other, args)
                }
                other if other.starts_with("fused_") => {
                    self.dispatch_fused_by_name(other, args)
                }
                other => Err(InterpretError::Unsupported(format!(
                    "metal leaf: doesn't handle `{other}`"
                ))),
            }
        }

        /// Resolve a `fused_<...>_f32` name back to its chain, sanity-check
        /// the args against the chain's expected layout, and dispatch
        /// via `dispatch_fused_chain`. Returns `Unsupported` if the name
        /// doesn't parse, if any op in the parsed chain has no fragment
        /// (shouldn't happen if the name came from the codegen pass),
        /// or if the args don't match the expected shape.
        fn dispatch_fused_by_name(&self, name: &str, args: &[Value]) -> Result<Value> {
            let (chain, dtype) = rce_fusion::codegen::parse_function_name_with_dtype(name).ok_or_else(|| {
                InterpretError::Unsupported(format!(
                    "metal leaf: name `{name}` looks fused but doesn't parse",
                ))
            })?;
            // Use the codegen layout to figure out how many uniform
            // buffers + scalars the chain expects, then slice `args`
            // accordingly. Dtype only affects buffer typing; the args
            // layout is the same for f32 and f16.
            let chain_refs: Vec<&str> = chain.iter().copied().collect();
            let emitted = rce_fusion::codegen::emit_chain_msl_dtype(&chain_refs, dtype).map_err(|e| {
                InterpretError::Unsupported(format!(
                    "metal leaf: fused chain `{name}` emit failed: {e}",
                ))
            })?;
            let n_uniforms = emitted.buffers.len() - 2; // minus input, output
            let n_scalars  = emitted.scalars.len();
            let want_args = 2 + n_uniforms + n_scalars;
            if args.len() != want_args {
                return Err(InterpretError::Unsupported(format!(
                    "metal leaf: fused chain `{name}` expected {want_args} args \
                     (input + {n_uniforms} uniforms + ngroups + {n_scalars} scalars), got {}",
                    args.len(),
                )));
            }
            let input = &args[0];
            let uniforms: Vec<&Value> = args[1..1 + n_uniforms].iter().collect();
            let ngroups = arg_i32(&args[1 + n_uniforms], "ngroups")?;
            if ngroups <= 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal leaf: fused chain `{name}` ngroups must be positive, got {ngroups}",
                )));
            }
            let scalars: Vec<f32> = args[2 + n_uniforms..]
                .iter()
                .map(|v| arg_f32(v, "fused-chain scalar"))
                .collect::<Result<_>>()?;
            // Route by dtype.
            match dtype {
                rce_fusion::codegen::Dtype::F32 =>
                    self.dispatch_fused_chain(&chain_refs, input, &uniforms, &scalars, ngroups as u32),
                rce_fusion::codegen::Dtype::F16 =>
                    self.dispatch_fused_chain_f16(&chain_refs, input, &uniforms, &scalars, ngroups as u32),
                rce_fusion::codegen::Dtype::BF16 =>
                    self.dispatch_fused_chain_bf16(&chain_refs, input, &uniforms, &scalars, ngroups as u32),
            }
        }

        /// Phase F matmul-epilogue dispatch. Resolves a
        /// `fused_matmul_<...>_f32` name back to its epilogue chain,
        /// validates the args layout, then calls
        /// `dispatch_matmul_epilogue`.
        ///
        /// Args contract:
        ///   args[0]              = input  (`[M, K]` tensor)
        ///   args[1]              = weight (`[K, N]` tensor)
        ///   args[2..2+nu]        = uniforms, each `[M, N]` tensor, in
        ///                          chain order
        ///   args[2+nu]           = M (i32 scalar)
        ///   args[3+nu]           = K (i32 scalar)
        ///   args[4+nu]           = N (i32 scalar)
        ///   args[5+nu..]         = scalars (f32), in chain order
        fn dispatch_matmul_epilogue_by_name(&self, name: &str, args: &[Value]) -> Result<Value> {
            let (epilogue, dtype) = rce_fusion::codegen_matmul_epilogue::parse_function_name_with_dtype(name).ok_or_else(|| {
                InterpretError::Unsupported(format!(
                    "metal leaf: name `{name}` looks matmul-fused but doesn't parse",
                ))
            })?;
            let epilogue_refs: Vec<&str> = epilogue.iter().copied().collect();
            let emitted = rce_fusion::codegen_matmul_epilogue::emit_matmul_epilogue_msl_dtype(&epilogue_refs, dtype)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal leaf: matmul-fused `{name}` emit failed: {e}",
                )))?;
            let n_uniforms = emitted.buffers.len() - 3; // minus input + weight + output
            let n_scalars  = emitted.scalars.len();

            // Two arg conventions, distinguished by total arg count.
            // The graph rewriter (Phase G) produces inferred-shape
            // calls — no explicit M/K/N — relying on the runtime tensor
            // shapes. Low-level callers can still pass M/K/N as i32
            // scalars.
            //
            //   Inferred shape: input is `[M, K]`, weight is `[K, N]`.
            //     args = [input, weight, ...uniforms, ...scalars]
            //     total = 2 + n_uniforms + n_scalars
            //
            //   Explicit dims: any shape; M, K, N supplied separately.
            //     args = [input, weight, ...uniforms, M, K, N, ...scalars]
            //     total = 5 + n_uniforms + n_scalars
            let want_args_inferred = 2 + n_uniforms + n_scalars;
            let want_args_explicit = 5 + n_uniforms + n_scalars;

            let (input, weight, uniforms, m, k, n, scalars) = if args.len() == want_args_inferred {
                let input  = &args[0];
                let weight = &args[1];
                let in_shape  = value_shape(input).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: matmul-fused `{name}` input must be a Tensor / DeviceTensor",
                )))?;
                let w_shape   = value_shape(weight).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: matmul-fused `{name}` weight must be a Tensor / DeviceTensor",
                )))?;
                if in_shape.len() != 2 || w_shape.len() != 2 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul-fused `{name}` inferred-shape form needs 2D input / weight; got shapes {in_shape:?} / {w_shape:?}",
                    )));
                }
                if in_shape[1] != w_shape[0] {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul-fused `{name}` K mismatch: input.shape[1]={} != weight.shape[0]={}",
                        in_shape[1], w_shape[0],
                    )));
                }
                let m = in_shape[0] as u32;
                let k = in_shape[1] as u32;
                let n = w_shape[1] as u32;
                let uniforms: Vec<&Value> = args[2..2 + n_uniforms].iter().collect();
                let scalars: Vec<f32> = args[2 + n_uniforms..]
                    .iter()
                    .map(|v| arg_f32(v, "matmul-fused scalar"))
                    .collect::<Result<_>>()?;
                (input, weight, uniforms, m, k, n, scalars)
            } else if args.len() == want_args_explicit {
                let input  = &args[0];
                let weight = &args[1];
                let uniforms: Vec<&Value> = args[2..2 + n_uniforms].iter().collect();
                let m = arg_i32(&args[2 + n_uniforms], "M")?;
                let k = arg_i32(&args[3 + n_uniforms], "K")?;
                let n = arg_i32(&args[4 + n_uniforms], "N")?;
                if m <= 0 || k <= 0 || n <= 0 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul-fused `{name}` M={m} K={k} N={n} must all be positive",
                    )));
                }
                let scalars: Vec<f32> = args[5 + n_uniforms..]
                    .iter()
                    .map(|v| arg_f32(v, "matmul-fused scalar"))
                    .collect::<Result<_>>()?;
                (input, weight, uniforms, m as u32, k as u32, n as u32, scalars)
            } else {
                return Err(InterpretError::Unsupported(format!(
                    "metal leaf: matmul-fused `{name}` expected {want_args_inferred} args \
                     (input + weight + {n_uniforms} uniforms + {n_scalars} scalars, with 2D shape inference) \
                     or {want_args_explicit} args (… + M + K + N), got {}",
                    args.len(),
                )));
            };

            // Phase J + M: route by dtype, then by shape (for f32).
            // f32 goes through the auto-router so rewriter-driven
            // graphs pick naive vs tiled. f16 and bf16 go through the
            // narrow-dtype path (no auto-routing yet — tiled f16 / bf16
            // is Phase N work). Callers that want explicit control
            // stay on the low-level dispatchers.
            match dtype {
                rce_fusion::codegen::Dtype::F32 =>
                    self.dispatch_matmul_auto(input, weight, m, k, n, &epilogue_refs, &uniforms, &scalars),
                rce_fusion::codegen::Dtype::F16 =>
                    self.dispatch_matmul_epilogue_f16(input, weight, m, k, n, &epilogue_refs, &uniforms, &scalars),
                rce_fusion::codegen::Dtype::BF16 =>
                    self.dispatch_matmul_epilogue_bf16(input, weight, m, k, n, &epilogue_refs, &uniforms, &scalars),
            }
        }

        /// Phase H double-matmul dispatch by name. Two arg conventions
        /// (mirroring Phase F's `dispatch_matmul_epilogue_by_name`):
        ///
        ///   Inferred shape: `A: [M, K1]`, `B: [K1, K2]`, `C: [K2, N]`.
        ///     args = [A, B, C, ...uniforms, ...scalars]
        ///
        ///   Explicit dims:
        ///     args = [A, B, C, ...uniforms, M, K1, K2, N, ...scalars]
        fn dispatch_matmul_matmul_by_name(&self, name: &str, args: &[Value]) -> Result<Value> {
            let (epilogue, dtype) = rce_fusion::codegen_matmul_matmul::parse_function_name_with_dtype(name).ok_or_else(|| {
                InterpretError::Unsupported(format!(
                    "metal leaf: name `{name}` looks matmul2-fused but doesn't parse",
                ))
            })?;
            let epilogue_refs: Vec<&str> = epilogue.iter().copied().collect();
            let emitted = rce_fusion::codegen_matmul_matmul::emit_matmul_matmul_msl_dtype(&epilogue_refs, dtype)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal leaf: matmul2-fused `{name}` emit failed: {e}",
                )))?;
            let n_uniforms = emitted.buffers.len() - 4; // minus A + B + C + output
            let n_scalars  = emitted.scalars.len();

            let want_args_inferred = 3 + n_uniforms + n_scalars;
            let want_args_explicit = 7 + n_uniforms + n_scalars;

            let (a, b, c, uniforms, m, k1, k2, n, scalars) = if args.len() == want_args_inferred {
                let a = &args[0];
                let b = &args[1];
                let c = &args[2];
                let a_shape = value_shape(a).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: matmul2-fused `{name}` A must be Tensor/DeviceTensor",
                )))?;
                let b_shape = value_shape(b).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: matmul2-fused `{name}` B must be Tensor/DeviceTensor",
                )))?;
                let c_shape = value_shape(c).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: matmul2-fused `{name}` C must be Tensor/DeviceTensor",
                )))?;
                if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul2-fused `{name}` inferred-shape form needs 2D A/B/C; got {a_shape:?} / {b_shape:?} / {c_shape:?}",
                    )));
                }
                if a_shape[1] != b_shape[0] {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul2-fused `{name}` K1 mismatch: A[1]={} != B[0]={}",
                        a_shape[1], b_shape[0],
                    )));
                }
                if b_shape[1] != c_shape[0] {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul2-fused `{name}` K2 mismatch: B[1]={} != C[0]={}",
                        b_shape[1], c_shape[0],
                    )));
                }
                let m  = a_shape[0] as u32;
                let k1 = a_shape[1] as u32;
                let k2 = b_shape[1] as u32;
                let n  = c_shape[1] as u32;
                let uniforms: Vec<&Value> = args[3..3 + n_uniforms].iter().collect();
                let scalars: Vec<f32> = args[3 + n_uniforms..]
                    .iter()
                    .map(|v| arg_f32(v, "matmul2-fused scalar"))
                    .collect::<Result<_>>()?;
                (a, b, c, uniforms, m, k1, k2, n, scalars)
            } else if args.len() == want_args_explicit {
                let a = &args[0];
                let b = &args[1];
                let c = &args[2];
                let uniforms: Vec<&Value> = args[3..3 + n_uniforms].iter().collect();
                let m  = arg_i32(&args[3 + n_uniforms], "M")?;
                let k1 = arg_i32(&args[4 + n_uniforms], "K1")?;
                let k2 = arg_i32(&args[5 + n_uniforms], "K2")?;
                let n  = arg_i32(&args[6 + n_uniforms], "N")?;
                if m <= 0 || k1 <= 0 || k2 <= 0 || n <= 0 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: matmul2-fused `{name}` M={m} K1={k1} K2={k2} N={n} must all be positive",
                    )));
                }
                let scalars: Vec<f32> = args[7 + n_uniforms..]
                    .iter()
                    .map(|v| arg_f32(v, "matmul2-fused scalar"))
                    .collect::<Result<_>>()?;
                (a, b, c, uniforms, m as u32, k1 as u32, k2 as u32, n as u32, scalars)
            } else {
                return Err(InterpretError::Unsupported(format!(
                    "metal leaf: matmul2-fused `{name}` expected {want_args_inferred} args \
                     (A + B + C + {n_uniforms} uniforms + {n_scalars} scalars, 2D shape inference) \
                     or {want_args_explicit} args (… + M + K1 + K2 + N), got {}",
                    args.len(),
                )));
            };

            match dtype {
                rce_fusion::codegen::Dtype::F32 =>
                    self.dispatch_matmul_matmul(a, b, c, m, k1, k2, n, &epilogue_refs, &uniforms, &scalars),
                rce_fusion::codegen::Dtype::F16 =>
                    self.dispatch_matmul_matmul_f16(a, b, c, m, k1, k2, n, &epilogue_refs, &uniforms, &scalars),
                rce_fusion::codegen::Dtype::BF16 =>
                    self.dispatch_matmul_matmul_bf16(a, b, c, m, k1, k2, n, &epilogue_refs, &uniforms, &scalars),
            }
        }

        /// Phase I tiled matmul dispatch by name. Same arg conventions
        /// as `dispatch_matmul_epilogue_by_name` (Phase F): inferred
        /// shape via 2D input + 2D weight, or explicit M/K/N scalars.
        /// The only behavioural difference vs Phase F is the kernel
        /// template emitted underneath.
        fn dispatch_matmul_tiled_by_name(&self, name: &str, args: &[Value]) -> Result<Value> {
            let (epilogue, dtype) = rce_fusion::codegen_matmul_tiled::parse_function_name_with_dtype(name).ok_or_else(|| {
                InterpretError::Unsupported(format!(
                    "metal leaf: name `{name}` looks tiled-matmul but doesn't parse",
                ))
            })?;
            let epilogue_refs: Vec<&str> = epilogue.iter().copied().collect();
            let emitted = rce_fusion::codegen_matmul_tiled::emit_matmul_tiled_msl_dtype(&epilogue_refs, dtype)
                .map_err(|e| InterpretError::Unsupported(format!(
                    "metal leaf: tiled-matmul `{name}` emit failed: {e}",
                )))?;
            let n_uniforms = emitted.buffers.len() - 3;
            let n_scalars  = emitted.scalars.len();

            let want_args_inferred = 2 + n_uniforms + n_scalars;
            let want_args_explicit = 5 + n_uniforms + n_scalars;

            let (input, weight, uniforms, m, k, n, scalars) = if args.len() == want_args_inferred {
                let input  = &args[0];
                let weight = &args[1];
                let in_shape = value_shape(input).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: tiled-matmul `{name}` input must be Tensor/DeviceTensor",
                )))?;
                let w_shape  = value_shape(weight).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: tiled-matmul `{name}` weight must be Tensor/DeviceTensor",
                )))?;
                if in_shape.len() != 2 || w_shape.len() != 2 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: tiled-matmul `{name}` inferred-shape form needs 2D input / weight; got {in_shape:?} / {w_shape:?}",
                    )));
                }
                if in_shape[1] != w_shape[0] {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: tiled-matmul `{name}` K mismatch: input.shape[1]={} != weight.shape[0]={}",
                        in_shape[1], w_shape[0],
                    )));
                }
                let m = in_shape[0] as u32;
                let k = in_shape[1] as u32;
                let n = w_shape[1] as u32;
                let uniforms: Vec<&Value> = args[2..2 + n_uniforms].iter().collect();
                let scalars: Vec<f32> = args[2 + n_uniforms..]
                    .iter()
                    .map(|v| arg_f32(v, "tiled-matmul scalar"))
                    .collect::<Result<_>>()?;
                (input, weight, uniforms, m, k, n, scalars)
            } else if args.len() == want_args_explicit {
                let input  = &args[0];
                let weight = &args[1];
                let uniforms: Vec<&Value> = args[2..2 + n_uniforms].iter().collect();
                let m = arg_i32(&args[2 + n_uniforms], "M")?;
                let k = arg_i32(&args[3 + n_uniforms], "K")?;
                let n = arg_i32(&args[4 + n_uniforms], "N")?;
                if m <= 0 || k <= 0 || n <= 0 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: tiled-matmul `{name}` M={m} K={k} N={n} must all be positive",
                    )));
                }
                let scalars: Vec<f32> = args[5 + n_uniforms..]
                    .iter()
                    .map(|v| arg_f32(v, "tiled-matmul scalar"))
                    .collect::<Result<_>>()?;
                (input, weight, uniforms, m as u32, k as u32, n as u32, scalars)
            } else {
                return Err(InterpretError::Unsupported(format!(
                    "metal leaf: tiled-matmul `{name}` expected {want_args_inferred} args \
                     (input + weight + {n_uniforms} uniforms + {n_scalars} scalars, 2D shape inference) \
                     or {want_args_explicit} args (… + M + K + N), got {}",
                    args.len(),
                )));
            };

            match dtype {
                rce_fusion::codegen::Dtype::F32 =>
                    self.dispatch_matmul_tiled(input, weight, m, k, n, &epilogue_refs, &uniforms, &scalars),
                rce_fusion::codegen::Dtype::F16 =>
                    self.dispatch_matmul_tiled_f16(input, weight, m, k, n, &epilogue_refs, &uniforms, &scalars),
                rce_fusion::codegen::Dtype::BF16 =>
                    self.dispatch_matmul_tiled_bf16(input, weight, m, k, n, &epilogue_refs, &uniforms, &scalars),
            }
        }

        /// Phase X: Phase Q (naive) FFN-swiglu dispatch via canonical
        /// name. Routes `fused_ffn_swiglu_f32` / `_f16` / `_bf16` to
        /// the right narrow / f32 entry point. Two arg conventions
        /// (mirroring the matmul `_by_name` dispatchers):
        ///
        ///   Inferred shape: `input: [M, K]`, `gate_W: [K, N]`, `up_W: [K, N]`.
        ///     args = [input, gate_W, up_W]
        ///
        ///   Explicit dims: any shape; M, K, N supplied separately.
        ///     args = [input, gate_W, up_W, M, K, N]
        fn dispatch_ffn_swiglu_by_name(&self, name: &str, args: &[Value]) -> Result<Value> {
            self.dispatch_ffn_swiglu_by_name_inner(name, args, /*tiled*/ false)
        }

        /// Phase X: Phase R (tiled) FFN-swiglu dispatch via canonical
        /// name. Same arg contract as the naive variant.
        fn dispatch_ffn_swiglu_tiled_by_name(&self, name: &str, args: &[Value]) -> Result<Value> {
            self.dispatch_ffn_swiglu_by_name_inner(name, args, /*tiled*/ true)
        }

        fn dispatch_ffn_swiglu_by_name_inner(
            &self,
            name: &str,
            args: &[Value],
            tiled: bool,
        ) -> Result<Value> {
            let label = if tiled { "ffn-swiglu-tiled" } else { "ffn-swiglu" };
            let dtype = if tiled {
                rce_fusion::codegen_ffn_swiglu_tiled::parse_function_name_with_dtype(name)
            } else {
                rce_fusion::codegen_ffn_swiglu::parse_function_name_with_dtype(name)
            }
            .ok_or_else(|| InterpretError::Unsupported(format!(
                "metal leaf: name `{name}` looks {label}-fused but doesn't parse",
            )))?;

            let want_inferred = 3;     // input, gate_W, up_W
            let want_explicit = 6;     // + M, K, N
            let (input, gate_w, up_w, m, k, n) = if args.len() == want_inferred {
                let input = &args[0];
                let gate_w = &args[1];
                let up_w = &args[2];
                let in_shape = value_shape(input).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: {label} `{name}` input must be a Tensor / DeviceTensor",
                )))?;
                let g_shape = value_shape(gate_w).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: {label} `{name}` gate_W must be a Tensor / DeviceTensor",
                )))?;
                let u_shape = value_shape(up_w).ok_or_else(|| InterpretError::Unsupported(format!(
                    "metal leaf: {label} `{name}` up_W must be a Tensor / DeviceTensor",
                )))?;
                if in_shape.len() != 2 || g_shape.len() != 2 || u_shape.len() != 2 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: {label} `{name}` inferred-shape form needs 2D tensors; got shapes \
                         input={in_shape:?} gate_W={g_shape:?} up_W={u_shape:?}",
                    )));
                }
                if in_shape[1] != g_shape[0] || g_shape != u_shape {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: {label} `{name}` shape mismatch: \
                         input[M,K]={in_shape:?}, gate_W[K,N]={g_shape:?}, up_W[K,N]={u_shape:?}",
                    )));
                }
                let m = in_shape[0] as u32;
                let k = in_shape[1] as u32;
                let n = g_shape[1] as u32;
                (input, gate_w, up_w, m, k, n)
            } else if args.len() == want_explicit {
                let m = arg_i32(&args[3], "M")?;
                let k = arg_i32(&args[4], "K")?;
                let n = arg_i32(&args[5], "N")?;
                if m <= 0 || k <= 0 || n <= 0 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal leaf: {label} `{name}` M={m} K={k} N={n} must all be positive",
                    )));
                }
                (&args[0], &args[1], &args[2], m as u32, k as u32, n as u32)
            } else {
                return Err(InterpretError::Unsupported(format!(
                    "metal leaf: {label} `{name}` expected {want_inferred} args \
                     (input + gate_W + up_W, with 2D shape inference) \
                     or {want_explicit} args (… + M + K + N), got {}",
                    args.len(),
                )));
            };

            match (tiled, dtype) {
                (false, rce_fusion::codegen::Dtype::F32) =>
                    self.dispatch_ffn_swiglu(input, gate_w, up_w, m, k, n),
                (false, rce_fusion::codegen::Dtype::F16) =>
                    self.dispatch_ffn_swiglu_f16(input, gate_w, up_w, m, k, n),
                (false, rce_fusion::codegen::Dtype::BF16) =>
                    self.dispatch_ffn_swiglu_bf16(input, gate_w, up_w, m, k, n),
                (true, rce_fusion::codegen::Dtype::F32) =>
                    self.dispatch_ffn_swiglu_tiled(input, gate_w, up_w, m, k, n),
                (true, rce_fusion::codegen::Dtype::F16) =>
                    self.dispatch_ffn_swiglu_tiled_f16(input, gate_w, up_w, m, k, n),
                (true, rce_fusion::codegen::Dtype::BF16) =>
                    self.dispatch_ffn_swiglu_tiled_bf16(input, gate_w, up_w, m, k, n),
            }
        }

        // ── Custom("silu") + Custom("gelu") ────────────────
        // Both are pure elementwise unary f32 ops: same pipeline shape,
        // different math. Share a helper.
        fn silu(&self, args: &[Value]) -> Result<Value> {
            self.elementwise_unary(args, "silu_f32", "silu")
        }

        fn elementwise_unary(
            &self,
            args: &[Value],
            pipeline_name: &'static str,
            op_label: &str,
        ) -> Result<Value> {
            if args.len() != 1 {
                return Err(InterpretError::Unsupported(format!(
                    "metal {op_label}: expected 1 arg, got {}",
                    args.len()
                )));
            }

            // Two paths:
            // - HOST input: copy bytes in, run kernel, copy bytes out → HostTensor.
            // - DEVICE input on Metal: skip both copies → DeviceTensor.
            match &args[0] {
                Value::Tensor(t) => {
                    if t.dtype != DType::F32 {
                        return Err(InterpretError::Unsupported(format!(
                            "metal {op_label}: f32 only, got {}",
                            t.dtype.name()
                        )));
                    }
                    let n = t.numel();
                    if n == 0 {
                        return HostTensor::new(t.shape.clone(), DType::F32, vec![])
                            .map(Value::Tensor)
                            .map_err(map_build_err);
                    }
                    let in_buf = self.device.new_buffer_with_data(
                        t.data.as_ptr() as *const _,
                        t.data.len() as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    let out_buf = self.device.new_buffer(
                        (n * 4) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    self.dispatch_unary(pipeline_name, op_label, &in_buf, &out_buf, n)?;
                    let bytes = read_buffer_bytes(&out_buf, n * 4);
                    HostTensor::new(t.shape.clone(), DType::F32, bytes)
                        .map(Value::Tensor)
                        .map_err(map_build_err)
                }
                Value::DeviceTensor(dt) => {
                    if dt.device != DeviceId::Metal {
                        return Err(InterpretError::Unsupported(format!(
                            "metal {op_label}: DeviceTensor on {} not Metal",
                            dt.device.name()
                        )));
                    }
                    if dt.dtype != DType::F32 {
                        return Err(InterpretError::Unsupported(format!(
                            "metal {op_label}: f32 only, got {}",
                            dt.dtype.name()
                        )));
                    }
                    let metal_mem = dt
                        .mem
                        .as_any()
                        .downcast_ref::<MetalDeviceMem>()
                        .ok_or_else(|| InterpretError::Unsupported(format!(
                            "metal {op_label}: DeviceTensor not from MetalLeaf"
                        )))?;
                    let n = dt.numel();
                    if n == 0 {
                        let empty = self.device.new_buffer(0, MTLResourceOptions::StorageModeShared);
                        return Ok(Value::DeviceTensor(DeviceTensor::new(
                            DeviceId::Metal,
                            DType::F32,
                            dt.shape.clone(),
                            Arc::new(MetalDeviceMem { buffer: empty, size_bytes: 0 }),
                        )));
                    }
                    let out_size = (n * 4) as u64;
                    let out_buf = self.device.new_buffer(
                        out_size,
                        MTLResourceOptions::StorageModeShared,
                    );
                    self.dispatch_unary(pipeline_name, op_label, &metal_mem.buffer, &out_buf, n)?;
                    Ok(Value::DeviceTensor(DeviceTensor::new(
                        DeviceId::Metal,
                        DType::F32,
                        dt.shape.clone(),
                        Arc::new(MetalDeviceMem {
                            buffer: out_buf,
                            size_bytes: out_size as usize,
                        }),
                    )))
                }
                other => Err(InterpretError::Unsupported(format!(
                    "metal {op_label}: arg 0 expected tensor, got {other:?}"
                ))),
            }
        }

        fn dispatch_unary(
            &self,
            pipeline_name: &'static str,
            op_label: &str,
            in_buf: &Buffer,
            out_buf: &Buffer,
            n: usize,
        ) -> Result<()> {
            let _ = op_label;
            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get(pipeline_name)
                .ok_or_else(|| missing(pipeline_name))?;

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf), 0);
            encoder.set_buffer(1, Some(out_buf), 0);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg_width = max_threads.min(n).max(1);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
            Ok(())
        }

        // ── Custom("swiglu") ───────────────────────────────
        //
        // Args: gate (tensor), up (tensor). Both same shape and length.
        // Output shape = gate.shape; output[i] = silu(gate[i]) * up[i].
        // DeviceTensor flow: if either input is DeviceTensor, output is too.
        fn swiglu(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 2 {
                return Err(InterpretError::Unsupported(format!(
                    "metal swiglu: expected 2 args (gate, up), got {}",
                    args.len()
                )));
            }
            let (g_buf, gate_shape, n_gate, gate_dev) =
                buffer_for_f32_input(&self.device, &args[0], "swiglu", "gate")?;
            let (u_buf, _, n_up, up_dev) =
                buffer_for_f32_input(&self.device, &args[1], "swiglu", "up")?;
            if n_gate != n_up {
                return Err(InterpretError::Unsupported(format!(
                    "metal swiglu: gate numel {n_gate} != up numel {n_up}"
                )));
            }
            let n = n_gate;
            if n == 0 {
                return finalize_output(
                    self.device.new_buffer(0, MTLResourceOptions::StorageModeShared),
                    0,
                    gate_shape,
                    gate_dev || up_dev,
                );
            }
            let out_size = (n * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines.get("swiglu_f32").ok_or_else(|| missing("swiglu_f32"))?;
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(g_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(u_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg_width = max_threads.min(n).max(1);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            finalize_output(out_buf, out_size as usize, gate_shape, gate_dev || up_dev)
        }

        // ── Custom("layer_norm") ───────────────────────────
        //
        // Args: x, weight, bias, ngroups (i32), eps (f32).
        // Group-aware: ngroups=1 = standard whole-vector LayerNorm.
        // DeviceTensor flow: if any tensor input is DeviceTensor, output is too.
        fn layer_norm(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 5 {
                return Err(InterpretError::Unsupported(format!(
                    "metal layer_norm: expected 5 args (x, weight, bias, ngroups, eps), got {}",
                    args.len()
                )));
            }
            let (in_buf, x_shape, total, x_dev) =
                buffer_for_f32_input(&self.device, &args[0], "layer_norm", "x")?;
            let (w_buf, _, w_total, w_dev) =
                buffer_for_f32_input(&self.device, &args[1], "layer_norm", "weight")?;
            let (b_buf, _, b_total, b_dev) =
                buffer_for_f32_input(&self.device, &args[2], "layer_norm", "bias")?;
            let ngroups = arg_i32(&args[3], "ngroups")?;
            let eps = arg_f32(&args[4], "eps")?;
            if w_total != total || b_total != total {
                return Err(InterpretError::Unsupported(format!(
                    "metal layer_norm: weight/bias numel ({w_total}, {b_total}) != x numel {total}"
                )));
            }
            if ngroups <= 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal layer_norm: ngroups must be positive, got {ngroups}"
                )));
            }
            let ngroups_u = ngroups as u32;
            if total % (ngroups_u as usize) != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal layer_norm: total {total} not divisible by ngroups {ngroups_u}"
                )));
            }
            let group_size = (total / ngroups_u as usize) as u32;

            let out_size = (total * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("layer_norm_f32_grouped")
                .ok_or_else(|| missing("layer_norm_f32_grouped"))?;

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(w_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(b_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4, 4, &ngroups_u as *const u32 as *const _);
            encoder.set_bytes(5, 4, &group_size as *const u32 as *const _);
            encoder.set_bytes(6, 4, &eps as *const f32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let dispatch_threads = ngroups_u as usize;
            let tg_width = max_threads.min(dispatch_threads).max(1);
            encoder.dispatch_threads(
                MTLSize { width: dispatch_threads as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(total, ngroups = ngroups_u, group_size, "metal layer_norm dispatched");
            finalize_output(out_buf, out_size as usize, x_shape, x_dev || w_dev || b_dev)
        }

        // ── Custom("rms_norm") ─────────────────────────────
        //
        // Args contract (matches rce-op-nn-prims::RmsNormLeaf):
        //   1. x        f32 tensor (1-D, length `total`)
        //   2. weight   f32 tensor of shape `(total,)`
        //   3. ngroups  i32 scalar
        //   4. eps      f32 scalar
        // DeviceTensor flow: if either tensor is DeviceTensor, output is too.
        fn rms_norm(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 4 {
                return Err(InterpretError::Unsupported(format!(
                    "metal rms_norm: expected 4 args (x, weight, ngroups, eps), got {}",
                    args.len()
                )));
            }
            let (in_buf, x_shape, total, x_dev) =
                buffer_for_f32_input(&self.device, &args[0], "rms_norm", "x")?;
            let (w_buf, _, w_total, w_dev) =
                buffer_for_f32_input(&self.device, &args[1], "rms_norm", "weight")?;
            let ngroups = arg_i32(&args[2], "ngroups")?;
            let eps = arg_f32(&args[3], "eps")?;
            if w_total != total {
                return Err(InterpretError::Unsupported(format!(
                    "metal rms_norm: weight numel {w_total} != x numel {total}"
                )));
            }
            if ngroups <= 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal rms_norm: ngroups must be positive, got {ngroups}"
                )));
            }
            let ngroups_u = ngroups as u32;
            if total % (ngroups_u as usize) != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal rms_norm: total {total} not divisible by ngroups {ngroups_u}"
                )));
            }
            let group_size = (total / ngroups_u as usize) as u32;

            let out_size = (total * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("rms_norm_f32_grouped")
                .ok_or_else(|| missing("rms_norm_f32_grouped"))?;

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(w_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            encoder.set_bytes(3, 4, &ngroups_u as *const u32 as *const _);
            encoder.set_bytes(4, 4, &group_size as *const u32 as *const _);
            encoder.set_bytes(5, 4, &eps as *const f32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let dispatch_threads = ngroups_u as usize;
            let tg_width = max_threads.min(dispatch_threads).max(1);
            encoder.dispatch_threads(
                MTLSize { width: dispatch_threads as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(total, ngroups = ngroups_u, group_size, "metal rms_norm dispatched");
            finalize_output(out_buf, out_size as usize, x_shape, x_dev || w_dev)
        }

        // ── Custom("silu_rms_norm") ────────────────────────
        //
        // First fused-Free-chain op. Equivalent to applying `silu`
        // then `rms_norm` to the same (x, weight, ngroups, eps) tuple
        // — one Metal dispatch instead of two. The intermediate silu
        // tensor is never materialised in memory; the kernel re-
        // evaluates silu in pass 2 because the intermediate doesn't
        // escape the thread.
        //
        // Same arg contract as `rms_norm`:
        //   1. x        f32 tensor (1-D, length `total`)
        //   2. weight   f32 tensor of shape `(total,)`
        //   3. ngroups  i32 scalar
        //   4. eps      f32 scalar
        // DeviceTensor flow: if either tensor is DeviceTensor, output is too.
        fn silu_rms_norm(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 4 {
                return Err(InterpretError::Unsupported(format!(
                    "metal silu_rms_norm: expected 4 args (x, weight, ngroups, eps), got {}",
                    args.len()
                )));
            }
            let (in_buf, x_shape, total, x_dev) =
                buffer_for_f32_input(&self.device, &args[0], "silu_rms_norm", "x")?;
            let (w_buf, _, w_total, w_dev) =
                buffer_for_f32_input(&self.device, &args[1], "silu_rms_norm", "weight")?;
            let ngroups = arg_i32(&args[2], "ngroups")?;
            let eps = arg_f32(&args[3], "eps")?;
            if w_total != total {
                return Err(InterpretError::Unsupported(format!(
                    "metal silu_rms_norm: weight numel {w_total} != x numel {total}"
                )));
            }
            if ngroups <= 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal silu_rms_norm: ngroups must be positive, got {ngroups}"
                )));
            }
            let ngroups_u = ngroups as u32;
            if total % (ngroups_u as usize) != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal silu_rms_norm: total {total} not divisible by ngroups {ngroups_u}"
                )));
            }
            let group_size = (total / ngroups_u as usize) as u32;

            let out_size = (total * 4) as u64;
            let out_buf = self.device.new_buffer(out_size, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("silu_rms_norm_f32_grouped")
                .ok_or_else(|| missing("silu_rms_norm_f32_grouped"))?;

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(w_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            encoder.set_bytes(3, 4, &ngroups_u as *const u32 as *const _);
            encoder.set_bytes(4, 4, &group_size as *const u32 as *const _);
            encoder.set_bytes(5, 4, &eps as *const f32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let dispatch_threads = ngroups_u as usize;
            let tg_width = max_threads.min(dispatch_threads).max(1);
            encoder.dispatch_threads(
                MTLSize { width: dispatch_threads as u64, height: 1, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(total, ngroups = ngroups_u, group_size, "metal silu_rms_norm dispatched (fused)");
            finalize_output(out_buf, out_size as usize, x_shape, x_dev || w_dev)
        }

        // ── Custom("attention") ────────────────────────────
        //
        // Args contract matches `rce-op-attention::AttentionLeaf`:
        //   1. Q              f32 [B, D]
        //   2. K              f32 [S, D]
        //   3. V              f32 [S, D]
        //   4. scale          f32 scalar
        //   5. sliding_window i32 scalar (0 = full attention)
        // Returns: f32 [B, D]. DeviceTensor flow: any DeviceTensor input → DeviceTensor output.
        fn attention(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 5 {
                return Err(InterpretError::Unsupported(format!(
                    "metal attention: expected 5 args, got {}",
                    args.len()
                )));
            }
            let (q_buf, q_shape, _q_n, q_dev) =
                buffer_for_f32_input(&self.device, &args[0], "attention", "Q")?;
            let (k_buf, k_shape, _k_n, k_dev) =
                buffer_for_f32_input(&self.device, &args[1], "attention", "K")?;
            let (v_buf, v_shape, _v_n, v_dev) =
                buffer_for_f32_input(&self.device, &args[2], "attention", "V")?;
            let scale = arg_f32(&args[3], "scale")?;
            let sliding = arg_i32(&args[4], "sliding_window")?;
            let on_device = q_dev || k_dev || v_dev;

            if q_shape.len() != 2 || k_shape.len() != 2 || v_shape.len() != 2 {
                return Err(InterpretError::Unsupported(format!(
                    "metal attention: rank-2 only (Q={:?}, K={:?}, V={:?})",
                    q_shape, k_shape, v_shape
                )));
            }
            let (batch, head_dim) = (q_shape[0], q_shape[1]);
            let (seq_len, k_dim) = (k_shape[0], k_shape[1]);
            let (v_seq, v_dim) = (v_shape[0], v_shape[1]);
            if k_dim != head_dim || v_dim != head_dim {
                return Err(InterpretError::Unsupported(format!(
                    "metal attention: head_dim mismatch (Q={head_dim}, K={k_dim}, V={v_dim})"
                )));
            }
            if seq_len != v_seq {
                return Err(InterpretError::Unsupported(format!(
                    "metal attention: K seq_len {seq_len} != V seq_len {v_seq}"
                )));
            }
            if batch == 0 || head_dim == 0 {
                return Err(InterpretError::Unsupported(
                    "metal attention: zero-size dim, fall through".into(),
                ));
            }

            let (kv_start, active_len) = if sliding > 0 && (sliding as usize) < seq_len {
                let s = sliding as usize;
                (seq_len - s, s)
            } else {
                (0usize, seq_len)
            };

            let out_size_bytes = batch * head_dim * 4;
            if active_len == 0 {
                // Zero-length output buffer.
                let zero_buf = self.device.new_buffer(
                    out_size_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                return finalize_output(
                    zero_buf,
                    out_size_bytes,
                    vec![batch, head_dim],
                    on_device,
                );
            }

            let out_buf = self.device.new_buffer(
                out_size_bytes as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("attention_f32")
                .ok_or_else(|| missing("attention_f32"))?;

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(q_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(k_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(v_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            let head_dim_u = head_dim as u32;
            let seq_len_u = seq_len as u32;
            let kv_start_u = kv_start as u32;
            let active_len_u = active_len as u32;
            encoder.set_bytes(4, 4, &head_dim_u as *const u32 as *const _);
            encoder.set_bytes(5, 4, &seq_len_u as *const u32 as *const _);
            encoder.set_bytes(6, 4, &kv_start_u as *const u32 as *const _);
            encoder.set_bytes(7, 4, &active_len_u as *const u32 as *const _);
            encoder.set_bytes(8, 4, &scale as *const f32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg_width = max_threads.min(head_dim).max(1);
            encoder.dispatch_threads(
                MTLSize { width: head_dim as u64, height: batch as u64, depth: 1 },
                MTLSize { width: tg_width as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(batch, head_dim, seq_len, active_len, "metal attention dispatched");
            finalize_output(out_buf, out_size_bytes, vec![batch, head_dim], on_device)
        }

        // ── Custom("embedding_lookup") ─────────────────────
        //
        // Args: embed_table f32 [V, H], token_ids i32 [S], hidden_size i32.
        // Output: f32 [S, H]. DeviceTensor flow: if `embed_table` is a
        // DeviceTensor, the output stays on device (the big tensor —
        // vocab×hidden = up to ~400M elems on Llama-3-70B). `token_ids`
        // is i32 and small (per-call only); we don't bother routing it
        // device-resident.
        fn embedding_lookup(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 3 {
                return Err(InterpretError::Unsupported(format!(
                    "metal embedding_lookup: expected 3 args, got {}",
                    args.len()
                )));
            }
            let (table_buf, _table_shape, table_total, table_dev) =
                buffer_for_f32_input(&self.device, &args[0], "embedding_lookup", "embed_table")?;

            // token_ids: HostTensor only (i32). Upload bytes.
            let ids = arg_tensor(&args[1], "token_ids")?;
            if ids.dtype != DType::I32 {
                return Err(InterpretError::Unsupported(format!(
                    "metal embedding_lookup: token_ids i32 only, got {} (CPU leaf will pick i64)",
                    ids.dtype.name()
                )));
            }
            let hidden = arg_i32(&args[2], "hidden_size")?;
            if hidden <= 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal embedding_lookup: hidden_size must be positive, got {hidden}"
                )));
            }
            let hidden_u = hidden as u32;
            if table_total % (hidden_u as usize) != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal embedding_lookup: embed_table numel {table_total} not divisible by hidden_size {hidden_u}"
                )));
            }
            let vocab_u = (table_total / hidden_u as usize) as u32;
            let seq_len_u = ids.numel() as u32;

            let ids_buf = self.device.new_buffer_with_data(
                ids.data.as_ptr() as *const _,
                ids.data.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_size_bytes = seq_len_u as usize * hidden_u as usize * 4;
            let out_buf = self
                .device
                .new_buffer(out_size_bytes as u64, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("embedding_lookup_f32")
                .ok_or_else(|| missing("embedding_lookup_f32"))?;

            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(table_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&ids_buf), 0);
            encoder.set_buffer(2, Some(&out_buf), 0);
            encoder.set_bytes(3, 4, &vocab_u as *const u32 as *const _);
            encoder.set_bytes(4, 4, &hidden_u as *const u32 as *const _);
            encoder.set_bytes(5, 4, &seq_len_u as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let dispatch = seq_len_u as usize;
            let tg = max_threads.min(dispatch).max(1);
            encoder.dispatch_threads(
                MTLSize { width: dispatch as u64, height: 1, depth: 1 },
                MTLSize { width: tg as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            finalize_output(
                out_buf,
                out_size_bytes,
                vec![seq_len_u as usize, hidden_u as usize],
                table_dev,
            )
        }

        // ── Custom("moe_combine") ──────────────────────────
        //
        // Args: gate_logits f32 [E], expert_outputs f32 [E, D], top_k i32.
        // Output: f32 [D] = Σ_{k∈topK} softmax(gate over topK)[k] * experts[k].
        //
        // Top-K + softmax weights are computed on the host (small N,
        // requires sort-like behavior); only the weighted reduce runs
        // on the GPU. DeviceTensor flow: if `expert_outputs` (the big
        // tensor) is a DeviceTensor, the output stays on device.
        // gate_logits is small (E ≤ 256) so we always read it host-side
        // — even when it arrives as DeviceTensor we copy E·4 bytes back.
        fn moe_combine(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 3 {
                return Err(InterpretError::Unsupported(format!(
                    "metal moe_combine: expected 3 args, got {}",
                    args.len()
                )));
            }
            let logits_f: Vec<f32> = match &args[0] {
                Value::Tensor(t) => {
                    if t.dtype != DType::F32 {
                        return Err(InterpretError::Unsupported(format!(
                            "metal moe_combine: gate_logits f32 only, got {}", t.dtype.name()
                        )));
                    }
                    bytemuck::try_cast_slice::<u8, f32>(&t.data)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal moe_combine: gate_logits not f32-aligned ({e})"
                        )))?
                        .to_vec()
                }
                Value::DeviceTensor(dt) => {
                    if dt.device != DeviceId::Metal || dt.dtype != DType::F32 {
                        return Err(InterpretError::Unsupported(format!(
                            "metal moe_combine: gate_logits DeviceTensor must be Metal/f32 (got {}/{})",
                            dt.device.name(), dt.dtype.name()
                        )));
                    }
                    let metal_mem = dt.mem.as_any().downcast_ref::<MetalDeviceMem>()
                        .ok_or_else(|| InterpretError::Unsupported(
                            "metal moe_combine: gate_logits DeviceTensor not from MetalLeaf".into()
                        ))?;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            metal_mem.buffer.contents() as *const u8,
                            metal_mem.size_bytes,
                        )
                    };
                    bytemuck::try_cast_slice::<u8, f32>(bytes)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal moe_combine: gate_logits device bytes not f32-aligned ({e})"
                        )))?
                        .to_vec()
                }
                other => {
                    return Err(InterpretError::Unsupported(format!(
                        "metal moe_combine: gate_logits expected tensor, got {other:?}"
                    )));
                }
            };
            let (experts_buf, experts_shape, experts_total, experts_dev) =
                buffer_for_f32_input(&self.device, &args[1], "moe_combine", "expert_outputs")?;
            let top_k = arg_i32(&args[2], "top_k")?;

            let num_experts = logits_f.len();
            if num_experts == 0 {
                return Err(InterpretError::Unsupported(
                    "metal moe_combine: num_experts == 0".into(),
                ));
            }
            if experts_total % num_experts != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal moe_combine: expert_outputs numel {experts_total} not divisible by num_experts {num_experts}"
                )));
            }
            let d_model = experts_total / num_experts;
            if top_k <= 0 || (top_k as usize) > num_experts {
                return Err(InterpretError::Unsupported(format!(
                    "metal moe_combine: top_k must be 1..={num_experts}, got {top_k}"
                )));
            }
            let top_k = top_k as usize;
            let _ = experts_shape;

            // Host-side top-K + softmax (matches CPU leaf semantics).
            let mut idx: Vec<u32> = (0..num_experts as u32).collect();
            idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
                logits_f[b as usize]
                    .partial_cmp(&logits_f[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(top_k);

            let max_l = idx.iter().map(|&i| logits_f[i as usize]).fold(f32::NEG_INFINITY, f32::max);
            let mut weights = vec![0.0f32; top_k];
            let mut sum = 0.0f32;
            for (slot, &i) in idx.iter().enumerate() {
                let e = (logits_f[i as usize] - max_l).exp();
                weights[slot] = e;
                sum += e;
            }
            for w in weights.iter_mut() { *w /= sum; }
            let idx_bytes: &[u8] = bytemuck::cast_slice(&idx);
            let idx_buf = self.device.new_buffer_with_data(
                idx_bytes.as_ptr() as *const _,
                idx_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let w_bytes: &[u8] = bytemuck::cast_slice(&weights);
            let w_buf = self.device.new_buffer_with_data(
                w_bytes.as_ptr() as *const _,
                w_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_size = d_model * 4;
            let out_buf = self.device.new_buffer(
                out_size as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("moe_combine_f32")
                .ok_or_else(|| missing("moe_combine_f32"))?;
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(experts_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&idx_buf), 0);
            encoder.set_buffer(2, Some(&w_buf), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            let d_u = d_model as u32;
            let k_u = top_k as u32;
            encoder.set_bytes(4, 4, &d_u as *const u32 as *const _);
            encoder.set_bytes(5, 4, &k_u as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg = max_threads.min(d_model).max(1);
            encoder.dispatch_threads(
                MTLSize { width: d_model as u64, height: 1, depth: 1 },
                MTLSize { width: tg as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            finalize_output(out_buf, out_size, vec![d_model], experts_dev)
        }

        // ── Custom("group_norm") ──────────────────────────
        //
        // Args: x f32 [N, C, HW] (3-D), gamma f32 [C], bias f32 [C],
        //       N i32, G i32, eps f32.
        // C and HW are inferred from x.shape: x = [N, C, HW].
        // Output: f32 [N, C, HW]. DeviceTensor flow: any DeviceTensor input → DeviceTensor output.
        fn group_norm(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 6 {
                return Err(InterpretError::Unsupported(format!(
                    "metal group_norm: expected 6 args (x, gamma, bias, N, G, eps), got {}",
                    args.len()
                )));
            }
            let (x_buf, x_shape, x_total, x_dev) =
                buffer_for_f32_input(&self.device, &args[0], "group_norm", "x")?;
            let (g_buf, _gs, g_total, g_dev) =
                buffer_for_f32_input(&self.device, &args[1], "group_norm", "gamma")?;
            let (b_buf, _bs, b_total, b_dev) =
                buffer_for_f32_input(&self.device, &args[2], "group_norm", "bias")?;
            let n_arg = arg_i32(&args[3], "N")?;
            let g_arg = arg_i32(&args[4], "G")?;
            let eps = arg_f32(&args[5], "eps")?;
            if x_shape.len() != 3 {
                return Err(InterpretError::Unsupported(format!(
                    "metal group_norm: x must be 3-D [N, C, HW], got shape {x_shape:?}"
                )));
            }
            let n_x = x_shape[0];
            let c = x_shape[1];
            let hw = x_shape[2];
            if n_arg <= 0 || g_arg <= 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal group_norm: N and G must be positive (got N={n_arg}, G={g_arg})"
                )));
            }
            if n_arg as usize != n_x {
                return Err(InterpretError::Unsupported(format!(
                    "metal group_norm: N arg {n_arg} != x.shape[0] {n_x}"
                )));
            }
            let g = g_arg as usize;
            if c % g != 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal group_norm: C={c} not divisible by G={g}"
                )));
            }
            if g_total != c || b_total != c {
                return Err(InterpretError::Unsupported(format!(
                    "metal group_norm: gamma/bias numel ({g_total}, {b_total}) != C {c}"
                )));
            }
            let _ = x_total;

            let on_device = x_dev || g_dev || b_dev;
            let out_size = n_x * c * hw * 4;
            let out_buf = self
                .device
                .new_buffer(out_size as u64, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("group_norm_f32")
                .ok_or_else(|| missing("group_norm_f32"))?;
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(x_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(g_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(b_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            let n_u = n_x as u32;
            let g_u = g as u32;
            let c_u = c as u32;
            let hw_u = hw as u32;
            encoder.set_bytes(4, 4, &n_u as *const u32 as *const _);
            encoder.set_bytes(5, 4, &g_u as *const u32 as *const _);
            encoder.set_bytes(6, 4, &c_u as *const u32 as *const _);
            encoder.set_bytes(7, 4, &hw_u as *const u32 as *const _);
            encoder.set_bytes(8, 4, &eps as *const f32 as *const _);

            // Dispatch (G, N) — one thread per (n, g) pair.
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg_w = max_threads.min(g).max(1);
            encoder.dispatch_threads(
                MTLSize { width: g as u64, height: n_x as u64, depth: 1 },
                MTLSize { width: tg_w as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(N = n_x, G = g, C = c, HW = hw, "metal group_norm dispatched");
            finalize_output(out_buf, out_size, vec![n_x, c, hw], on_device)
        }

        // ── Custom("topk_mask") ──────────────────────────
        //
        // Args: logits f32 [N], top_k i32. Output: f32 [N] with 1.0
        // at the top-K positions and 0.0 elsewhere. Top-K selection
        // is host-side (small N, requires partial sort); the GPU
        // writes the final mask buffer.
        fn topk_mask(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 2 {
                return Err(InterpretError::Unsupported(format!(
                    "metal topk_mask: expected 2 args (logits, top_k), got {}",
                    args.len()
                )));
            }
            let (logits_buf, logits_shape, n, logits_dev) =
                buffer_for_f32_input(&self.device, &args[0], "topk_mask", "logits")?;
            let _ = logits_buf;
            let _ = logits_shape;
            let top_k = arg_i32(&args[1], "top_k")?;
            if top_k <= 0 || (top_k as usize) > n {
                return Err(InterpretError::Unsupported(format!(
                    "metal topk_mask: top_k must be 1..={n}, got {top_k}"
                )));
            }
            let top_k = top_k as usize;

            // Read logits to host for the partial sort (cheap — small N).
            let logits_f: Vec<f32> = match &args[0] {
                Value::Tensor(t) => {
                    bytemuck::try_cast_slice::<u8, f32>(&t.data)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal topk_mask: logits not f32-aligned ({e})"
                        )))?
                        .to_vec()
                }
                Value::DeviceTensor(dt) => {
                    let metal_mem = dt.mem.as_any().downcast_ref::<MetalDeviceMem>()
                        .ok_or_else(|| InterpretError::Unsupported(
                            "metal topk_mask: logits DeviceTensor not from MetalLeaf".into()
                        ))?;
                    let bytes = unsafe {
                        std::slice::from_raw_parts(
                            metal_mem.buffer.contents() as *const u8,
                            metal_mem.size_bytes,
                        )
                    };
                    bytemuck::try_cast_slice::<u8, f32>(bytes)
                        .map_err(|e| InterpretError::Unsupported(format!(
                            "metal topk_mask: logits device bytes not f32-aligned ({e})"
                        )))?
                        .to_vec()
                }
                _ => unreachable!(),
            };
            let mut idx: Vec<u32> = (0..n as u32).collect();
            idx.select_nth_unstable_by(top_k - 1, |&a, &b| {
                logits_f[b as usize]
                    .partial_cmp(&logits_f[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            idx.truncate(top_k);

            let idx_bytes: &[u8] = bytemuck::cast_slice(&idx);
            let idx_buf = self.device.new_buffer_with_data(
                idx_bytes.as_ptr() as *const _,
                idx_bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_size = n * 4;
            let out_buf = self.device.new_buffer(
                out_size as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("topk_mask_f32")
                .ok_or_else(|| missing("topk_mask_f32"))?;
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&idx_buf), 0);
            encoder.set_buffer(1, Some(&out_buf), 0);
            let n_u = n as u32;
            let k_u = top_k as u32;
            encoder.set_bytes(2, 4, &n_u as *const u32 as *const _);
            encoder.set_bytes(3, 4, &k_u as *const u32 as *const _);
            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg = max_threads.min(n).max(1);
            encoder.dispatch_threads(
                MTLSize { width: n as u64, height: 1, depth: 1 },
                MTLSize { width: tg as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            finalize_output(out_buf, out_size, vec![n], logits_dev)
        }

        // ── Custom("argmax") ───────────────────────────────
        //
        // Args: input f32 [N]. Output: i32 [1] = argmax index.
        //
        // No DeviceTensor flow on output — argmax produces an integer
        // index that callers usually want host-side (token sampling,
        // greedy decoding). DeviceTensor input is accepted via the
        // helper.
        fn argmax(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 1 {
                return Err(InterpretError::Unsupported(format!(
                    "metal argmax: expected 1 arg, got {}", args.len()
                )));
            }
            let (in_buf, _shape, n, _on_device) =
                buffer_for_f32_input(&self.device, &args[0], "argmax", "input")?;
            if n == 0 {
                return Err(InterpretError::Unsupported(
                    "metal argmax: empty input".into(),
                ));
            }

            let out_buf = self
                .device
                .new_buffer(4, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("argmax_f32")
                .ok_or_else(|| missing("argmax_f32"))?;
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(&out_buf), 0);
            let n_u = n as u32;
            encoder.set_bytes(2, 4, &n_u as *const u32 as *const _);
            encoder.dispatch_threads(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 1, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            let raw = out_buf.contents() as *const u32;
            let idx = unsafe { *raw } as i32;
            HostTensor::from_i32(vec![1], &[idx])
                .map(Value::Tensor)
                .map_err(map_build_err)
        }

        // ── Custom("conv2d") ───────────────────────────────
        //
        // Args (14): input, weight, bias, N, H, W, C_in, C_out, K_h,
        // K_w, S_h, S_w, P_h, P_w. Layouts: input [N, H, W, C_in]
        // (NHWC), weight [C_out, K_h, K_w, C_in] (OHWI), bias [C_out].
        // Output: f32 [N, H_out, W_out, C_out]. DeviceTensor flow on.
        fn conv2d(&self, args: &[Value]) -> Result<Value> {
            if args.len() != 14 {
                return Err(InterpretError::Unsupported(format!(
                    "metal conv2d: expected 14 args, got {}",
                    args.len()
                )));
            }
            let (in_buf, _is, _, in_dev)   = buffer_for_f32_input(&self.device, &args[0], "conv2d", "input")?;
            let (w_buf, _ws, _, w_dev)     = buffer_for_f32_input(&self.device, &args[1], "conv2d", "weight")?;
            let (bias_buf, _bs, _, b_dev)  = buffer_for_f32_input(&self.device, &args[2], "conv2d", "bias")?;
            let n     = arg_i32(&args[3],  "N")?;
            let h     = arg_i32(&args[4],  "H")?;
            let w     = arg_i32(&args[5],  "W")?;
            let c_in  = arg_i32(&args[6],  "C_in")?;
            let c_out = arg_i32(&args[7],  "C_out")?;
            let k_h   = arg_i32(&args[8],  "K_h")?;
            let k_w   = arg_i32(&args[9],  "K_w")?;
            let s_h   = arg_i32(&args[10], "S_h")?;
            let s_w   = arg_i32(&args[11], "S_w")?;
            let p_h   = arg_i32(&args[12], "P_h")?;
            let p_w   = arg_i32(&args[13], "P_w")?;
            for (label, v) in [("N", n), ("H", h), ("W", w), ("C_in", c_in), ("C_out", c_out), ("K_h", k_h), ("K_w", k_w), ("S_h", s_h), ("S_w", s_w)] {
                if v <= 0 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal conv2d: {label} must be positive, got {v}"
                    )));
                }
            }
            if p_h < 0 || p_w < 0 {
                return Err(InterpretError::Unsupported(format!(
                    "metal conv2d: padding must be non-negative (got P_h={p_h}, P_w={p_w})"
                )));
            }
            let n_u = n as u32; let h_u = h as u32; let w_u = w as u32;
            let cin_u = c_in as u32; let cout_u = c_out as u32;
            let kh_u = k_h as u32; let kw_u = k_w as u32;
            let sh_u = s_h as u32; let sw_u = s_w as u32;
            let ph_u = p_h as u32; let pw_u = p_w as u32;
            let h_out_u = (h_u + 2 * ph_u).saturating_sub(kh_u) / sh_u + 1;
            let w_out_u = (w_u + 2 * pw_u).saturating_sub(kw_u) / sw_u + 1;
            let on_device = in_dev || w_dev || b_dev;

            let out_total = (n_u as usize) * (h_out_u as usize) * (w_out_u as usize) * (cout_u as usize);
            let out_size = out_total * 4;
            let out_buf = self.device.new_buffer(out_size as u64, MTLResourceOptions::StorageModeShared);

            let pipelines = self.pipelines.lock().expect("metal pipeline mutex");
            let pipeline = pipelines
                .get("conv2d_naive_f32")
                .ok_or_else(|| missing("conv2d_naive_f32"))?;
            let cmd_buf = self.queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(in_buf.as_buf()), 0);
            encoder.set_buffer(1, Some(w_buf.as_buf()), 0);
            encoder.set_buffer(2, Some(bias_buf.as_buf()), 0);
            encoder.set_buffer(3, Some(&out_buf), 0);
            encoder.set_bytes(4,  4, &n_u    as *const u32 as *const _);
            encoder.set_bytes(5,  4, &h_u    as *const u32 as *const _);
            encoder.set_bytes(6,  4, &w_u    as *const u32 as *const _);
            encoder.set_bytes(7,  4, &cin_u  as *const u32 as *const _);
            encoder.set_bytes(8,  4, &cout_u as *const u32 as *const _);
            encoder.set_bytes(9,  4, &kh_u   as *const u32 as *const _);
            encoder.set_bytes(10, 4, &kw_u   as *const u32 as *const _);
            encoder.set_bytes(11, 4, &sh_u   as *const u32 as *const _);
            encoder.set_bytes(12, 4, &sw_u   as *const u32 as *const _);
            encoder.set_bytes(13, 4, &ph_u   as *const u32 as *const _);
            encoder.set_bytes(14, 4, &pw_u   as *const u32 as *const _);
            encoder.set_bytes(15, 4, &h_out_u as *const u32 as *const _);
            encoder.set_bytes(16, 4, &w_out_u as *const u32 as *const _);

            let max_threads = pipeline.max_total_threads_per_threadgroup() as usize;
            let tg_w = max_threads.min(cout_u as usize).max(1);
            encoder.dispatch_threads(
                MTLSize {
                    width: cout_u as u64,
                    height: w_out_u as u64,
                    depth: (n_u as u64) * (h_out_u as u64),
                },
                MTLSize { width: tg_w as u64, height: 1, depth: 1 },
            );
            encoder.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            trace!(
                N = n_u, H = h_u, W = w_u, C_in = cin_u, C_out = cout_u,
                K_h = kh_u, K_w = kw_u,
                "metal conv2d dispatched"
            );
            finalize_output(
                out_buf,
                out_size,
                vec![n_u as usize, h_out_u as usize, w_out_u as usize, cout_u as usize],
                on_device,
            )
        }
    }

    /// Input-buffer wrapper that lets device-resident and host-input
    /// paths share a single kernel-dispatch implementation. `Owned` is
    /// what we built from a `HostTensor` (uploaded bytes); `Borrowed`
    /// is the existing `metal::Buffer` inside a `MetalDeviceMem` we
    /// reuse without copying.
    pub(super) enum InBuf<'a> {
        Owned(Buffer),
        Borrowed(&'a Buffer),
    }

    impl<'a> InBuf<'a> {
        pub(super) fn as_buf(&self) -> &Buffer {
            match self {
                InBuf::Owned(b) => b,
                InBuf::Borrowed(b) => b,
            }
        }
    }

    /// Prepare a Metal buffer for a `Value` argument. Returns the
    /// buffer (`Owned` if uploaded from `HostTensor`, `Borrowed` if
    /// reused from `DeviceTensor`), the tensor's shape, and `was_device`
    /// (true when the source was a DeviceTensor — used to decide
    /// whether the kernel's output should also be a DeviceTensor).
    pub(super) fn buffer_for_f32_input<'a>(
        device: &Device,
        arg: &'a Value,
        op_label: &str,
        arg_label: &str,
    ) -> Result<(InBuf<'a>, Vec<usize>, usize, bool)> {
        match arg {
            Value::Tensor(t) => {
                if t.dtype != DType::F32 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal {op_label}: {arg_label} f32 only, got {}",
                        t.dtype.name()
                    )));
                }
                let buf = device.new_buffer_with_data(
                    t.data.as_ptr() as *const _,
                    t.data.len() as u64,
                    MTLResourceOptions::StorageModeShared,
                );
                Ok((InBuf::Owned(buf), t.shape.clone(), t.numel(), false))
            }
            Value::DeviceTensor(dt) => {
                if dt.device != DeviceId::Metal {
                    return Err(InterpretError::Unsupported(format!(
                        "metal {op_label}: {arg_label} DeviceTensor on {} not Metal",
                        dt.device.name()
                    )));
                }
                if dt.dtype != DType::F32 {
                    return Err(InterpretError::Unsupported(format!(
                        "metal {op_label}: {arg_label} f32 only, got {}",
                        dt.dtype.name()
                    )));
                }
                let metal_mem = dt
                    .mem
                    .as_any()
                    .downcast_ref::<MetalDeviceMem>()
                    .ok_or_else(|| InterpretError::Unsupported(format!(
                        "metal {op_label}: {arg_label} DeviceTensor not produced by MetalLeaf"
                    )))?;
                Ok((
                    InBuf::Borrowed(&metal_mem.buffer),
                    dt.shape.clone(),
                    dt.numel(),
                    true,
                ))
            }
            other => Err(InterpretError::Unsupported(format!(
                "metal {op_label}: {arg_label} expected tensor, got {other:?}"
            ))),
        }
    }

    /// Wrap a freshly-produced `metal::Buffer` as either a
    /// `Value::Tensor` (read bytes back to host) or `Value::DeviceTensor`
    /// (keep it on the GPU), mirroring the inputs' residency.
    pub(super) fn finalize_output(
        out_buf: Buffer,
        out_size_bytes: usize,
        shape: Vec<usize>,
        keep_on_device: bool,
    ) -> Result<Value> {
        if keep_on_device {
            let mem = MetalDeviceMem {
                buffer: out_buf,
                size_bytes: out_size_bytes,
            };
            Ok(Value::DeviceTensor(DeviceTensor::new(
                DeviceId::Metal,
                DType::F32,
                shape,
                Arc::new(mem),
            )))
        } else {
            let bytes = read_buffer_bytes(&out_buf, out_size_bytes);
            HostTensor::new(shape, DType::F32, bytes)
                .map(Value::Tensor)
                .map_err(map_build_err)
        }
    }

    /// Pick a 2D threadgroup that doesn't exceed `max_threads` total and
    /// fits within the (width, height) dispatch bounds. Conservatively
    /// shape-aware: rectangular grids get rectangular tiles. Returns at
    /// least (1, 1) so degenerate inputs don't divide by zero downstream.
    pub(super) fn pick_2d_threadgroup(max_threads: usize, width: usize, height: usize) -> (usize, usize) {
        // Start with a square-ish baseline; clip to the actual grid.
        let mut tg_w = 16usize;
        let mut tg_h = 16usize;
        tg_w = tg_w.min(width).max(1);
        tg_h = tg_h.min(height).max(1);
        // If we've over-allocated, scale down preferring the wider axis.
        while tg_w * tg_h > max_threads {
            if tg_w >= tg_h && tg_w > 1 { tg_w /= 2; }
            else if tg_h > 1 { tg_h /= 2; }
            else { break; }
        }
        (tg_w.max(1), tg_h.max(1))
    }

    fn register(
        device: &Device,
        cache: &mut HashMap<&'static str, ComputePipelineState>,
        fn_name: &'static str,
        msl: &str,
    ) -> Option<()> {
        let lib = device
            .new_library_with_source(msl, &CompileOptions::new())
            .ok()?;
        let func = lib.get_function(fn_name, None).ok()?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .ok()?;
        cache.insert(fn_name, pipeline);
        Some(())
    }

    /// Extract a `Value`'s declared shape. Returns `None` for non-tensor
    /// values. Used by `dispatch_matmul_epilogue_by_name` to infer `M, K, N`
    /// from a 2D input / weight pair (the graph-rewriter args contract).
    pub(super) fn value_shape(v: &Value) -> Option<Vec<usize>> {
        match v {
            Value::Tensor(t) => Some(t.shape.clone()),
            Value::DeviceTensor(dt) => Some(dt.shape.clone()),
            _ => None,
        }
    }

    /// Extract a `Value`'s data as a borrowed `Vec<f32>`. f16 dispatch
    /// converts to half on upload — the host side stays f32 today (no
    /// HostTensor::F16 yet). Returns `None` for non-f32-tensor values.
    pub(super) fn value_as_f32(v: &Value) -> Option<Vec<f32>> {
        match v {
            Value::Tensor(t) => t.as_f32().ok().map(|s| s.to_vec()),
            _ => None,
        }
    }

    /// Convert an f32 slice to f16 byte representation (2 bytes per
    /// element, little-endian). Used by `dispatch_fused_chain_f16` to
    /// pack host data for upload.
    pub(super) fn f32_slice_to_f16_bytes(src: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(src.len() * 2);
        for &x in src {
            let h = half::f16::from_f32(x);
            out.extend_from_slice(&h.to_le_bytes());
        }
        out
    }

    /// Convert an f16 byte slice (2 bytes per element, little-endian)
    /// back to f32. Used to read the half-precision output buffer.
    pub(super) fn f16_bytes_to_f32_vec(src: &[u8]) -> Vec<f32> {
        src.chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    /// Same conversions for the bf16 variant (Phase L).
    pub(super) fn f32_slice_to_bf16_bytes(src: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(src.len() * 2);
        for &x in src {
            let b = half::bf16::from_f32(x);
            out.extend_from_slice(&b.to_le_bytes());
        }
        out
    }
    pub(super) fn bf16_bytes_to_f32_vec(src: &[u8]) -> Vec<f32> {
        src.chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    /// Phase O fast-path: get a value's data as bytes in the target
    /// narrow dtype. If the input is a `Value::Tensor` already in that
    /// dtype, returns the raw stored bytes (cloned, no math); else
    /// falls back to f32 → narrow conversion. The fast-path is the
    /// whole point of `HostTensor::F16` / `HostTensor::BF16` landing
    /// — host callers that already have narrow data don't pay the
    /// double-convert tax.
    pub(super) fn value_as_narrow_bytes(
        v: &Value,
        dtype: rce_fusion::codegen::Dtype,
        want_numel: usize,
        label: &str,
        op: &str,
    ) -> Result<Vec<u8>> {
        debug_assert!(dtype != rce_fusion::codegen::Dtype::F32);
        // Fast-path: HostTensor matches the target dtype.
        if let Value::Tensor(t) = v {
            let matches = match dtype {
                rce_fusion::codegen::Dtype::F16 => t.dtype == DType::F16,
                rce_fusion::codegen::Dtype::BF16 => t.dtype == DType::BF16,
                rce_fusion::codegen::Dtype::F32 => false,
            };
            if matches {
                if t.numel() != want_numel {
                    return Err(InterpretError::Unsupported(format!(
                        "metal {op}: {label} numel {} != expected {want_numel}",
                        t.numel(),
                    )));
                }
                return Ok(t.data.clone());
            }
        }
        // Slow-path: pull f32, convert to narrow bytes.
        let f = value_as_f32(v).ok_or_else(|| InterpretError::Unsupported(format!(
            "metal {op}: {label} must be f32 or {} HostTensor", dtype.name(),
        )))?;
        if f.len() != want_numel {
            return Err(InterpretError::Unsupported(format!(
                "metal {op}: {label} numel {} != expected {want_numel}", f.len(),
            )));
        }
        Ok(match dtype {
            rce_fusion::codegen::Dtype::F16 => f32_slice_to_f16_bytes(&f),
            rce_fusion::codegen::Dtype::BF16 => f32_slice_to_bf16_bytes(&f),
            rce_fusion::codegen::Dtype::F32 => unreachable!(),
        })
    }

    /// Phase O: wrap narrow output bytes as a HostTensor of the
    /// matching dtype, no f32 conversion. Callers that want f32 can
    /// call `t.as_f32_promoted()` afterwards.
    pub(super) fn narrow_bytes_into_host_tensor(
        bytes: Vec<u8>,
        shape: Vec<usize>,
        dtype: rce_fusion::codegen::Dtype,
    ) -> Result<Value> {
        match dtype {
            rce_fusion::codegen::Dtype::F16 =>
                HostTensor::from_f16_bytes(shape, bytes).map(Value::Tensor),
            rce_fusion::codegen::Dtype::BF16 =>
                HostTensor::from_bf16_bytes(shape, bytes).map(Value::Tensor),
            rce_fusion::codegen::Dtype::F32 => unreachable!(),
        }
    }

    fn arg_tensor<'a>(v: &'a Value, label: &str) -> Result<&'a HostTensor> {
        match v {
            Value::Tensor(t) => Ok(t),
            other => Err(InterpretError::Unsupported(format!(
                "metal: {label} expected tensor, got {other:?}"
            ))),
        }
    }

    fn arg_i32(v: &Value, label: &str) -> Result<i32> {
        match v {
            Value::Scalar(ScalarValue::I32(x)) => Ok(*x),
            Value::Scalar(ScalarValue::I64(x)) => Ok(*x as i32),
            other => Err(InterpretError::Unsupported(format!(
                "metal: {label} expected i32 scalar, got {other:?}"
            ))),
        }
    }

    fn arg_f32(v: &Value, label: &str) -> Result<f32> {
        match v {
            Value::Scalar(ScalarValue::F32(x)) => Ok(*x),
            Value::Scalar(ScalarValue::F64(x)) => Ok(*x as f32),
            other => Err(InterpretError::Unsupported(format!(
                "metal: {label} expected f32 scalar, got {other:?}"
            ))),
        }
    }

    fn require_f32(t: &HostTensor, label: &str) -> Result<()> {
        if t.dtype != DType::F32 {
            return Err(InterpretError::Unsupported(format!(
                "metal: {label} f32 only, got {}",
                t.dtype.name()
            )));
        }
        Ok(())
    }

    fn read_buffer_bytes(buf: &metal::Buffer, len_bytes: usize) -> Vec<u8> {
        let ptr = buf.contents() as *const u8;
        unsafe { std::slice::from_raw_parts(ptr, len_bytes).to_vec() }
    }

    fn missing(name: &str) -> InterpretError {
        InterpretError::Unsupported(format!("metal: pipeline `{name}` not registered"))
    }

    fn map_build_err(e: InterpretError) -> InterpretError {
        InterpretError::Unsupported(format!("metal: build out tensor: {e:?}"))
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(all(test, rce_metal))]
mod tests {
    use super::*;
    use rce_op_nn_prims::{RmsNormLeaf, SiluLeaf};
    use rce_vm_backend_cpu::{DType, HostTensor, InterpretError, ScalarValue};

    fn metal_or_skip() -> Option<MetalLeaf> {
        let leaf = MetalLeaf::new();
        if leaf.is_none() {
            eprintln!("metal unavailable; skipping");
        }
        leaf
    }

    fn t1(v: &[f32]) -> Value {
        Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap())
    }

    // ── Phase B of the Free-chain codegen ──────────────────────────────
    // `dispatch_fused_chain` parity vs the hand-coded paths it generalises,
    // plus pipeline-cache absorption proof.

    #[test]
    fn fused_chain_silu_matches_hand_silu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..1024).map(|i| (i as f32 / 50.0) - 10.0).collect();
        let input = t1(&xs);

        let want = match metal_leaf.execute(0, "silu", std::slice::from_ref(&input)).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Single-element chain. ngroups=1 → group_size=total; single-pass.
        let got = match metal_leaf.dispatch_fused_chain(
            &["silu"], &input, &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        assert_eq!(want.len(), got.len());
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "fused [silu] vs hand silu: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn fused_chain_silu_gelu_matches_sequential_silu_then_gelu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..2048).map(|i| (i as f32 / 60.0).sin() * 1.5 - 0.4).collect();
        let input = t1(&xs);

        let silu_out = match metal_leaf.execute(0, "silu", std::slice::from_ref(&input)).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let want = match metal_leaf.execute(0, "gelu", &[t1(&silu_out)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_fused_chain(
            &["silu", "gelu"], &input, &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5,
            "fused [silu, gelu] vs sequential: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn fused_chain_silu_rms_norm_matches_hand_silu_rms_norm() {
        // The strongest parity check: the codegen-emitted kernel must
        // match the hand-written `silu_rms_norm_f32_grouped` kernel within
        // the same tolerance the hand kernel claims vs sequential.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };

        let total = 2048usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 50.0).sin() * 0.7 - 0.3).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 1000.0).cos() * 0.1).collect();
        let input = t1(&x);
        let weight = t1(&w);

        // Hand path.
        let want = match metal_leaf.execute(
            0, "silu_rms_norm",
            &[input.clone(), weight.clone(),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Codegen path.
        let got = match metal_leaf.dispatch_fused_chain(
            &["silu", "rms_norm"], &input, &[&weight], &[1e-6], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5,
            "fused [silu, rms_norm] vs hand silu_rms_norm: max_abs_err {max_err:.3e}");
        eprintln!("codegen [silu, rms_norm] vs hand kernel: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn fused_chain_pipeline_cache_absorbs_repeat_dispatches() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..256).map(|i| (i as f32 / 30.0).cos()).collect();
        let input = t1(&xs);

        // Starts empty.
        assert_eq!(metal_leaf.fused_pipeline_count(), 0);

        // First dispatch compiles + caches one pipeline.
        let _ = metal_leaf.dispatch_fused_chain(&["silu"], &input, &[], &[], 1).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), 1);

        // Second dispatch of the SAME chain hits the cache.
        let _ = metal_leaf.dispatch_fused_chain(&["silu"], &input, &[], &[], 1).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), 1);

        // A DIFFERENT chain adds a new entry.
        let _ = metal_leaf.dispatch_fused_chain(&["silu", "gelu"], &input, &[], &[], 1).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), 2);

        // Repeating chain #2 still hits the cache.
        let _ = metal_leaf.dispatch_fused_chain(&["silu", "gelu"], &input, &[], &[], 1).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), 2);
    }

    // ── Phase C: codegen-fused dispatch via execute_named ──────────────
    // The graph rewriter (`rce_fusion::fuse_with_codegen`) emits
    // `Custom("fused_<...>_f32")` nodes. The interpreter looks them up
    // via `OpDispatch::claims` / `OpDispatch::execute`. These tests
    // exercise that lookup path end-to-end through `MetalLeaf::execute`.

    #[test]
    fn metal_leaf_claims_fused_chain_names() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Static handles still work.
        assert!(metal_leaf.claims("silu"));
        assert!(metal_leaf.claims("rms_norm"));
        // Codegen-emitted names from chains where every op has a
        // fragment AND the chain is emit_chain_msl-acceptable.
        assert!(metal_leaf.claims("fused_silu_rms_norm_f32"));
        assert!(metal_leaf.claims("fused_silu_rms_norm_gelu_f32"));
        assert!(metal_leaf.claims("fused_rms_norm_gelu_f32"));
        // Garbage: doesn't parse → not claimed.
        assert!(!metal_leaf.claims("fused_not_a_real_op_f32"));
        assert!(!metal_leaf.claims("not_fused_at_all"));
    }

    #[test]
    fn metal_execute_fused_silu_rms_norm_matches_hand_path() {
        // Dispatch via `MetalLeaf::execute("fused_silu_rms_norm_f32", ...)`
        // — the path the interpreter takes when the fusion pass has
        // rewritten the graph. Args follow `apply_fusion`'s contract:
        // (x, weight, ngroups, eps). Result must match the hand-coded
        // `silu_rms_norm` dispatch within the same tolerance.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };

        let total = 2048usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 50.0).sin() * 0.7 - 0.3).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 1000.0).cos() * 0.1).collect();

        let args = [
            t1(&x),
            t1(&w),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::F32(1e-6)),
        ];
        let want = match metal_leaf.execute(0, "silu_rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(0, "fused_silu_rms_norm_f32", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5,
            "execute(fused_silu_rms_norm_f32) vs execute(silu_rms_norm): max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_fused_three_node_chain_matches_sequential() {
        // `fused_silu_rms_norm_gelu_f32` — three-node chain with no
        // hand-coded equivalent. Compare against the sequential
        // dispatch silu → rms_norm → gelu, each through `execute`.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };

        let total = 2048usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 50.0).sin() * 0.5 - 0.2).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 800.0).sin() * 0.05).collect();

        // Sequential path: silu(x) → rms_norm(silu_out, w, 1, eps) → gelu(rms_out).
        let silu_out = match metal_leaf.execute(0, "silu", &[t1(&x)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let rms_out = match metal_leaf.execute(
            0, "rms_norm",
            &[t1(&silu_out), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let want = match metal_leaf.execute(0, "gelu", &[t1(&rms_out)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };

        // Fused path through execute_named.
        let got = match metal_leaf.execute(
            0, "fused_silu_rms_norm_gelu_f32",
            &[t1(&x), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "execute(fused_silu_rms_norm_gelu_f32) vs sequential: max_abs_err {max_err:.3e}");
        eprintln!("3-node fused chain via execute: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_execute_fused_unknown_chain_errors_gracefully() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Looks like a fused name but the body doesn't parse — should
        // surface as Unsupported, not panic.
        let err = metal_leaf.execute(0, "fused_zzz_f32", &[t1(&[1.0])]).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("doesn't parse") || msg.contains("emit failed"),
                    "expected parse / emit error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ── Phase D: auto-generated per-chain parity ───────────────────────
    // Enumerate every chain shape the codegen can emit (length 2 or 3,
    // exactly one reduction, all ops in the fragment table), build a
    // synthetic activation, and assert the fused dispatch matches the
    // sequential equivalent. Single test, deterministic shapes, no
    // hand-maintained per-chain test methods.
    //
    // When a new fragment lands in `rce_fusion::codegen::fragments`,
    // updating `ELEMENTWISE_OPS` / `REDUCTION_OPS` below extends this
    // sweep to every chain shape involving it — no per-fragment test
    // boilerplate.

    /// Run the chain sequentially through `MetalLeaf::execute`, feeding
    /// each op's output into the next op's first arg. For reduction ops
    /// we know the args contract (x, weight, ngroups, eps); for the
    /// elementwise ops we know (x). Same args contract as the codegen
    /// path expects on the fused side, so the fixture stays symmetric.
    fn run_sequential(
        metal_leaf: &MetalLeaf,
        chain: &[&str],
        x: &[f32],
        weight: &[f32],
        ngroups: i32,
        eps: f32,
    ) -> Vec<f32> {
        let mut current: Vec<f32> = x.to_vec();
        for op in chain {
            let result = match *op {
                "silu" | "gelu" => metal_leaf
                    .execute(0, op, &[t1(&current)])
                    .unwrap_or_else(|e| panic!("sequential `{op}`: {e:?}")),
                "rms_norm" => metal_leaf
                    .execute(0, "rms_norm", &[
                        t1(&current),
                        t1(weight),
                        Value::Scalar(ScalarValue::I32(ngroups)),
                        Value::Scalar(ScalarValue::F32(eps)),
                    ])
                    .unwrap_or_else(|e| panic!("sequential rms_norm: {e:?}")),
                other => panic!("run_sequential: no per-op recipe for `{other}`"),
            };
            current = match result {
                Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
                other => panic!("sequential `{op}` returned non-tensor: {other:?}"),
            };
        }
        current
    }

    fn enumerate_phase_c_chains() -> Vec<Vec<&'static str>> {
        // Phase C scope: chains of length 2 or 3, ALL ops in the codegen
        // fragment table, EXACTLY one reduction-bearing op. Today's
        // fragments: silu, gelu, rms_norm (one reduction).
        const ELEMENTWISE: &[&str] = &["silu", "gelu"];
        const REDUCTIONS:  &[&str] = &["rms_norm"];
        let mut out = Vec::new();
        // Length 2: every (elementwise, reduction) and (reduction, elementwise)
        // pairing.
        for r in REDUCTIONS {
            for e in ELEMENTWISE {
                out.push(vec![*e, *r]);
                out.push(vec![*r, *e]);
            }
        }
        // Length 3: reduction at each position; the other two slots are
        // independently picked from ELEMENTWISE.
        for r in REDUCTIONS {
            for e1 in ELEMENTWISE {
                for e2 in ELEMENTWISE {
                    out.push(vec![*r, *e1, *e2]); // reduction at 0
                    out.push(vec![*e1, *r, *e2]); // reduction at 1
                    out.push(vec![*e1, *e2, *r]); // reduction at 2
                }
            }
        }
        out
    }

    #[test]
    fn metal_fused_dispatch_matches_sequential_for_every_codegen_chain() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };

        let total = 1024usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 47.0).sin() * 0.6 - 0.15).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 700.0).cos() * 0.08).collect();
        let ngroups: i32 = 1;
        let eps: f32 = 1e-6;

        let chains = enumerate_phase_c_chains();
        assert!(!chains.is_empty(), "fragment table empty?");

        let mut max_err_seen: f32 = 0.0;
        let mut worst_chain: Option<Vec<&'static str>> = None;
        let mut covered = 0usize;
        for chain in &chains {
            // Skip ones the emitter explicitly rejects (defence in depth —
            // the enumerator shouldn't produce them, but if a future
            // fragment changes shape rules this surfaces the mismatch).
            if rce_fusion::codegen::emit_chain_msl(chain).is_err() {
                continue;
            }
            covered += 1;
            let canonical = rce_fusion::codegen::canonical_function_name(chain);
            let got = match metal_leaf.execute(
                0, &canonical,
                &[t1(&x), t1(&w),
                  Value::Scalar(ScalarValue::I32(ngroups)),
                  Value::Scalar(ScalarValue::F32(eps))],
            ).unwrap_or_else(|e| panic!("fused dispatch `{canonical}`: {e:?}")) {
                Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
                other => panic!("fused `{canonical}` returned non-tensor: {other:?}"),
            };
            let want = run_sequential(&metal_leaf, chain, &x, &w, ngroups, eps);
            assert_eq!(got.len(), want.len(),
                "chain `{canonical}`: length mismatch (got {} vs want {})", got.len(), want.len());
            let max_err = want.iter().zip(got.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 1e-4,
                "chain `{canonical}` ({chain:?}) max_abs_err {max_err:.3e} ≥ 1e-4 \
                 — codegen no longer matches sequential");
            if max_err > max_err_seen {
                max_err_seen = max_err;
                worst_chain = Some(chain.clone());
            }
        }
        // Don't let the enumerator drift to zero silently.
        assert!(covered >= 16,
            "Phase D should sweep ≥ 16 chains for the current fragment table, swept {covered}");
        eprintln!(
            "Phase D swept {covered} chains; worst max_abs_err = {:.3e} on chain {:?}",
            max_err_seen, worst_chain,
        );
    }

    // ── Phase E.a: swiglu fragment, end-to-end ──────────────────────────
    // swiglu is the first 2-input fragment: takes gate (the chain input
    // / running register) and reads `up` from a uniform buffer per
    // element. These tests prove the codegen handles the new layout end
    // to end through `dispatch_fused_chain` + `execute_named`.

    #[test]
    fn metal_dispatch_fused_swiglu_alone_matches_hand_swiglu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 1024usize;
        let gate: Vec<f32> = (0..total).map(|i| (i as f32 / 40.0).sin() * 0.7 - 0.1).collect();
        let up:   Vec<f32> = (0..total).map(|i| 0.5 + (i as f32 / 250.0).cos() * 0.2).collect();

        let want = match metal_leaf.execute(0, "swiglu", &[t1(&gate), t1(&up)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Chain `[swiglu]`: input = gate, uniforms = [up], scalars = [],
        // ngroups = 1.
        let got = match metal_leaf.dispatch_fused_chain(
            &["swiglu"], &t1(&gate), &[&t1(&up)], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5,
            "dispatch_fused_chain([swiglu]) vs hand swiglu: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_swiglu_chain_via_codegen_path() {
        // Full `execute_named` round-trip: name → parse → emit → cache → dispatch.
        // Chain `[rms_norm, swiglu]`: input passes through rms_norm's
        // 2-pass template, then swiglu reads `up` per element post-scale.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 1024usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 40.0).sin() * 0.5).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 600.0).cos() * 0.05).collect();
        let up: Vec<f32> = (0..total).map(|i| 0.6 + (i as f32 / 320.0).sin() * 0.1).collect();

        // Sequential: rms_norm(x, w, 1, eps) → swiglu(rms_out, up).
        let rms_out = match metal_leaf.execute(
            0, "rms_norm",
            &[t1(&x), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let want = match metal_leaf.execute(
            0, "swiglu", &[t1(&rms_out), t1(&up)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };

        // Codegen via execute_named. Args order: input, weight, up,
        // ngroups, eps. Uniforms are in chain order (rms_norm's weight
        // before swiglu's up); scalars are in chain order (rms_norm's
        // eps; swiglu has none).
        let got = match metal_leaf.execute(
            0, "fused_rms_norm_swiglu_f32",
            &[t1(&x), t1(&w), t1(&up),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "execute(fused_rms_norm_swiglu_f32) vs sequential: max_abs_err {max_err:.3e}");
        eprintln!("rms_norm→swiglu via codegen: max_abs_err = {max_err:.3e}");
    }

    // ── Phase F: matmul-epilogue codegen end-to-end ────────────────────
    // The dispatched fused kernel must produce the same result as
    // doing matmul on CPU, then running the epilogue chain through
    // sequential MetalLeaf::execute calls.

    /// Naive matmul reference: `out[i, j] = sum_k input[i, k] * weight[k, j]`.
    /// f32 only, row-major, deterministic. Used by every Phase F parity test.
    fn cpu_matmul(input: &[f32], weight: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += input[i * k + kk] * weight[kk * n + j];
                }
                out[i * n + j] = acc;
            }
        }
        out
    }

    #[test]
    fn metal_matmul_epilogue_silu_matches_cpu_matmul_then_silu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 24usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.3).collect();

        // Reference: CPU matmul → MetalLeaf silu.
        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };

        // Fused dispatch: matmul + silu in one kernel.
        let got = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight),
            m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        assert_eq!(want.len(), got.len(), "{} vs {}", want.len(), got.len());
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "fused matmul+silu vs CPU matmul+silu: max_abs_err {max_err:.3e}");
        eprintln!("matmul[{m}x{k}]@weight[{k}x{n}]+silu fused: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_epilogue_silu_gelu_matches_sequential() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 64usize; let n = 32usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 25.0).sin() * 0.6).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 33.0).cos() * 0.25).collect();

        // Reference: CPU matmul → silu → gelu.
        let mm = cpu_matmul(&input, &weight, m, k, n);
        let after_silu = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let want = match metal_leaf.execute(0, "gelu", &[t1(&after_silu)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Fused.
        let got = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight),
            m as u32, k as u32, n as u32,
            &["silu", "gelu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "matmul+silu+gelu fused vs sequential: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_epilogue_swiglu_with_up_uniform_matches_sequential() {
        // The interesting one: swiglu's `up` is a per-output uniform
        // of shape [M, N]. Tests that the 2D indexing rewrite
        // (`base + i` → `i * N + j`) actually lines up between the
        // codegen emitter and the dispatch layout.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 12usize; let k = 48usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 19.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 27.0).cos() * 0.2).collect();
        let up:     Vec<f32> = (0..m*n).map(|i| 0.5 + (i as f32 / 13.0).sin() * 0.3).collect();

        // Reference: CPU matmul → swiglu(matmul_out, up).
        let gate = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "swiglu", &[t1(&gate), t1(&up)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Fused.
        let got = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight),
            m as u32, k as u32, n as u32,
            &["swiglu"], &[&t1(&up)], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "matmul+swiglu fused vs sequential: max_abs_err {max_err:.3e}");
        eprintln!("matmul+swiglu fused (FFN-style): max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_matmul_epilogue_round_trips() {
        // Full name-string path: `execute("fused_matmul_silu_f32", args)`.
        // Args contract: (input, weight, M, K, N) for 0-uniform-0-scalar chain.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 16usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.3).collect();

        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul_silu_f32",
            &[t1(&input), t1(&weight),
              Value::Scalar(ScalarValue::I32(m as i32)),
              Value::Scalar(ScalarValue::I32(k as i32)),
              Value::Scalar(ScalarValue::I32(n as i32))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "execute(fused_matmul_silu_f32) via name parse: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_epilogue_claims_and_routes_correctly() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // matmul-epilogue names are claimed.
        assert!(metal_leaf.claims("fused_matmul_f32"));
        assert!(metal_leaf.claims("fused_matmul_silu_f32"));
        assert!(metal_leaf.claims("fused_matmul_silu_gelu_f32"));
        assert!(metal_leaf.claims("fused_matmul_swiglu_f32"));
        // Per-element chain names with `matmul` in them aren't valid (the
        // per-element parser doesn't list matmul as a fragment).
        assert!(!metal_leaf.claims("fused_matmul_rms_norm_f32"));
        // Garbage in either family stays unclaimed.
        assert!(!metal_leaf.claims("fused_matmul_zzz_f32"));
        assert!(!metal_leaf.claims("not_fused"));
    }

    #[test]
    fn metal_matmul_epilogue_cache_absorbs_repeats() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.fused_pipeline_count();
        let input:  Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let weight: Vec<f32> = (0..32).map(|i| i as f32 * 0.02).collect();
        // First call compiles + caches.
        let _ = metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), 4, 8, 4, &["silu"], &[], &[],
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 1);
        // Second call hits the cache.
        let _ = metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), 4, 8, 4, &["silu"], &[], &[],
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 1);
        // Different epilogue → new entry.
        let _ = metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), 4, 8, 4, &["gelu"], &[], &[],
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 2);
    }

    // ── Phase G: matmul-epilogue dispatched via shape inference ─────────
    // The graph-rewriter path passes 2D tensors and lets the dispatcher
    // derive M, K, N from shapes. These tests pin that path.

    /// Build a 2D tensor with the given shape `[rows, cols]` from a flat
    /// data slice. Used by the shape-inference tests below to feed
    /// `dispatch_matmul_epilogue_by_name` without explicit dim scalars.
    fn t2(rows: usize, cols: usize, data: &[f32]) -> Value {
        assert_eq!(data.len(), rows * cols, "t2 shape mismatch: {rows}x{cols} != {}", data.len());
        Value::Tensor(HostTensor::from_f32(vec![rows, cols], data).unwrap())
    }

    #[test]
    fn metal_execute_named_matmul_silu_via_shape_inference() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 12usize; let k = 24usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 19.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 27.0).cos() * 0.3).collect();

        // Reference: CPU matmul → silu.
        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Shape-inferred dispatch: 2D input, 2D weight, no M/K/N scalars.
        let got = match metal_leaf.execute(
            0, "fused_matmul_silu_f32",
            &[t2(m, k, &input), t2(k, n, &weight)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "shape-inferred fused_matmul_silu_f32 vs CPU: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_matmul_swiglu_via_shape_inference() {
        // The 2-uniform case: input + weight 2D-shaped, `up` is a 2D
        // tensor of shape [M, N]. Total args = 3 tensors.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 12usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 17.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.2).collect();
        let up:     Vec<f32> = (0..m*n).map(|i| 0.6 + (i as f32 / 13.0).sin() * 0.15).collect();

        let gate = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "swiglu", &[t1(&gate), t1(&up)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul_swiglu_f32",
            // up can be 1D (length M*N) — the kernel reads it as a flat
            // buffer indexed at i*N+j, identical to the 2D case. Mixing
            // 2D matmul args with 1D up is fine; only input/weight need
            // 2D for the shape inference.
            &[t2(m, k, &input), t2(k, n, &weight), t1(&up)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "shape-inferred fused_matmul_swiglu_f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_matmul_epilogue_rejects_1d_in_inferred_path() {
        // Mixing the inferred form's arg count with 1D input/weight
        // surfaces as Unsupported rather than panicking.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let err = metal_leaf.execute(
            0, "fused_matmul_silu_f32",
            // Args count = 2 → enters shape-inference path. But t1 is 1D.
            &[t1(&[1.0, 2.0]), t1(&[1.0, 2.0])],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("2D"),
                    "expected shape error mentioning 2D, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ── Phase H: double-matmul codegen end-to-end ──────────────────────
    // Each thread computes one output `(i, j)` element by walking K2
    // and, for each l, accumulating `(A @ B)[i, l] * C[l, j]`. The
    // intermediate never materializes. These tests confirm the
    // dispatched kernel matches a 2-kernel sequential matmul + silu
    // within tolerance.

    #[test]
    fn metal_matmul_matmul_silu_matches_sequential_double_matmul_then_silu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 6usize; let k1 = 12usize; let k2 = 18usize; let n = 8usize;
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 19.0).cos() * 0.3).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        // Reference: CPU (A @ B), CPU (intermediate @ C), Metal silu.
        let ab = cpu_matmul(&a, &b, m, k1, k2);
        let abc = cpu_matmul(&ab, &c, m, k2, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&abc)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };

        // Fused single dispatch.
        let got = match metal_leaf.dispatch_matmul_matmul(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        assert_eq!(want.len(), got.len());
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-3,
            "fused A@B@C → silu vs sequential: max_abs_err {max_err:.3e}");
        eprintln!("matmul2+silu fused [{m}x{k1}]@[{k1}x{k2}]@[{k2}x{n}]: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_matmul_empty_epilogue_is_plain_triple_product() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k1 = 8usize; let k2 = 6usize; let n = 5usize;
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 17.0).cos() * 0.3).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 19.0).sin() * 0.2).collect();
        let ab = cpu_matmul(&a, &b, m, k1, k2);
        let want = cpu_matmul(&ab, &c, m, k2, n);
        let got = match metal_leaf.dispatch_matmul_matmul(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "plain matmul2 vs CPU: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_matmul_swiglu_matches_sequential() {
        // matmul2 → swiglu. swiglu's `up` is [M, N], indexed at i*N+j —
        // proves the 2D rewrite (`base + i` → `i * N + j`) is also
        // correct in the Phase H template.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k1 = 8usize; let k2 = 6usize; let n = 5usize;
        let a:  Vec<f32> = (0..m*k1).map(|i| (i as f32 / 9.0).sin() * 0.4).collect();
        let b:  Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 13.0).cos() * 0.3).collect();
        let c:  Vec<f32> = (0..k2*n).map(|i| (i as f32 / 17.0).sin() * 0.2).collect();
        let up: Vec<f32> = (0..m*n).map(|i| 0.6 + (i as f32 / 7.0).sin() * 0.15).collect();

        let ab = cpu_matmul(&a, &b, m, k1, k2);
        let gate = cpu_matmul(&ab, &c, m, k2, n);
        let want = match metal_leaf.execute(0, "swiglu", &[t1(&gate), t1(&up)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_matmul(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["swiglu"], &[&t1(&up)], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "matmul2+swiglu vs sequential: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_matmul2_via_shape_inference() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k1 = 8usize; let k2 = 6usize; let n = 5usize;
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 17.0).cos() * 0.3).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 19.0).sin() * 0.2).collect();
        let ab = cpu_matmul(&a, &b, m, k1, k2);
        let abc = cpu_matmul(&ab, &c, m, k2, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&abc)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul2_silu_f32",
            &[t2(m, k1, &a), t2(k1, k2, &b), t2(k2, n, &c)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "execute(fused_matmul2_silu_f32) shape-inferred: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_matmul_claims_correctly() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        assert!(metal_leaf.claims("fused_matmul2_f32"));
        assert!(metal_leaf.claims("fused_matmul2_silu_f32"));
        assert!(metal_leaf.claims("fused_matmul2_swiglu_f32"));
        // Family disambiguation: Phase F's matmul-epilogue parser must
        // reject Phase H names, and vice versa. Each family routes to
        // its own dispatch path via `execute_named`'s prefix match.
        assert!(!rce_fusion::codegen_matmul_epilogue::parse_function_name("fused_matmul2_silu_f32").is_some());
        assert!(!rce_fusion::codegen_matmul_matmul::parse_function_name("fused_matmul_silu_f32").is_some());
        // Garbage stays unclaimed.
        assert!(!metal_leaf.claims("fused_matmul2_not_a_real_op_f32"));
    }

    // ── Phase I: tiled matmul kernel end-to-end ─────────────────────────
    // The tiled kernel uses threadgroup-shared `A_tile` + `B_tile`
    // populated by every thread in the threadgroup before each K-block
    // multiply-add. These tests confirm parity with naive matmul on
    // both tile-aligned (M, N divisible by 16) and non-aligned shapes
    // (where the boundary threadgroups load partial tiles).

    #[test]
    fn metal_matmul_tiled_silu_tile_aligned_matches_cpu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Tile-aligned: 16, 32 are multiples of TILE_M = TILE_K = TILE_N = 16.
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 19.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.3).collect();

        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight),
            m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "tile-aligned tiled matmul+silu vs CPU: max_abs_err {max_err:.3e}");
        eprintln!("tile-aligned [{m}x{k}]@[{k}x{n}]+silu: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_tiled_silu_non_tile_aligned_matches_cpu() {
        // None of M=13, K=27, N=11 divides by 16 — exercises the
        // boundary-threadgroup path where some tile slots load 0 and
        // some output writes are guarded out.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 13usize; let k = 27usize; let n = 11usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).cos() * 0.25).collect();

        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight),
            m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        assert_eq!(got.len(), m * n);
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "non-aligned tiled matmul+silu vs CPU: max_abs_err {max_err:.3e}");
        eprintln!("non-aligned [{m}x{k}]@[{k}x{n}]+silu: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_tiled_swiglu_with_up_uniform_matches_sequential() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 32usize; let k = 48usize; let n = 24usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 21.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 29.0).cos() * 0.2).collect();
        let up:     Vec<f32> = (0..m*n).map(|i| 0.55 + (i as f32 / 13.0).sin() * 0.18).collect();

        let gate = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "swiglu", &[t1(&gate), t1(&up)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight),
            m as u32, k as u32, n as u32,
            &["swiglu"], &[&t1(&up)], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "tiled matmul+swiglu: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_matmul_tiled_via_shape_inference() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 24usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).cos() * 0.3).collect();
        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul_tiled_silu_f32",
            &[t2(m, k, &input), t2(k, n, &weight)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "execute(fused_matmul_tiled_silu_f32) shape-inferred: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_tiled_claims_and_disambiguates() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        assert!(metal_leaf.claims("fused_matmul_tiled_f32"));
        assert!(metal_leaf.claims("fused_matmul_tiled_silu_f32"));
        assert!(metal_leaf.claims("fused_matmul_tiled_swiglu_f32"));
        // All three matmul families coexist + disambiguate at the
        // parser level. The execute_named prefix order (tiled before
        // bare matmul; matmul2 before bare matmul) routes each to its
        // own dispatch path.
        assert!(metal_leaf.claims("fused_matmul_silu_f32"));   // Phase F
        assert!(metal_leaf.claims("fused_matmul2_silu_f32"));  // Phase H
        assert!(!metal_leaf.claims("fused_matmul_tiled_not_a_real_op_f32"));
    }

    // ── Phase J: shape-based matmul auto-routing ───────────────────────
    // `route_matmul` is a pure heuristic; these tests pin its decisions
    // at known boundary shapes and confirm `dispatch_matmul_auto`
    // produces the same arithmetic as both underlying templates.

    #[test]
    fn route_matmul_picks_naive_for_tiny_output() {
        // M*N = 4*4 = 16 ≤ TILE_M*TILE_N = 256 → Naive.
        assert_eq!(MetalLeaf::route_matmul(4, 32, 4), MatmulRouting::Naive);
        // M*N = 16*16 = 256 exactly at the boundary → still Naive (`<=`).
        assert_eq!(MetalLeaf::route_matmul(16, 32, 16), MatmulRouting::Naive);
    }

    #[test]
    fn route_matmul_picks_naive_for_thin_matrices() {
        // M is < TILE_M/2 = 8 → Naive, regardless of N.
        assert_eq!(MetalLeaf::route_matmul(4, 32, 128), MatmulRouting::Naive);
        // Symmetric: N is < TILE_N/2 = 8.
        assert_eq!(MetalLeaf::route_matmul(128, 32, 4), MatmulRouting::Naive);
    }

    #[test]
    fn route_matmul_picks_tiled_for_medium_and_larger() {
        // M=32, N=32 (= 2 tiles each way) → Tiled.
        assert_eq!(MetalLeaf::route_matmul(32, 32, 32), MatmulRouting::Tiled);
        // FFN-bottleneck-style shape.
        assert_eq!(MetalLeaf::route_matmul(64, 512, 2048), MatmulRouting::Tiled);
    }

    #[test]
    fn dispatch_matmul_auto_matches_both_underlying_templates() {
        // Two shapes — one routes naive, one routes tiled — and the
        // auto-dispatched output matches the corresponding low-level
        // dispatch within tolerance. Confirms the router actually
        // routes (no silent fallthrough).
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };

        // Naive-routing shape.
        {
            let m = 6usize; let k = 24usize; let n = 6usize;
            assert_eq!(MetalLeaf::route_matmul(m as u32, k as u32, n as u32), MatmulRouting::Naive);
            let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
            let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).cos() * 0.3).collect();
            let want = match metal_leaf.dispatch_matmul_epilogue(
                &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
                &["silu"], &[], &[],
            ).unwrap() {
                Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
                _ => panic!(),
            };
            let got = match metal_leaf.dispatch_matmul_auto(
                &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
                &["silu"], &[], &[],
            ).unwrap() {
                Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
                _ => panic!(),
            };
            let max_err = want.iter().zip(got.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 1e-5, "naive-routed auto vs naive: max_abs_err {max_err:.3e}");
        }

        // Tiled-routing shape.
        {
            let m = 32usize; let k = 48usize; let n = 32usize;
            assert_eq!(MetalLeaf::route_matmul(m as u32, k as u32, n as u32), MatmulRouting::Tiled);
            let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
            let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.25).collect();
            let want = match metal_leaf.dispatch_matmul_tiled(
                &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
                &["silu"], &[], &[],
            ).unwrap() {
                Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
                _ => panic!(),
            };
            let got = match metal_leaf.dispatch_matmul_auto(
                &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
                &["silu"], &[], &[],
            ).unwrap() {
                Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
                _ => panic!(),
            };
            let max_err = want.iter().zip(got.iter())
                .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_err < 1e-5, "tiled-routed auto vs tiled: max_abs_err {max_err:.3e}");
        }
    }

    #[test]
    fn execute_named_fused_matmul_routes_through_auto() {
        // Before Phase J: `execute("fused_matmul_silu_f32", ...)` always
        // ran the naive kernel. After Phase J: it routes via
        // `dispatch_matmul_auto`. For a tile-sized output (32x32), the
        // router picks Tiled — and the result still matches CPU
        // reference within the same tolerance.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 32usize; let k = 48usize; let n = 32usize;
        assert_eq!(MetalLeaf::route_matmul(m as u32, k as u32, n as u32), MatmulRouting::Tiled);

        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 14.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.25).collect();
        let mm = cpu_matmul(&input, &weight, m, k, n);
        let want = match metal_leaf.execute(0, "silu", &[t1(&mm)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul_silu_f32",
            &[t2(m, k, &input), t2(k, n, &weight)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "execute(fused_matmul_silu_f32) auto-routed to tiled: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn auto_routing_picks_different_pipelines_per_route() {
        // After dispatching at a naive-routed shape and then a
        // tiled-routed shape via `dispatch_matmul_auto`, the
        // `fused_pipelines` cache holds entries for BOTH the naive AND
        // the tiled function names — proving the router actually used
        // both templates.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.fused_pipeline_count();

        // Naive shape.
        let input_n:  Vec<f32> = (0..4*8).map(|i| i as f32 * 0.01).collect();
        let weight_n: Vec<f32> = (0..8*4).map(|i| i as f32 * 0.02).collect();
        let _ = metal_leaf.dispatch_matmul_auto(
            &t1(&input_n), &t1(&weight_n), 4, 8, 4, &["silu"], &[], &[],
        ).unwrap();

        // Tiled shape.
        let input_t:  Vec<f32> = (0..32*32).map(|i| i as f32 * 0.001).collect();
        let weight_t: Vec<f32> = (0..32*32).map(|i| i as f32 * 0.002).collect();
        let _ = metal_leaf.dispatch_matmul_auto(
            &t1(&input_t), &t1(&weight_t), 32, 32, 32, &["silu"], &[], &[],
        ).unwrap();

        // Two distinct pipelines should have been compiled (one per
        // function name family).
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 2,
            "auto-routing should have compiled two distinct kernels (one naive, one tiled)");
    }

    // ── Phase K: f16 fragment variants end-to-end ──────────────────────
    // f16 kernel reads / writes half-precision buffers; per-element
    // math runs in f32 registers. Host tensors stay f32 (no f16
    // HostTensor upstream yet); conversions happen on upload + download.
    // Tolerance widens vs f32 to account for the half-precision storage.

    #[test]
    fn metal_dispatch_fused_chain_f16_matches_f32_within_half_tolerance() {
        // The chain `silu → rms_norm` in f16 vs the same chain in f32.
        // f16 narrowing introduces ~1e-3 error per element (mantissa is
        // 10 bits + implicit 1); both kernels run the same f32 math
        // internally and only diverge on the half store / load.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 1024usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 50.0).sin() * 0.7 - 0.3).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 1000.0).cos() * 0.1).collect();

        let want = match metal_leaf.dispatch_fused_chain(
            &["silu", "rms_norm"], &t1(&x), &[&t1(&w)], &[1e-6], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_fused_chain_f16(
            &["silu", "rms_norm"], &t1(&x), &[&t1(&w)], &[1e-6], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        assert_eq!(got.len(), want.len());
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // f16 storage has ~3 decimal digits of precision; rms_norm
        // output magnitude is ~1.0 so absolute error around 1e-3.
        assert!(max_err < 5e-3,
            "f16 vs f32 fused_silu_rms_norm: max_abs_err {max_err:.3e} ≥ 5e-3");
        eprintln!("f16 silu→rms_norm vs f32: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_fused_chain_f16_silu_alone_matches_f32() {
        // Pure elementwise chain — no reduction — so the f32 register
        // semantics are even tighter. f16 still loses precision on
        // store; tolerance reflects that.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..256).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();
        let want_f32 = match metal_leaf.dispatch_fused_chain(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got_f16 = match metal_leaf.dispatch_fused_chain_f16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap() {
            // Phase O: output is a narrow HostTensor; widen via promotion.
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        let max_err = want_f32.iter().zip(got_f16.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // silu output magnitude ≤ |x| so error roughly proportional to
        // input scale (0.5 here) × half-precision relative epsilon.
        assert!(max_err < 1e-3,
            "f16 silu vs f32 silu: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_routes_f16_via_dtype() {
        // The full execute_named path: `fused_silu_rms_norm_f16`
        // parses out dtype = F16 and routes to the f16 dispatcher.
        // Result matches the f32 reference within the f16 tolerance.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 512usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 40.0).sin() * 0.6).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 800.0).cos() * 0.08).collect();

        let want = match metal_leaf.execute(
            0, "fused_silu_rms_norm_f32",
            &[t1(&x), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_silu_rms_norm_f16",
            &[t1(&x), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 5e-3,
            "execute(fused_silu_rms_norm_f16) vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_claims_both_f32_and_f16_chain_names() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Both dtypes claimed by the per-element parser.
        assert!(metal_leaf.claims("fused_silu_f32"));
        assert!(metal_leaf.claims("fused_silu_f16"));
        assert!(metal_leaf.claims("fused_silu_rms_norm_f16"));
        assert!(metal_leaf.claims("fused_swiglu_f16"));
        // Garbage dtype rejected.
        assert!(!metal_leaf.claims("fused_silu_f64"));
        // bf16 lands in Phase L; this test was authored under the
        // Phase K-only assumption that only f32 + f16 were claimed.
        // After Phase L, bf16 IS claimed too — see
        // `metal_claims_bf16_chain_names` below for the positive
        // assertions.
        assert!(metal_leaf.claims("fused_silu_bf16"));
    }

    #[test]
    fn metal_f16_and_f32_chains_use_separate_pipeline_cache_entries() {
        // Same chain at different dtypes ⇒ distinct function names ⇒
        // distinct pipelines. The cache holds both after dispatching one
        // of each.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.fused_pipeline_count();
        let xs: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let _ = metal_leaf.dispatch_fused_chain(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        let _ = metal_leaf.dispatch_fused_chain_f16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 2,
            "f32 and f16 of the same chain should occupy distinct cache slots");
    }

    // ── Phase L: bf16 dispatch end-to-end ───────────────────────────────
    // bf16 has the same exponent range as f32 but only 7 mantissa bits
    // (vs 10 for f16). For activation-like values that have a wide
    // dynamic range but moderate per-step precision needs (the common
    // transformer regime), bf16 storage is more robust than f16 even
    // though absolute error per element is slightly larger.

    #[test]
    fn metal_dispatch_fused_chain_bf16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 1024usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 50.0).sin() * 0.7 - 0.3).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 1000.0).cos() * 0.1).collect();

        let want = match metal_leaf.dispatch_fused_chain(
            &["silu", "rms_norm"], &t1(&x), &[&t1(&w)], &[1e-6], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_fused_chain_bf16(
            &["silu", "rms_norm"], &t1(&x), &[&t1(&w)], &[1e-6], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // bf16 has ~7 mantissa bits ⇒ relative epsilon ~7.8e-3. For
        // rms-normed output ~1.0 magnitude, absolute err around 1e-2.
        assert!(max_err < 2e-2,
            "bf16 silu→rms_norm vs f32: max_abs_err {max_err:.3e}");
        eprintln!("bf16 silu→rms_norm vs f32: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_routes_bf16_via_dtype() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 512usize;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 40.0).sin() * 0.6).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 800.0).cos() * 0.08).collect();

        let want = match metal_leaf.execute(
            0, "fused_silu_rms_norm_f32",
            &[t1(&x), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_silu_rms_norm_bf16",
            &[t1(&x), t1(&w),
              Value::Scalar(ScalarValue::I32(1)),
              Value::Scalar(ScalarValue::F32(1e-6))],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 2e-2,
            "execute(fused_silu_rms_norm_bf16) vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_claims_bf16_chain_names() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        assert!(metal_leaf.claims("fused_silu_bf16"));
        assert!(metal_leaf.claims("fused_silu_rms_norm_bf16"));
        assert!(metal_leaf.claims("fused_swiglu_bf16"));
        // Three dtypes coexist for the same chain.
        assert!(metal_leaf.claims("fused_silu_f32"));
        assert!(metal_leaf.claims("fused_silu_f16"));
        assert!(metal_leaf.claims("fused_silu_bf16"));
        // Garbage rejected.
        assert!(!metal_leaf.claims("fused_silu_int8"));
    }

    #[test]
    fn metal_three_dtypes_cache_three_distinct_pipelines() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.fused_pipeline_count();
        let xs: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let _ = metal_leaf.dispatch_fused_chain(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        let _ = metal_leaf.dispatch_fused_chain_f16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        let _ = metal_leaf.dispatch_fused_chain_bf16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 3,
            "same chain at three dtypes should compile three pipelines");
    }

    // ── Phase M: matmul-epilogue f16 + bf16 end-to-end ──────────────────

    #[test]
    fn metal_dispatch_matmul_epilogue_f16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.3).collect();

        let want = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_epilogue_f16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // matmul accumulator (K=32) introduces additional rounding;
        // f16 storage on top widens tolerance.
        assert!(max_err < 5e-3,
            "matmul-epilogue f16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_bf16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.3).collect();

        let want = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_epilogue_bf16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 3e-2,
            "matmul-epilogue bf16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_routes_matmul_f16_via_dtype() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 12usize; let k = 24usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.25).collect();

        let want = match metal_leaf.execute(
            0, "fused_matmul_silu_f32",
            &[t2(m, k, &input), t2(k, n, &weight)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul_silu_f16",
            &[t2(m, k, &input), t2(k, n, &weight)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 5e-3,
            "execute(fused_matmul_silu_f16) vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_matmul_epilogue_claims_all_three_dtypes() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        assert!(metal_leaf.claims("fused_matmul_silu_f32"));
        assert!(metal_leaf.claims("fused_matmul_silu_f16"));
        assert!(metal_leaf.claims("fused_matmul_silu_bf16"));
        assert!(metal_leaf.claims("fused_matmul_swiglu_f16"));
        assert!(metal_leaf.claims("fused_matmul_swiglu_bf16"));
    }

    // ── Phase N: matmul² (Phase H) + tiled matmul (Phase I) × f16/bf16 ───

    #[test]
    fn metal_dispatch_matmul_matmul_f16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 6usize; let k1 = 12usize; let k2 = 18usize; let n = 8usize;
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 19.0).cos() * 0.3).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();
        let want = match metal_leaf.dispatch_matmul_matmul(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_matmul_f16(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Two matmul accumulators in sequence give wider tolerance than
        // single matmul; ~5e-3 is realistic.
        assert!(max_err < 1e-2,
            "matmul² f16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_matmul_matmul_bf16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 6usize; let k1 = 12usize; let k2 = 18usize; let n = 8usize;
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 19.0).cos() * 0.3).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();
        let want = match metal_leaf.dispatch_matmul_matmul(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_matmul_bf16(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 5e-2, "matmul² bf16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_matmul_tiled_f16_tile_aligned_matches_f32() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 19.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.3).collect();
        let want = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_tiled_f16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 5e-3, "tiled f16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_matmul_tiled_bf16_non_aligned_matches_f32() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 13usize; let k = 27usize; let n = 11usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).cos() * 0.25).collect();
        let want = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_matmul_tiled_bf16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 3e-2, "tiled bf16 non-aligned vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_execute_named_routes_matmul2_and_tiled_by_dtype() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        assert!(metal_leaf.claims("fused_matmul2_silu_f16"));
        assert!(metal_leaf.claims("fused_matmul2_silu_bf16"));
        assert!(metal_leaf.claims("fused_matmul_tiled_silu_f16"));
        assert!(metal_leaf.claims("fused_matmul_tiled_silu_bf16"));

        let m = 4usize; let k1 = 8usize; let k2 = 6usize; let n = 5usize;
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 17.0).cos() * 0.3).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 19.0).sin() * 0.2).collect();
        let want = match metal_leaf.execute(
            0, "fused_matmul2_silu_f32",
            &[t2(m, k1, &a), t2(k1, k2, &b), t2(k2, n, &c)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_matmul2_silu_f16",
            &[t2(m, k1, &a), t2(k1, k2, &b), t2(k2, n, &c)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-2,
            "execute(fused_matmul2_silu_f16) vs f32: max_abs_err {max_err:.3e}");
    }

    // ── Phase O: HostTensor::F16 / BF16 fast-path ────────────────────────
    // The narrow dispatchers used to ALWAYS pull f32 from HostTensor
    // and convert to narrow bytes on upload. Phase O adds the
    // fast-path: if HostTensor is already in the target dtype, the
    // raw bytes are uploaded directly. Output stays as a narrow
    // HostTensor — callers that want f32 widen via as_f32_promoted.

    fn t1_f16(v: &[f32]) -> Value {
        Value::Tensor(HostTensor::from_f32_as_f16(vec![v.len()], v).unwrap())
    }
    fn t1_bf16(v: &[f32]) -> Value {
        Value::Tensor(HostTensor::from_f32_as_bf16(vec![v.len()], v).unwrap())
    }

    #[test]
    fn metal_dispatch_fused_chain_f16_accepts_native_f16_host_tensor() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 256usize;
        let xs: Vec<f32> = (0..total).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();

        // f32 input path (slow path: converts on upload).
        let from_f32 = match metal_leaf.dispatch_fused_chain_f16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        // f16 input path (fast path: raw bytes uploaded directly).
        let from_f16 = match metal_leaf.dispatch_fused_chain_f16(
            &["silu"], &t1_f16(&xs), &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        // Both should match exactly — same MSL, same input bytes after
        // either-direction f32↔f16 round-trip. Same on-GPU work means
        // same result.
        let max_err = from_f32.iter().zip(from_f16.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-7,
            "fast-path (f16 HostTensor) and slow-path (f32 HostTensor) must match: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_fused_chain_bf16_accepts_native_bf16_host_tensor() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 256usize;
        let xs: Vec<f32> = (0..total).map(|i| (i as f32 / 17.0).sin() * 0.5).collect();
        let from_f32 = match metal_leaf.dispatch_fused_chain_bf16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        let from_bf16 = match metal_leaf.dispatch_fused_chain_bf16(
            &["silu"], &t1_bf16(&xs), &[], &[], 1,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32_promoted().unwrap(),
            _ => panic!(),
        };
        let max_err = from_f32.iter().zip(from_bf16.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-7,
            "bf16 fast-path vs slow-path: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_fused_chain_f16_output_is_native_f16_host_tensor() {
        // The output side: result should be a `Value::Tensor` whose
        // `dtype` is `DType::F16` — no f32 widening on the way out.
        // The bytes ARE the GPU's f16 output, copied through.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let out = metal_leaf.dispatch_fused_chain_f16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, DType::F16,
                    "Phase O: f16 dispatch must return a Tensor with DType::F16, got {}", t.dtype.name());
                assert_eq!(t.data.len(), xs.len() * 2,
                    "Phase O: f16 output byte count = 2 × numel ({} × 2)", xs.len());
                // The raw f16 byte view is exposed for downstream
                // callers that want to skip the f32 widening.
                let bytes = t.as_f16_bytes().expect("DType::F16 must expose f16 bytes");
                assert_eq!(bytes.len(), xs.len() * 2);
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn metal_dispatch_fused_chain_bf16_output_is_native_bf16_host_tensor() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..32).map(|i| i as f32 * 0.01).collect();
        let out = metal_leaf.dispatch_fused_chain_bf16(
            &["silu"], &t1(&xs), &[], &[], 1,
        ).unwrap();
        match out {
            Value::Tensor(t) => {
                assert_eq!(t.dtype, DType::BF16);
                let bytes = t.as_bf16_bytes().expect("DType::BF16 must expose bf16 bytes");
                assert_eq!(bytes.len(), xs.len() * 2);
            }
            other => panic!("expected Tensor, got {other:?}"),
        }
    }

    #[test]
    fn host_tensor_f16_round_trips_through_promotion() {
        // The basic Phase O constructors / accessors round-trip
        // f32 → f16 storage → f32 within half-precision tolerance.
        let xs: Vec<f32> = (0..256).map(|i| (i as f32 / 25.0).sin() * 0.7).collect();
        let t = HostTensor::from_f32_as_f16(vec![xs.len()], &xs).unwrap();
        assert_eq!(t.dtype, DType::F16);
        assert_eq!(t.data.len(), xs.len() * 2);
        // Raw byte accessor.
        assert!(t.as_f16_bytes().is_some());
        assert!(t.as_bf16_bytes().is_none());
        // Widen via promotion.
        let widened = t.as_f32_promoted().unwrap();
        let max_err = xs.iter().zip(widened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // f16 has ~3 decimal digits; |xs| < 1 so absolute err < 1e-3.
        assert!(max_err < 1e-3,
            "f16 storage round-trip via as_f32_promoted: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn host_tensor_bf16_round_trips_through_promotion() {
        let xs: Vec<f32> = (0..256).map(|i| (i as f32 / 25.0).sin() * 0.7).collect();
        let t = HostTensor::from_f32_as_bf16(vec![xs.len()], &xs).unwrap();
        assert_eq!(t.dtype, DType::BF16);
        assert!(t.as_bf16_bytes().is_some());
        let widened = t.as_f32_promoted().unwrap();
        let max_err = xs.iter().zip(widened.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // bf16 has ~2 decimal digits of mantissa; absolute err < ~1e-2.
        assert!(max_err < 1e-2,
            "bf16 storage round-trip: max_abs_err {max_err:.3e}");
    }

    // ── Phase Q: FFN-swiglu dual matmul kernel ──────────────────────────
    // The kernel computes `output = silu(input @ gate_W) * (input @ up_W)`
    // in one Metal dispatch. These tests confirm the dispatched result
    // matches a CPU reference within tolerance and that the pipeline
    // cache picks it up under the canonical name.

    fn cpu_ffn_swiglu(
        input: &[f32], gate_w: &[f32], up_w: &[f32],
        m: usize, k: usize, n: usize,
    ) -> Vec<f32> {
        // Reference: two matmuls then silu*mul.
        let mut g = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k { acc += input[i*k + kk] * gate_w[kk*n + j]; }
            g[i*n + j] = acc;
        }}
        let mut u = vec![0.0f32; m * n];
        for i in 0..m { for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k { acc += input[i*k + kk] * up_w[kk*n + j]; }
            u[i*n + j] = acc;
        }}
        g.iter().zip(u.iter()).map(|(gg, uu)| {
            let s = gg / (1.0 + (-gg).exp());
            s * uu
        }).collect()
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_matches_cpu_reference() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let want = cpu_ffn_swiglu(&input, &gate_w, &up_w, m, k, n);
        let got = match metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "ffn-swiglu Metal vs CPU: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_ffn_swiglu_pipeline_cached_under_canonical_name() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.fused_pipeline_count();
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input:  Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| i as f32 * 0.02).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| i as f32 * 0.03).collect();

        let _ = metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap();
        // First call compiles + caches.
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 1);
        // Repeat hits the cache.
        let _ = metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 1);
    }

    #[test]
    fn metal_claims_ffn_swiglu_canonical_name() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Phase X: all three dtypes now claimed; execute_named routes
        // them through dispatch_ffn_swiglu_by_name.
        assert!(metal_leaf.claims("fused_ffn_swiglu_f32"));
        assert!(metal_leaf.claims("fused_ffn_swiglu_f16"));
        assert!(metal_leaf.claims("fused_ffn_swiglu_bf16"));
        // Wrong family / made-up dtype still rejected.
        assert!(!metal_leaf.claims("fused_ffn_silu_f32"));
        assert!(!metal_leaf.claims("fused_ffn_swiglu_f64"));
    }

    // ── Phase R: tiled FFN-swiglu kernel ──────────────────────────────

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_matches_cpu_reference() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let want = cpu_ffn_swiglu(&input, &gate_w, &up_w, m, k, n);
        let got = match metal_leaf.dispatch_ffn_swiglu_tiled(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "ffn-swiglu-tiled Metal vs CPU: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_non_aligned_shape_works() {
        // Boundary threadgroups (M=13, K=27, N=11 — none divisible by
        // TILE_*=16) load some zeros into the bounds-violating tile
        // slots; the kernel still produces correct output.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 13usize; let k = 27usize; let n = 11usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).cos() * 0.25).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).sin() * 0.25).collect();
        let want = cpu_ffn_swiglu(&input, &gate_w, &up_w, m, k, n);
        let got = match metal_leaf.dispatch_ffn_swiglu_tiled(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        assert_eq!(got.len(), m * n);
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "ffn-swiglu-tiled non-aligned vs CPU: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_claims_ffn_swiglu_tiled_canonical_name() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Phase X: tiled f32 / f16 / bf16 all claimed.
        assert!(metal_leaf.claims("fused_ffn_swiglu_tiled_f32"));
        assert!(metal_leaf.claims("fused_ffn_swiglu_tiled_f16"));
        assert!(metal_leaf.claims("fused_ffn_swiglu_tiled_bf16"));
        // Family disambiguation: Phase Q's name still works.
        assert!(metal_leaf.claims("fused_ffn_swiglu_f32"));
        // Garbage rejected.
        assert!(!metal_leaf.claims("fused_ffn_swiglu_tiled_f64"));
    }

    // ── Phase X: execute_named OpDispatch routing for FFN-swiglu ──────
    // Every canonical name (3 dtypes × naive/tiled) must round-trip
    // through `execute(...)` and match the direct dispatch_* path.
    // The inferred-shape arg convention (input + gate_W + up_W, M/K/N
    // pulled from tensor shapes) is the one graph rewriters emit.

    fn ffn_swiglu_test_inputs(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();
        (input, gate_w, up_w)
    }

    #[test]
    fn metal_execute_named_routes_ffn_swiglu_f32_via_inferred_shape() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k, n) = (8, 32, 16);
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let want = match metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w), m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_ffn_swiglu_f32",
            &[t2(m, k, &input), t2(k, n, &gate_w), t2(k, n, &up_w)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(want, got, "execute_named f32 must match direct dispatch_ffn_swiglu");
    }

    #[test]
    fn metal_execute_named_routes_ffn_swiglu_f16_via_inferred_shape() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k, n) = (8, 32, 16);
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let want = match metal_leaf.dispatch_ffn_swiglu_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w), m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_ffn_swiglu_f16",
            &[t2(m, k, &input), t2(k, n, &gate_w), t2(k, n, &up_w)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(want, got, "execute_named f16 must match direct dispatch_ffn_swiglu_f16");
    }

    #[test]
    fn metal_execute_named_routes_ffn_swiglu_bf16_via_inferred_shape() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k, n) = (8, 32, 16);
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let want = match metal_leaf.dispatch_ffn_swiglu_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w), m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_ffn_swiglu_bf16",
            &[t2(m, k, &input), t2(k, n, &gate_w), t2(k, n, &up_w)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(want, got, "execute_named bf16 must match direct dispatch_ffn_swiglu_bf16");
    }

    #[test]
    fn metal_execute_named_routes_ffn_swiglu_tiled_f32_via_inferred_shape() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k, n) = (16, 32, 16);
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let want = match metal_leaf.dispatch_ffn_swiglu_tiled(
            &t1(&input), &t1(&gate_w), &t1(&up_w), m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_ffn_swiglu_tiled_f32",
            &[t2(m, k, &input), t2(k, n, &gate_w), t2(k, n, &up_w)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(want, got, "execute_named tiled f32 must match direct dispatch_ffn_swiglu_tiled");
    }

    #[test]
    fn metal_execute_named_routes_ffn_swiglu_tiled_f16_via_inferred_shape() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k, n) = (16, 32, 16);
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let want = match metal_leaf.dispatch_ffn_swiglu_tiled_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w), m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_ffn_swiglu_tiled_f16",
            &[t2(m, k, &input), t2(k, n, &gate_w), t2(k, n, &up_w)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(want, got);
    }

    #[test]
    fn metal_execute_named_routes_ffn_swiglu_tiled_bf16_via_inferred_shape() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k, n) = (16, 32, 16);
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let want = match metal_leaf.dispatch_ffn_swiglu_tiled_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w), m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "fused_ffn_swiglu_tiled_bf16",
            &[t2(m, k, &input), t2(k, n, &gate_w), t2(k, n, &up_w)],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(want, got);
    }

    #[test]
    fn metal_execute_named_ffn_swiglu_rejects_wrong_arg_count() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let one = vec![0.0f32; 1];
        // 2 args is neither 3 (inferred) nor 6 (explicit).
        let err = metal_leaf.execute(
            0, "fused_ffn_swiglu_f32", &[t1(&one), t1(&one)],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("expected 3 args") || msg.contains("expected 6 args")
                        || msg.contains("got 2"),
                    "expected arg-count error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn metal_execute_named_ffn_swiglu_rejects_shape_mismatch() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // input [4, 8], gate_W [16, 8] — K mismatches between input(K=8)
        // and gate_W(K=16).
        let input  = vec![0.0f32; 4*8];
        let gate_w = vec![0.0f32; 16*8];
        let up_w   = vec![0.0f32; 16*8];
        let err = metal_leaf.execute(
            0, "fused_ffn_swiglu_f32",
            &[t2(4, 8, &input), t2(16, 8, &gate_w), t2(16, 8, &up_w)],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("shape mismatch") || msg.contains("M,K") || msg.contains("K,N"),
                    "expected shape-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ── Phase S: weight upload cache ──────────────────────────────────

    #[test]
    fn metal_weight_cache_grows_then_releases() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.cached_weight_count();
        let data1: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let data2: Vec<f32> = (0..128).map(|i| i as f32 * 0.02).collect();
        let h1 = metal_leaf.cache_weight_f32(&data1);
        let h2 = metal_leaf.cache_weight_f32(&data2);
        assert_ne!(h1, h2, "distinct uploads must return distinct handles");
        assert_eq!(metal_leaf.cached_weight_count(), baseline + 2);
        assert!(metal_leaf.release_weight(h1));
        assert_eq!(metal_leaf.cached_weight_count(), baseline + 1);
        // Releasing the same handle twice returns false the second time.
        assert!(!metal_leaf.release_weight(h1));
        assert!(metal_leaf.release_weight(h2));
        assert_eq!(metal_leaf.cached_weight_count(), baseline);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_matches_uncached() {
        // The cached dispatch must produce the same output as
        // `dispatch_ffn_swiglu_tiled` — same kernel, just a different
        // upload path for the weights.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let uncached = match metal_leaf.dispatch_ffn_swiglu_tiled(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };

        let gh = metal_leaf.cache_weight_f32(&gate_w);
        let uh = metal_leaf.cache_weight_f32(&up_w);
        let cached = match metal_leaf.dispatch_ffn_swiglu_tiled_cached(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        // Bit-identical: same kernel, same inputs (just bytes from
        // different buffer sources).
        assert_eq!(uncached, cached,
            "cached vs uncached dispatch must produce identical output");
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_rejects_bad_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01).collect();
        let err = metal_leaf.dispatch_ffn_swiglu_tiled_cached(
            &t1(&input), 0xDEADBEEF, 0xCAFEBABE,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("not in cache"),
                    "expected cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_rejects_wrong_size_weight() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01).collect();
        // Cache a weight of the wrong size.
        let bad_gate: Vec<f32> = vec![0.0f32; 16]; // expected K*N = 32
        let good_up: Vec<f32> = vec![0.0f32; k*n];
        let gh = metal_leaf.cache_weight_f32(&bad_gate);
        let uh = metal_leaf.cache_weight_f32(&good_up);
        let err = metal_leaf.dispatch_ffn_swiglu_tiled_cached(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("numel") && msg.contains("K*N"),
                    "expected numel mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    // ── Phase Y: cached + narrow dtype FFN-swiglu ─────────────────────
    // cache_weight_{f16,bf16} returns a tagged handle; the
    // _cached_{f16,bf16} dispatch variants validate the tag and run
    // the narrow tiled kernel with cached weight buffers.

    #[test]
    fn metal_cache_weight_handles_distinct_per_dtype() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.cached_weight_count();
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let h32 = metal_leaf.cache_weight_f32(&data);
        let h16 = metal_leaf.cache_weight_f16(&data);
        let hbf = metal_leaf.cache_weight_bf16(&data);
        assert_ne!(h32, h16);
        assert_ne!(h16, hbf);
        assert_ne!(h32, hbf);
        assert_eq!(metal_leaf.cached_weight_count(), baseline + 3);
        assert!(metal_leaf.release_weight(h32));
        assert!(metal_leaf.release_weight(h16));
        assert!(metal_leaf.release_weight(hbf));
        assert_eq!(metal_leaf.cached_weight_count(), baseline);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_f16_matches_uncached_f16() {
        // Bit-identical to dispatch_ffn_swiglu_tiled_f16 — same kernel,
        // weights via cache instead of per-call upload.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);

        let uncached = match metal_leaf.dispatch_ffn_swiglu_tiled_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let gh = metal_leaf.cache_weight_f16(&gate_w);
        let uh = metal_leaf.cache_weight_f16(&up_w);
        let cached = match metal_leaf.dispatch_ffn_swiglu_tiled_cached_f16(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached, "cached f16 must be bit-identical to uncached f16");
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_bf16_matches_uncached_bf16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let uncached = match metal_leaf.dispatch_ffn_swiglu_tiled_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let gh = metal_leaf.cache_weight_bf16(&gate_w);
        let uh = metal_leaf.cache_weight_bf16(&up_w);
        let cached = match metal_leaf.dispatch_ffn_swiglu_tiled_cached_bf16(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached, "cached bf16 must be bit-identical to uncached bf16");
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_f32_rejects_narrow_handle() {
        // Phase Z: the f32 cached dispatcher uses the tagged lookup
        // too — passing an f16 handle must surface a clean dtype
        // error, not a buffer-size mismatch.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let gate_w = vec![0.0f32; k*n];
        let up_w   = vec![0.0f32; k*n];
        let gh_f16 = metal_leaf.cache_weight_f16(&gate_w);
        let uh_f32 = metal_leaf.cache_weight_f32(&up_w);
        let err = metal_leaf.dispatch_ffn_swiglu_tiled_cached(
            &t1(&input), gh_f16, uh_f32,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as f16") && msg.contains("needs f32"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(gh_f16);
        metal_leaf.release_weight(uh_f32);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_f16_rejects_wrong_dtype_handle() {
        // Pass f32-cached handle to the f16 dispatcher — must error
        // with a dtype-tag mismatch, not silently dispatch.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let gate_w = vec![0.0f32; k*n];
        let up_w   = vec![0.0f32; k*n];
        let gh_f32 = metal_leaf.cache_weight_f32(&gate_w);
        let uh_f16 = metal_leaf.cache_weight_f16(&up_w);
        let err = metal_leaf.dispatch_ffn_swiglu_tiled_cached_f16(
            &t1(&input), gh_f32, uh_f16,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as f32") && msg.contains("needs f16"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(gh_f32);
        metal_leaf.release_weight(uh_f16);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_f16_rejects_bad_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input: Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01).collect();
        let err = metal_leaf.dispatch_ffn_swiglu_tiled_cached_f16(
            &t1(&input), 0xDEAD_BEEF, 0xCAFE_BABE,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("not in cache"),
                    "expected cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_cached_f16_rejects_wrong_size() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        // gate sized for K*N=32, but cache up_w sized 16 → mismatch.
        let bad_up: Vec<f32> = vec![0.0f32; 16];
        let good_gate: Vec<f32> = vec![0.0f32; k*n];
        let gh = metal_leaf.cache_weight_f16(&good_gate);
        let uh = metal_leaf.cache_weight_f16(&bad_up);
        let err = metal_leaf.dispatch_ffn_swiglu_tiled_cached_f16(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("numel") && msg.contains("K*N"),
                    "expected numel mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    // ── Phase AA: matmul-epilogue cached dispatch ─────────────────────
    // dispatch_matmul_epilogue_cached(weight_handle) is bit-identical
    // to dispatch_matmul_epilogue(weight_value) for the same weight
    // data — same kernel, weight comes from cache instead of per-call
    // upload. Validates tag (must be Dtype::F32), handle existence,
    // and K*N size.

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_matches_uncached_empty_epilogue() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let uncached = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f32(&weight);
        let cached = match metal_leaf.dispatch_matmul_epilogue_cached(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-epilogue cached (empty epilogue) must be bit-identical to uncached");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_matches_uncached_silu_epilogue() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 11.0).cos() * 0.3).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).sin() * 0.25).collect();
        let uncached = match metal_leaf.dispatch_matmul_epilogue(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f32(&weight);
        let cached = match metal_leaf.dispatch_matmul_epilogue_cached(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-epilogue cached (epilogue=[silu]) must be bit-identical to uncached");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_rejects_narrow_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input  = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let wh_f16 = metal_leaf.cache_weight_f16(&weight);
        let err = metal_leaf.dispatch_matmul_epilogue_cached(
            &t1(&input), wh_f16, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as f16") && msg.contains("needs f32"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(wh_f16);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_rejects_bad_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let err = metal_leaf.dispatch_matmul_epilogue_cached(
            &t1(&input), 0xDEAD_BEEF, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("not in cache"),
                    "expected cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_rejects_wrong_size_weight() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        // Cache a weight sized for K=16 instead of K=8 → numel 64 != K*N=32.
        let bad_weight = vec![0.0f32; 16*n];
        let wh = metal_leaf.cache_weight_f32(&bad_weight);
        let err = metal_leaf.dispatch_matmul_epilogue_cached(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("numel") && msg.contains("K*N"),
                    "expected numel mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(wh);
    }

    // ── Phase BB: matmul-epilogue cached + narrow dtype ───────────────
    // dispatch_matmul_epilogue_cached_{f16,bf16} must be bit-identical
    // to the uncached narrow path (Phase M) for the same data — and
    // the tagged cache must reject cross-dtype handles symmetrically
    // with Phase Y/Z.

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_f16_matches_uncached_f16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let uncached = match metal_leaf.dispatch_matmul_epilogue_f16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f16(&weight);
        let cached = match metal_leaf.dispatch_matmul_epilogue_cached_f16(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-epilogue cached f16 must be bit-identical to uncached f16");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_bf16_matches_uncached_bf16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let uncached = match metal_leaf.dispatch_matmul_epilogue_bf16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_bf16(&weight);
        let cached = match metal_leaf.dispatch_matmul_epilogue_cached_bf16(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-epilogue cached bf16 must be bit-identical to uncached bf16");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_f16_empty_epilogue_matches_uncached() {
        // Plain f16 matmul through the cached path — covers the
        // "no epilogue" code generation branch.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 17.0).sin() * 0.35).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).cos() * 0.25).collect();
        let uncached = match metal_leaf.dispatch_matmul_epilogue_f16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f16(&weight);
        let cached = match metal_leaf.dispatch_matmul_epilogue_cached_f16(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached);
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_f16_rejects_wrong_dtype_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input  = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let wh_bf16 = metal_leaf.cache_weight_bf16(&weight);
        let err = metal_leaf.dispatch_matmul_epilogue_cached_f16(
            &t1(&input), wh_bf16, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as bf16") && msg.contains("needs f16"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(wh_bf16);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_bf16_rejects_wrong_dtype_handle() {
        // Symmetric to the f16 case: cache as f32, dispatch as bf16.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input  = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let wh_f32 = metal_leaf.cache_weight_f32(&weight);
        let err = metal_leaf.dispatch_matmul_epilogue_cached_bf16(
            &t1(&input), wh_f32, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as f32") && msg.contains("needs bf16"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(wh_f32);
    }

    #[test]
    fn metal_dispatch_matmul_epilogue_cached_f16_rejects_bad_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let err = metal_leaf.dispatch_matmul_epilogue_cached_f16(
            &t1(&input), 0xCAFE_F00D, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("not in cache"),
                    "expected cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ── Phase CC: tiled matmul-epilogue cached + narrow dtype ────────
    // Same kernel as Phase I / N (tiled matmul + epilogue) but weight
    // comes from cache_weight_*. Bit-identical to uncached tiled
    // dispatch for each dtype × epilogue combo.

    #[test]
    fn metal_dispatch_matmul_tiled_cached_matches_uncached_silu_epilogue() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Tile-aligned shape — same dims the Phase I non-aligned test
        // uses, large enough to exercise the K-block walk.
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let uncached = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f32(&weight);
        let cached = match metal_leaf.dispatch_matmul_tiled_cached(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-tiled cached must be bit-identical to uncached for the same data");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_tiled_cached_matches_uncached_empty_epilogue() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Non-tile-aligned shape — exercises the boundary-threadgroup
        // bounds-guard path (M=13, K=27, N=11 — none divisible by 16).
        let m = 13usize; let k = 27usize; let n = 11usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 11.0).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 17.0).cos() * 0.25).collect();
        let uncached = match metal_leaf.dispatch_matmul_tiled(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f32(&weight);
        let cached = match metal_leaf.dispatch_matmul_tiled_cached(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached);
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_tiled_cached_f16_matches_uncached_f16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let uncached = match metal_leaf.dispatch_matmul_tiled_f16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_f16(&weight);
        let cached = match metal_leaf.dispatch_matmul_tiled_cached_f16(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-tiled-cached f16 must be bit-identical to uncached f16");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_tiled_cached_bf16_matches_uncached_bf16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let weight: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let uncached = match metal_leaf.dispatch_matmul_tiled_bf16(
            &t1(&input), &t1(&weight), m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let wh = metal_leaf.cache_weight_bf16(&weight);
        let cached = match metal_leaf.dispatch_matmul_tiled_cached_bf16(
            &t1(&input), wh, m as u32, k as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-tiled-cached bf16 must be bit-identical to uncached bf16");
        metal_leaf.release_weight(wh);
    }

    #[test]
    fn metal_dispatch_matmul_tiled_cached_rejects_narrow_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input  = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let wh_f16 = metal_leaf.cache_weight_f16(&weight);
        let err = metal_leaf.dispatch_matmul_tiled_cached(
            &t1(&input), wh_f16, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as f16") && msg.contains("needs f32"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(wh_f16);
    }

    #[test]
    fn metal_dispatch_matmul_tiled_cached_f16_rejects_wrong_dtype_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input  = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let wh_bf16 = metal_leaf.cache_weight_bf16(&weight);
        let err = metal_leaf.dispatch_matmul_tiled_cached_f16(
            &t1(&input), wh_bf16, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("cached as bf16") && msg.contains("needs f16"),
                    "expected dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(wh_bf16);
    }

    #[test]
    fn metal_dispatch_matmul_tiled_cached_rejects_bad_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let err = metal_leaf.dispatch_matmul_tiled_cached(
            &t1(&input), 0xBADCAFE, m as u32, k as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("not in cache"),
                    "expected cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ── Phase DD: double-matmul cached (Phase H + cache) ──────────────
    // Same kernel as dispatch_matmul_matmul / _f16 / _bf16, but B + C
    // come from cache. Tag-validated. A and uniforms still upload
    // per-call.

    fn matmul_matmul_test_inputs(
        m: usize, k1: usize, k2: usize, n: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let a: Vec<f32> = (0..m*k1).map(|i| (i as f32 / 11.0).sin() * 0.4).collect();
        let b: Vec<f32> = (0..k1*k2).map(|i| (i as f32 / 17.0).cos() * 0.25).collect();
        let c: Vec<f32> = (0..k2*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();
        (a, b, c)
    }

    #[test]
    fn metal_dispatch_matmul_matmul_cached_matches_uncached() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k1, k2, n) = (4, 8, 6, 4);
        let (a, b, c) = matmul_matmul_test_inputs(m, k1, k2, n);
        let uncached = match metal_leaf.dispatch_matmul_matmul(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let bh = metal_leaf.cache_weight_f32(&b);
        let ch = metal_leaf.cache_weight_f32(&c);
        let cached = match metal_leaf.dispatch_matmul_matmul_cached(
            &t1(&a), bh, ch,
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "matmul-matmul cached must be bit-identical to uncached for same data");
        metal_leaf.release_weight(bh);
        metal_leaf.release_weight(ch);
    }

    #[test]
    fn metal_dispatch_matmul_matmul_cached_f16_matches_uncached_f16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k1, k2, n) = (4, 8, 6, 4);
        let (a, b, c) = matmul_matmul_test_inputs(m, k1, k2, n);
        let uncached = match metal_leaf.dispatch_matmul_matmul_f16(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let bh = metal_leaf.cache_weight_f16(&b);
        let ch = metal_leaf.cache_weight_f16(&c);
        let cached = match metal_leaf.dispatch_matmul_matmul_cached_f16(
            &t1(&a), bh, ch,
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached);
        metal_leaf.release_weight(bh);
        metal_leaf.release_weight(ch);
    }

    #[test]
    fn metal_dispatch_matmul_matmul_cached_bf16_matches_uncached_bf16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k1, k2, n) = (4, 8, 6, 4);
        let (a, b, c) = matmul_matmul_test_inputs(m, k1, k2, n);
        let uncached = match metal_leaf.dispatch_matmul_matmul_bf16(
            &t1(&a), &t1(&b), &t1(&c),
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let bh = metal_leaf.cache_weight_bf16(&b);
        let ch = metal_leaf.cache_weight_bf16(&c);
        let cached = match metal_leaf.dispatch_matmul_matmul_cached_bf16(
            &t1(&a), bh, ch,
            m as u32, k1 as u32, k2 as u32, n as u32,
            &["silu"], &[], &[],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached);
        metal_leaf.release_weight(bh);
        metal_leaf.release_weight(ch);
    }

    #[test]
    fn metal_dispatch_matmul_matmul_cached_rejects_narrow_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k1, k2, n) = (2, 4, 3, 2);
        let a = vec![0.0f32; m*k1];
        let b = vec![0.0f32; k1*k2];
        let c = vec![0.0f32; k2*n];
        let bh_f16 = metal_leaf.cache_weight_f16(&b);
        let ch_f32 = metal_leaf.cache_weight_f32(&c);
        let err = metal_leaf.dispatch_matmul_matmul_cached(
            &t1(&a), bh_f16, ch_f32,
            m as u32, k1 as u32, k2 as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("B handle") && msg.contains("cached as f16")
                        && msg.contains("needs f32"),
                    "expected B dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(bh_f16);
        metal_leaf.release_weight(ch_f32);
    }

    #[test]
    fn metal_dispatch_matmul_matmul_cached_f16_rejects_c_wrong_dtype() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k1, k2, n) = (2, 4, 3, 2);
        let a = vec![0.0f32; m*k1];
        let b = vec![0.0f32; k1*k2];
        let c = vec![0.0f32; k2*n];
        let bh = metal_leaf.cache_weight_f16(&b);
        let ch_bf16 = metal_leaf.cache_weight_bf16(&c);
        let err = metal_leaf.dispatch_matmul_matmul_cached_f16(
            &t1(&a), bh, ch_bf16,
            m as u32, k1 as u32, k2 as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("C handle") && msg.contains("cached as bf16")
                        && msg.contains("needs f16"),
                    "expected C dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(bh);
        metal_leaf.release_weight(ch_bf16);
    }

    #[test]
    fn metal_dispatch_matmul_matmul_cached_rejects_bad_b_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let (m, k1, k2, n) = (2, 4, 3, 2);
        let a = vec![0.0f32; m*k1];
        let c = vec![0.0f32; k2*n];
        let ch = metal_leaf.cache_weight_f32(&c);
        let err = metal_leaf.dispatch_matmul_matmul_cached(
            &t1(&a), 0xDEAD_F00D, ch,
            m as u32, k1 as u32, k2 as u32, n as u32,
            &[], &[], &[],
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("B handle") && msg.contains("not in cache"),
                    "expected B cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(ch);
    }

    // ── Phase EE: naive FFN-swiglu cached (Phase Q + cache) ───────────
    // Same kernel as Phase Q / Phase V's naive narrow path, but gate
    // and up come from `cache_weight_*` handles. Completes the cached
    // matrix: every shipped chain shape × dtype now has a cached form.

    #[test]
    fn metal_dispatch_ffn_swiglu_cached_matches_uncached() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let uncached = match metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let gh = metal_leaf.cache_weight_f32(&gate_w);
        let uh = metal_leaf.cache_weight_f32(&up_w);
        let cached = match metal_leaf.dispatch_ffn_swiglu_cached(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached,
            "ffn-swiglu (naive) cached must be bit-identical to uncached");
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_cached_f16_matches_uncached_f16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let uncached = match metal_leaf.dispatch_ffn_swiglu_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let gh = metal_leaf.cache_weight_f16(&gate_w);
        let uh = metal_leaf.cache_weight_f16(&up_w);
        let cached = match metal_leaf.dispatch_ffn_swiglu_cached_f16(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached);
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_cached_bf16_matches_uncached_bf16() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let (input, gate_w, up_w) = ffn_swiglu_test_inputs(m, k, n);
        let uncached = match metal_leaf.dispatch_ffn_swiglu_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        let gh = metal_leaf.cache_weight_bf16(&gate_w);
        let uh = metal_leaf.cache_weight_bf16(&up_w);
        let cached = match metal_leaf.dispatch_ffn_swiglu_cached_bf16(
            &t1(&input), gh, uh,
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(), _ => panic!(),
        };
        assert_eq!(uncached, cached);
        metal_leaf.release_weight(gh);
        metal_leaf.release_weight(uh);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_cached_rejects_narrow_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let gh_f16 = metal_leaf.cache_weight_f16(&weight);
        let uh_f32 = metal_leaf.cache_weight_f32(&weight);
        let err = metal_leaf.dispatch_ffn_swiglu_cached(
            &t1(&input), gh_f16, uh_f32,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("gate handle") && msg.contains("cached as f16")
                        && msg.contains("needs f32"),
                    "expected gate dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(gh_f16);
        metal_leaf.release_weight(uh_f32);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_cached_f16_rejects_wrong_dtype_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let weight = vec![0.0f32; k*n];
        let gh_f16 = metal_leaf.cache_weight_f16(&weight);
        let uh_bf16 = metal_leaf.cache_weight_bf16(&weight);
        let err = metal_leaf.dispatch_ffn_swiglu_cached_f16(
            &t1(&input), gh_f16, uh_bf16,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("up handle") && msg.contains("cached as bf16")
                        && msg.contains("needs f16"),
                    "expected up dtype-mismatch error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
        metal_leaf.release_weight(gh_f16);
        metal_leaf.release_weight(uh_bf16);
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_cached_rejects_bad_handle() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input = vec![0.0f32; m*k];
        let err = metal_leaf.dispatch_ffn_swiglu_cached(
            &t1(&input), 0xBAD0_BABE, 0xBAD0_F00D,
            m as u32, k as u32, n as u32,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("gate handle") && msg.contains("not in cache"),
                    "expected cache-miss error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ── Phase V: FFN-swiglu f16 / bf16 dispatch ──────────────────────

    #[test]
    fn metal_dispatch_ffn_swiglu_f16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let want = match metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_ffn_swiglu_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Two K=32 dot products + half storage on three buffers
        // widens tolerance vs single-matmul Phase M (5e-3).
        assert!(max_err < 1e-2,
            "ffn-swiglu f16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_bf16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 8usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let want = match metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_ffn_swiglu_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // bf16 has smaller mantissa than f16 — wider tolerance.
        assert!(max_err < 4e-2,
            "ffn-swiglu bf16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_f16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let want = match metal_leaf.dispatch_ffn_swiglu_tiled(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_ffn_swiglu_tiled_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-2,
            "ffn-swiglu-tiled f16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_tiled_bf16_matches_f32_within_tolerance() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let m = 16usize; let k = 32usize; let n = 16usize;
        let input:  Vec<f32> = (0..m*k).map(|i| (i as f32 / 13.0).sin() * 0.4).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| (i as f32 / 19.0).cos() * 0.2).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| (i as f32 / 23.0).sin() * 0.2).collect();

        let want = match metal_leaf.dispatch_ffn_swiglu_tiled(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.dispatch_ffn_swiglu_tiled_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 4e-2,
            "ffn-swiglu-tiled bf16 vs f32: max_abs_err {max_err:.3e}");
    }

    #[test]
    fn metal_ffn_swiglu_narrow_pipeline_cached_per_dtype() {
        // f32, f16, bf16 must compile to three distinct pipelines —
        // the canonical name carries the dtype.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let baseline = metal_leaf.fused_pipeline_count();
        let m = 4usize; let k = 8usize; let n = 4usize;
        let input:  Vec<f32> = (0..m*k).map(|i| i as f32 * 0.01).collect();
        let gate_w: Vec<f32> = (0..k*n).map(|i| i as f32 * 0.02).collect();
        let up_w:   Vec<f32> = (0..k*n).map(|i| i as f32 * 0.03).collect();

        let _ = metal_leaf.dispatch_ffn_swiglu(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap();
        let _ = metal_leaf.dispatch_ffn_swiglu_f16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap();
        let _ = metal_leaf.dispatch_ffn_swiglu_bf16(
            &t1(&input), &t1(&gate_w), &t1(&up_w),
            m as u32, k as u32, n as u32,
        ).unwrap();
        assert_eq!(metal_leaf.fused_pipeline_count(), baseline + 3,
            "same shape at three dtypes must compile three pipelines");
    }

    #[test]
    fn metal_dispatch_ffn_swiglu_f16_rejects_zero_dims() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let one = vec![0.0f32; 1];
        let err = metal_leaf.dispatch_ffn_swiglu_f16(
            &t1(&one), &t1(&one), &t1(&one), 0, 1, 1,
        ).unwrap_err();
        match err {
            InterpretError::Unsupported(msg) => {
                assert!(msg.contains("must all be positive"),
                    "expected dim-zero error, got: {msg}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn metal_silu_matches_cpu_reference() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu = SiluLeaf::new();
        let xs: Vec<f32> = (0..1024).map(|i| (i as f32 / 50.0) - 10.0).collect();
        let arg = t1(&xs);

        let want = match cpu.execute(0, "silu", std::slice::from_ref(&arg)).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(0, "silu", std::slice::from_ref(&arg)).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal silu max_abs_err {max_err}");
        eprintln!("metal silu: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_rms_norm_ngroups_one_matches_cpu() {
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu = RmsNormLeaf::new();

        let total = 256;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 30.0).sin() * 2.0).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 100.0)).collect();
        let args = vec![
            t1(&x),
            t1(&w),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::F32(1e-6)),
        ];

        let want = match cpu.execute(0, "rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(0, "rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal rms_norm (ngroups=1) max_abs_err {max_err}");
        eprintln!("metal rms_norm ngroups=1: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_rms_norm_grouped_matches_cpu() {
        // Mamba2 group-norm shape: ngroups>1, group_size = d_inner/ngroups.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu = RmsNormLeaf::new();

        let ngroups = 8;
        let group_size = 64;
        let total = ngroups * group_size;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32 / 25.0).cos() - 0.5) * 1.5).collect();
        let w: Vec<f32> = (0..total).map(|i| 0.9 + (i as f32 / 200.0)).collect();
        let args = vec![
            t1(&x),
            t1(&w),
            Value::Scalar(ScalarValue::I32(ngroups as i32)),
            Value::Scalar(ScalarValue::F32(1e-6)),
        ];

        let want = match cpu.execute(0, "rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(0, "rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal rms_norm (grouped) max_abs_err {max_err}");
        eprintln!(
            "metal rms_norm ngroups={ngroups} group_size={group_size}: max_abs_err = {max_err:.3e}"
        );
    }

    #[test]
    fn metal_silu_rms_norm_matches_sequential_silu_then_rms_norm() {
        // The fusion correctness property: dispatching the fused
        // `silu_rms_norm` kernel produces the same output (up to f32
        // reduction order) as separately dispatching `silu` then
        // feeding its output into `rms_norm`. This is exactly what
        // `rce-placement::fusion_eligible(&["silu","rms_norm"])`
        // promises is safe — both ops are in the Free fraction; a
        // fusion pass can rewrite a `silu → rms_norm` chain into a
        // single `silu_rms_norm` Apply node without changing the
        // program's observable output.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };

        let total = 2048;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 50.0).sin() * 0.7 - 0.3).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 1000.0).cos() * 0.1).collect();
        let ngroups = Value::Scalar(ScalarValue::I32(1));
        let eps     = Value::Scalar(ScalarValue::F32(1e-6));

        // Sequential path: silu(x) → rms_norm(silu_out, w, ...).
        let silu_out = match metal_leaf.execute(0, "silu", &[t1(&x)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            other => panic!("silu returned non-tensor: {other:?}"),
        };
        let want = match metal_leaf.execute(
            0, "rms_norm",
            &[t1(&silu_out), t1(&w), ngroups.clone(), eps.clone()],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            other => panic!("rms_norm returned non-tensor: {other:?}"),
        };

        // Fused path: silu_rms_norm(x, w, ngroups, eps).
        let got = match metal_leaf.execute(
            0, "silu_rms_norm",
            &[t1(&x), t1(&w), ngroups, eps],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            other => panic!("silu_rms_norm returned non-tensor: {other:?}"),
        };

        assert_eq!(want.len(), got.len());
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "fused silu_rms_norm max_abs_err {max_err:.3e} (≥1e-4 — fusion semantics broken)");
        eprintln!("metal silu_rms_norm fused-vs-sequential: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_silu_rms_norm_realistic_llm_ffn_shape() {
        // Transformer FFN pre-norm: d_model = 4096. Same fixture style
        // as the rms_norm production-size test, swapped to the fused op.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let total = 4096;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 80.0).cos() * 0.5).collect();
        let w: Vec<f32> = (0..total).map(|i| 0.9 + (i as f32 / 800.0)).collect();
        let ngroups = Value::Scalar(ScalarValue::I32(1));
        let eps     = Value::Scalar(ScalarValue::F32(1e-6));

        let silu_out = match metal_leaf.execute(0, "silu", &[t1(&x)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let want = match metal_leaf.execute(
            0, "rms_norm",
            &[t1(&silu_out), t1(&w), ngroups.clone(), eps.clone()],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(
            0, "silu_rms_norm",
            &[t1(&x), t1(&w), ngroups, eps],
        ).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap().to_vec(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4,
            "fused silu_rms_norm at d_model=4096 max_abs_err {max_err:.3e}");
        eprintln!("metal silu_rms_norm d_model=4096: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_rms_norm_realistic_mamba2_2_7b_shape() {
        // mamba2-2.7b group-norm: d_inner=5120, ngroups=1 by default
        // (head-dim group norm is on a different shape). Just exercise
        // that the kernel handles the real production size.
        let metal_leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu = RmsNormLeaf::new();

        let total = 5120;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 100.0).sin() * 0.7).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 1000.0)).collect();
        let args = vec![
            t1(&x),
            t1(&w),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::F32(1e-6)),
        ];

        let want = match cpu.execute(0, "rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match metal_leaf.execute(0, "rms_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-4, "metal rms_norm (mamba2-2.7b) max_abs_err {max_err}");
        eprintln!("metal rms_norm mamba2-2.7b: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_gelu_matches_inline_reference() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        // gelu(x) with tanh approx, computed in f32 the same way the
        // shader does it. Inline reference — no CPU leaf yet.
        fn gelu_ref(x: f32) -> f32 {
            const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
            const COEFF: f32 = 0.044715;
            let x3 = x * x * x;
            let mut inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            if inner > 10.0 { inner = 10.0; }
            if inner < -10.0 { inner = -10.0; }
            0.5 * x * (1.0 + inner.tanh())
        }

        let xs: Vec<f32> = (0..1024).map(|i| (i as f32 / 50.0) - 10.0).collect();
        let want: Vec<f32> = xs.iter().map(|x| gelu_ref(*x)).collect();
        let arg = t1(&xs);
        let got = match leaf.execute(0, "gelu", std::slice::from_ref(&arg)).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal gelu max_abs_err {max_err}");
        eprintln!("metal gelu: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_swiglu_matches_inline_reference() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        let n = 1024;
        let gate: Vec<f32> = (0..n).map(|i| (i as f32 / 30.0).sin() - 0.5).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 / 25.0).cos() + 0.2).collect();
        let want: Vec<f32> = gate.iter().zip(up.iter()).map(|(g, u)| {
            let silu_g = if *g >= 0.0 {
                *g / (1.0 + (-*g).exp())
            } else {
                let e = g.exp();
                *g * e / (1.0 + e)
            };
            silu_g * *u
        }).collect();

        let args = vec![t1(&gate), t1(&up)];
        let got = match leaf.execute(0, "swiglu", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal swiglu max_abs_err {max_err}");
        eprintln!("metal swiglu: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_layer_norm_ngroups_one_matches_inline_reference() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        let total = 256;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 / 30.0).sin() * 2.0).collect();
        let w: Vec<f32> = (0..total).map(|i| 1.0 + (i as f32 / 100.0)).collect();
        let b: Vec<f32> = (0..total).map(|i| 0.01 * (i as f32) - 1.0).collect();
        let eps = 1e-6f32;

        // Inline reference: standard LayerNorm (mean+variance) with
        // ngroups=1 — the whole vector is one group.
        let mean = x.iter().sum::<f32>() / total as f32;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / total as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let want: Vec<f32> = x.iter().enumerate()
            .map(|(i, v)| (v - mean) * inv_std * w[i] + b[i])
            .collect();

        let args = vec![
            t1(&x), t1(&w), t1(&b),
            Value::Scalar(ScalarValue::I32(1)),
            Value::Scalar(ScalarValue::F32(eps)),
        ];
        let got = match leaf.execute(0, "layer_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal layer_norm max_abs_err {max_err}");
        eprintln!("metal layer_norm ngroups=1: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_layer_norm_grouped_matches_inline_reference() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        let ngroups = 4usize;
        let group_size = 64usize;
        let total = ngroups * group_size;
        let x: Vec<f32> = (0..total).map(|i| ((i as f32 / 25.0).cos() - 0.5) * 1.5).collect();
        let w: Vec<f32> = (0..total).map(|i| 0.9 + (i as f32 / 200.0)).collect();
        let b: Vec<f32> = (0..total).map(|i| -0.1 + (i as f32 / 500.0)).collect();
        let eps = 1e-6f32;

        let mut want = vec![0.0f32; total];
        for g in 0..ngroups {
            let s = g * group_size;
            let e = s + group_size;
            let slice = &x[s..e];
            let mean = slice.iter().sum::<f32>() / group_size as f32;
            let var = slice.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / group_size as f32;
            let inv_std = 1.0 / (var + eps).sqrt();
            for i in 0..group_size {
                want[s + i] = (slice[i] - mean) * inv_std * w[s + i] + b[s + i];
            }
        }

        let args = vec![
            t1(&x), t1(&w), t1(&b),
            Value::Scalar(ScalarValue::I32(ngroups as i32)),
            Value::Scalar(ScalarValue::F32(eps)),
        ];
        let got = match leaf.execute(0, "layer_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal layer_norm grouped max_abs_err {max_err}");
        eprintln!("metal layer_norm ngroups={ngroups} group_size={group_size}: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_attention_decode_mode_matches_cpu() {
        // Decode mode: B=1, attend across S keys/values, head_dim=64.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        use rce_op_attention::AttentionLeaf;
        let cpu = AttentionLeaf::new();

        let head_dim = 64usize;
        let seq_len = 32usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.013 - 0.4).sin()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.005 - 0.2).cos() * 0.3).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i as f32) * 0.007 + 0.1).sin() * 0.5).collect();

        let q_t = Value::Tensor(HostTensor::from_f32(vec![1, head_dim], &q).unwrap());
        let k_t = Value::Tensor(HostTensor::from_f32(vec![seq_len, head_dim], &k).unwrap());
        let v_t = Value::Tensor(HostTensor::from_f32(vec![seq_len, head_dim], &v).unwrap());
        let scale_v = Value::Scalar(ScalarValue::F32(scale));
        let sliding = Value::Scalar(ScalarValue::I32(0));
        let args = vec![q_t, k_t, v_t, scale_v, sliding];

        let want = match cpu.execute(0, "attention", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match leaf.execute(0, "attention", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        assert_eq!(want.len(), got.len());
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal attention max_abs_err {max_err}");
        eprintln!("metal attention decode (head_dim={head_dim} seq_len={seq_len}): max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_attention_with_sliding_window_matches_cpu() {
        // Sliding window of 16 over a 64-position context.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        use rce_op_attention::AttentionLeaf;
        let cpu = AttentionLeaf::new();

        let head_dim = 32usize;
        let seq_len = 64usize;
        let sliding = 16i32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.02 - 0.3).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.011 + 0.05).sin() * 0.4).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.009 - 0.1).cos() * 0.6).collect();

        let q_t = Value::Tensor(HostTensor::from_f32(vec![1, head_dim], &q).unwrap());
        let k_t = Value::Tensor(HostTensor::from_f32(vec![seq_len, head_dim], &k).unwrap());
        let v_t = Value::Tensor(HostTensor::from_f32(vec![seq_len, head_dim], &v).unwrap());
        let args = vec![
            q_t, k_t, v_t,
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(sliding)),
        ];

        let want = match cpu.execute(0, "attention", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match leaf.execute(0, "attention", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal attention sliding-window max_abs_err {max_err}");
        eprintln!("metal attention sliding window={sliding}: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_attention_prefill_batch_gt_one_matches_cpu() {
        // Prefill mode: B>1 (multiple Q rows attending to the same K/V).
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        use rce_op_attention::AttentionLeaf;
        let cpu = AttentionLeaf::new();

        let head_dim = 16usize;
        let batch = 8usize;
        let seq_len = 24usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..batch * head_dim).map(|i| ((i as f32) * 0.017 - 0.5).sin()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.008 + 0.1).cos() * 0.4).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.003 - 0.2).sin() * 0.7).collect();

        let q_t = Value::Tensor(HostTensor::from_f32(vec![batch, head_dim], &q).unwrap());
        let k_t = Value::Tensor(HostTensor::from_f32(vec![seq_len, head_dim], &k).unwrap());
        let v_t = Value::Tensor(HostTensor::from_f32(vec![seq_len, head_dim], &v).unwrap());
        let args = vec![
            q_t, k_t, v_t,
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(0)),
        ];

        let want = match cpu.execute(0, "attention", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match leaf.execute(0, "attention", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal attention prefill max_abs_err {max_err}");
        eprintln!("metal attention prefill (batch={batch} seq={seq_len}): max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_embedding_lookup_matches_cpu() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu = rce_op_nn_prims::EmbeddingLookupLeaf::new();

        // Vocab=128, hidden=64, seq=8.
        let vocab = 128usize;
        let hidden = 64usize;
        let table: Vec<f32> = (0..vocab * hidden)
            .map(|i| ((i as f32) * 0.001 - 0.3).sin())
            .collect();
        let ids = vec![5i32, 0, 127, 42, 99, 13, 200 /* OOV */, 64];

        let table_t = Value::Tensor(HostTensor::from_f32(vec![vocab, hidden], &table).unwrap());
        let ids_t = Value::Tensor(HostTensor::from_i32(vec![ids.len()], &ids).unwrap());
        let args = vec![
            table_t, ids_t,
            Value::Scalar(ScalarValue::I32(hidden as i32)),
        ];

        let want = match cpu.execute(0, "embedding_lookup", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match leaf.execute(0, "embedding_lookup", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        assert_eq!(want, got, "embedding_lookup CPU vs Metal must be byte-identical");
        eprintln!("metal embedding_lookup: bit-perfect over {} elements", got.len());
    }

    #[test]
    fn metal_moe_combine_matches_cpu() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu = rce_op_nn_prims::MoeCombineLeaf::new();

        // 16 experts, d_model=128, top_k=4 — close to nemotron_h_moe shape.
        let num_experts = 16usize;
        let d_model = 128usize;
        let top_k = 4i32;

        let logits: Vec<f32> = (0..num_experts).map(|i| ((i as f32) * 0.13 - 1.0).sin()).collect();
        let experts: Vec<f32> = (0..num_experts * d_model)
            .map(|i| ((i as f32) * 0.0021 + 0.1).cos() * 0.7)
            .collect();

        let args = vec![
            Value::Tensor(HostTensor::from_f32(vec![num_experts], &logits).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![num_experts, d_model], &experts).unwrap()),
            Value::Scalar(ScalarValue::I32(top_k)),
        ];

        let want = match cpu.execute(0, "moe_combine", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let got = match leaf.execute(0, "moe_combine", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal moe_combine max_abs_err {max_err}");
        eprintln!("metal moe_combine top_k={top_k}: max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_device_tensor_chain_silu_then_silu_no_host_roundtrip() {
        // Demonstrates the DeviceTensor flow: HOST → GPU once at the
        // start, two consecutive silu Apply nodes execute on GPU
        // without a host roundtrip between them, then GPU → HOST once
        // at the end. The intermediate is `Value::DeviceTensor`,
        // never copied back to host.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        let xs: Vec<f32> = (0..1024).map(|i| (i as f32 / 50.0) - 10.0).collect();
        let host_in = HostTensor::from_f32(vec![xs.len()], &xs).unwrap();

        // 1. Promote to GPU.
        let dev_in = leaf.promote_to_device(&host_in).expect("promote");
        match &dev_in {
            Value::DeviceTensor(dt) => {
                assert_eq!(dt.device, rce_vm_backend_cpu::DeviceId::Metal);
                assert_eq!(dt.shape, vec![xs.len()]);
            }
            _ => panic!("promote did not produce DeviceTensor"),
        }

        // 2. silu — DeviceTensor in, DeviceTensor out (no host copy).
        let dev_mid = leaf.execute(0, "silu", &[dev_in]).unwrap();
        let mid_dt = match &dev_mid {
            Value::DeviceTensor(dt) => dt.clone(),
            _ => panic!("silu(DeviceTensor) did not return DeviceTensor"),
        };

        // 3. silu again — same chain, still on GPU.
        let dev_out = leaf.execute(0, "silu", std::slice::from_ref(&dev_mid_to_value(&mid_dt))).unwrap();
        let out_dt = match dev_out {
            Value::DeviceTensor(dt) => dt,
            _ => panic!("silu(DeviceTensor) did not return DeviceTensor"),
        };

        // 4. Demote back to host.
        let host_out = leaf.demote_to_host(&out_dt).expect("demote");
        let got = match host_out {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!("demote did not produce HostTensor"),
        };

        // Reference: silu(silu(x)) on host.
        let want: Vec<f32> = xs.iter().map(|&x| {
            let s1 = if x >= 0.0 { x / (1.0 + (-x).exp()) } else { let e = x.exp(); x * e / (1.0 + e) };
            if s1 >= 0.0 { s1 / (1.0 + (-s1).exp()) } else { let e = s1.exp(); s1 * e / (1.0 + e) }
        }).collect();

        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "device-resident silu(silu(x)) max_abs_err {max_err}");
        eprintln!("device-resident silu→silu chain: max_abs_err = {max_err:.3e} over {} elements", got.len());
    }

    /// Reborrow a `DeviceTensor` reference into a `Value` slice. Lets
    /// the round-trip test pass an existing DeviceTensor as a
    /// `&[Value]` arg without consuming it.
    fn dev_mid_to_value(dt: &rce_vm_backend_cpu::DeviceTensor) -> Value {
        Value::DeviceTensor(dt.clone())
    }

    #[test]
    fn metal_device_resident_full_kernel_chain() {
        // Full Metal stack chained device-resident: silu → rms_norm →
        // gelu → swiglu (with `up` on host) → layer_norm. Five Apply
        // nodes back-to-back; HOST→GPU once at the start, GPU→HOST
        // once at the end. Validates that every Metal kernel correctly
        // accepts DeviceTensor input and returns DeviceTensor output.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let cpu_silu = rce_op_nn_prims::SiluLeaf::new();
        let cpu_rms = rce_op_nn_prims::RmsNormLeaf::new();
        let cpu_gelu = rce_op_nn_prims::GeluLeaf::new();
        let cpu_swi = rce_op_nn_prims::SwigluLeaf::new();
        let cpu_ln = rce_op_nn_prims::LayerNormLeaf::new();

        let n = 256usize;
        let xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013) - 0.5).collect();
        let w_rms: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32 * 0.001)).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).sin() + 0.3).collect();
        let w_ln: Vec<f32> = vec![1.0; n];
        let b_ln: Vec<f32> = vec![0.0; n];

        // ── CPU reference path ─────────────────────────────────
        let one = Value::Scalar(ScalarValue::I32(1));
        let eps = Value::Scalar(ScalarValue::F32(1e-6));
        fn t1(v: &[f32]) -> Value {
            Value::Tensor(HostTensor::from_f32(vec![v.len()], v).unwrap())
        }
        let r1 = match cpu_silu.execute(0, "silu", &[t1(&xs)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!()
        };
        let r2 = match cpu_rms.execute(0, "rms_norm", &[t1(&r1), t1(&w_rms), one.clone(), eps.clone()]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!()
        };
        let r3 = match cpu_gelu.execute(0, "gelu", &[t1(&r2)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!()
        };
        let r4 = match cpu_swi.execute(0, "swiglu", &[t1(&r3), t1(&up)]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!()
        };
        let want = match cpu_ln.execute(0, "layer_norm", &[t1(&r4), t1(&w_ln), t1(&b_ln), one.clone(), eps.clone()]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!()
        };

        // ── Metal device-resident path ─────────────────────────
        let host_in = HostTensor::from_f32(vec![n], &xs).unwrap();
        let mut cur = leaf.promote_to_device(&host_in).expect("promote x");
        cur = leaf.execute(0, "silu", std::slice::from_ref(&cur)).unwrap();
        match &cur {
            Value::DeviceTensor(_) => {}
            _ => panic!("silu did not return DeviceTensor"),
        }
        // rms_norm: [x_dev, w_dev (uploaded), 1, eps].
        let w_rms_dev = leaf
            .promote_to_device(&HostTensor::from_f32(vec![n], &w_rms).unwrap())
            .expect("promote w_rms");
        cur = leaf
            .execute(0, "rms_norm", &[cur.clone(), w_rms_dev.clone(), one.clone(), eps.clone()])
            .unwrap();
        match &cur { Value::DeviceTensor(_) => {} _ => panic!("rms_norm did not return DeviceTensor") }
        cur = leaf.execute(0, "gelu", std::slice::from_ref(&cur)).unwrap();
        match &cur { Value::DeviceTensor(_) => {} _ => panic!("gelu did not return DeviceTensor") }
        // swiglu: [gate_dev, up_dev]. Both DeviceTensors → output DeviceTensor.
        let up_dev = leaf
            .promote_to_device(&HostTensor::from_f32(vec![n], &up).unwrap())
            .expect("promote up");
        cur = leaf.execute(0, "swiglu", &[cur.clone(), up_dev.clone()]).unwrap();
        match &cur { Value::DeviceTensor(_) => {} _ => panic!("swiglu did not return DeviceTensor") }
        // layer_norm: [x_dev, w_dev, b_dev, 1, eps].
        let w_ln_dev = leaf
            .promote_to_device(&HostTensor::from_f32(vec![n], &w_ln).unwrap())
            .expect("promote w_ln");
        let b_ln_dev = leaf
            .promote_to_device(&HostTensor::from_f32(vec![n], &b_ln).unwrap())
            .expect("promote b_ln");
        cur = leaf
            .execute(0, "layer_norm", &[cur.clone(), w_ln_dev, b_ln_dev, one, eps])
            .unwrap();
        match &cur {
            Value::DeviceTensor(_) => {}
            _ => panic!("layer_norm did not return DeviceTensor"),
        }
        // Final demote.
        let dt = match cur {
            Value::DeviceTensor(d) => d,
            _ => unreachable!(),
        };
        let host_out = leaf.demote_to_host(&dt).expect("demote");
        let got = match host_out {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };

        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // Five GPU kernels in a row; each may have a ~ULP divergence.
        // Tolerance is 1e-4.
        assert!(max_err < 1e-4, "device-resident 5-kernel chain max_abs_err {max_err}");
        eprintln!(
            "device-resident chain (silu→rms_norm→gelu→swiglu→layer_norm) over {} elements: max_abs_err = {max_err:.3e}",
            n
        );
    }

    #[test]
    fn metal_device_resident_attention_chain() {
        // Full attention flow on device: promote Q/K/V → run attention
        // → demote. Validates that the 3-tensor input op picks up
        // device-resident inputs and returns a DeviceTensor.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        use rce_op_attention::AttentionLeaf;
        let cpu = AttentionLeaf::new();

        let head_dim = 32usize;
        let seq_len = 16usize;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..head_dim).map(|i| (i as f32 * 0.013).sin() - 0.4).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.005).cos() * 0.3).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32 * 0.007).sin() * 0.5).collect();

        let q_h = HostTensor::from_f32(vec![1, head_dim], &q).unwrap();
        let k_h = HostTensor::from_f32(vec![seq_len, head_dim], &k).unwrap();
        let v_h = HostTensor::from_f32(vec![seq_len, head_dim], &v).unwrap();

        let want = match cpu.execute(0, "attention", &[
            Value::Tensor(q_h.clone()),
            Value::Tensor(k_h.clone()),
            Value::Tensor(v_h.clone()),
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(0)),
        ]).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };

        let q_dev = leaf.promote_to_device(&q_h).unwrap();
        let k_dev = leaf.promote_to_device(&k_h).unwrap();
        let v_dev = leaf.promote_to_device(&v_h).unwrap();
        let out_dev = leaf.execute(0, "attention", &[
            q_dev, k_dev, v_dev,
            Value::Scalar(ScalarValue::F32(scale)),
            Value::Scalar(ScalarValue::I32(0)),
        ]).unwrap();
        let out_dt = match out_dev {
            Value::DeviceTensor(d) => d,
            _ => panic!("attention did not return DeviceTensor"),
        };
        let got = match leaf.demote_to_host(&out_dt).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "device-resident attention max_abs_err {max_err}");
        eprintln!("device-resident attention (head_dim={head_dim} seq={seq_len}): max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_group_norm_matches_inline_reference() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        // Mini-SDXL UNet shape: N=1, C=32, G=8, HW=4*4=16.
        let n = 1usize; let g = 8usize; let c = 32usize; let hw = 16usize;
        let total = n * c * hw;
        let x: Vec<f32> = (0..total).map(|i| (i as f32 * 0.013 - 0.5).sin()).collect();
        let gamma: Vec<f32> = (0..c).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let bias: Vec<f32> = (0..c).map(|i| -0.1 + (i as f32) * 0.005).collect();
        let eps = 1e-6f32;

        // Inline reference.
        let c_per_g = c / g;
        let num_elems = c_per_g * hw;
        let mut want = vec![0.0f32; total];
        for nn in 0..n {
            for gg in 0..g {
                let base_c = gg * c_per_g;
                let mut sum: f32 = 0.0; let mut sum_sq: f32 = 0.0;
                for cl in 0..c_per_g {
                    let cc = base_c + cl;
                    for hh in 0..hw {
                        let v = x[nn * c * hw + cc * hw + hh];
                        sum += v; sum_sq += v * v;
                    }
                }
                let mean = sum / num_elems as f32;
                let var = (sum_sq / num_elems as f32) - mean * mean;
                let inv_std = 1.0_f32 / (var.max(0.0) + eps).sqrt();
                for cl in 0..c_per_g {
                    let cc = base_c + cl;
                    for hh in 0..hw {
                        let idx = nn * c * hw + cc * hw + hh;
                        want[idx] = (x[idx] - mean) * inv_std * gamma[cc] + bias[cc];
                    }
                }
            }
        }

        let x_t = Value::Tensor(HostTensor::from_f32(vec![n, c, hw], &x).unwrap());
        let g_t = Value::Tensor(HostTensor::from_f32(vec![c], &gamma).unwrap());
        let b_t = Value::Tensor(HostTensor::from_f32(vec![c], &bias).unwrap());
        let args = vec![
            x_t, g_t, b_t,
            Value::Scalar(ScalarValue::I32(n as i32)),
            Value::Scalar(ScalarValue::I32(g as i32)),
            Value::Scalar(ScalarValue::F32(eps)),
        ];
        let got = match leaf.execute(0, "group_norm", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal group_norm max_abs_err {max_err}");
        eprintln!("metal group_norm (N={n} G={g} C={c} HW={hw}): max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_cross_device_via_host_roundtrip() {
        // Simulates the Metal→CUDA cross-device migration path
        // (without actually running CUDA): start on Metal, download via
        // the public `download_via_host` helper, re-upload to Metal,
        // verify the buffers are bit-identical. The same pattern works
        // for Metal→CUDA (using rce_op_cuda's upload helper) once
        // both leaves are on the same machine.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        let xs: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.011 - 0.5).sin() * 1.7).collect();
        let host_in = HostTensor::from_f32(vec![xs.len()], &xs).unwrap();

        // Step 1: HOST → Metal (allocates an MTLBuffer).
        let dev_v = leaf.promote_to_device(&host_in).expect("promote");
        let dev_a = match dev_v {
            Value::DeviceTensor(d) => d,
            _ => panic!("promote did not produce DeviceTensor"),
        };

        // Step 2: Metal → HOST via cross-device transfer API.
        let host_mid = super::download_via_host(&dev_a).expect("download_via_host");
        let mid_bytes = match host_mid {
            t => t.as_f32().unwrap(),
        };
        // Bit-identical to original input — no math performed.
        assert_eq!(mid_bytes, xs, "round-trip lost data");

        // Step 3: HOST → Metal again, confirm shape/contents match.
        let host_back = HostTensor::from_f32(vec![mid_bytes.len()], &mid_bytes).unwrap();
        let dev_v2 = leaf.promote_to_device(&host_back).expect("re-promote");
        let dev_b = match dev_v2 {
            Value::DeviceTensor(d) => d,
            _ => panic!(),
        };
        // Run silu on both and confirm same output (sanity that the
        // re-uploaded buffer is functionally equivalent).
        let out_a = match leaf.execute(0, "silu", &[Value::DeviceTensor(dev_a)]).unwrap() {
            Value::DeviceTensor(d) => super::download_via_host(&d).unwrap().as_f32().unwrap(),
            _ => panic!(),
        };
        let out_b = match leaf.execute(0, "silu", &[Value::DeviceTensor(dev_b)]).unwrap() {
            Value::DeviceTensor(d) => super::download_via_host(&d).unwrap().as_f32().unwrap(),
            _ => panic!(),
        };
        assert_eq!(out_a, out_b, "silu after host roundtrip diverges");
        eprintln!(
            "metal cross-device roundtrip: {} elements pass through Metal→Host→Metal bit-perfect",
            xs.len()
        );
    }

    #[test]
    fn metal_topk_mask_basic() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let logits = vec![0.1f32, 0.9, 0.3, 0.7, 0.2, 0.5, 0.6, 0.4];
        let args = vec![
            Value::Tensor(HostTensor::from_f32(vec![logits.len()], &logits).unwrap()),
            Value::Scalar(ScalarValue::I32(3)),
        ];
        let got = match leaf.execute(0, "topk_mask", &args).unwrap() {
            Value::Tensor(t) => t.as_f32().unwrap(),
            _ => panic!(),
        };
        // Top-3 indices: [1, 3, 6] (logits 0.9, 0.7, 0.6).
        let want = vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0];
        assert_eq!(got, want);
        eprintln!("metal topk_mask: bit-perfect");
    }

    #[test]
    fn metal_argmax_matches_cpu_scan() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };

        let n = 128usize;
        // Place the maximum at a known index.
        let mut xs: Vec<f32> = (0..n).map(|i| (i as f32 * 0.0011) - 0.3).collect();
        let want_idx: usize = 73;
        xs[want_idx] = 999.0;

        let arg = Value::Tensor(HostTensor::from_f32(vec![n], &xs).unwrap());
        let result = leaf.execute(0, "argmax", std::slice::from_ref(&arg)).unwrap();
        let out = match result {
            Value::Tensor(t) => t.as_i32().unwrap(),
            _ => panic!("expected i32 tensor"),
        };
        assert_eq!(out, vec![want_idx as i32], "metal argmax wrong index");
        eprintln!("metal argmax (n={n}): correct index {want_idx}");
    }

    #[test]
    fn metal_argmax_realistic_vocab_size() {
        // Sampling from a 32K-vocab logits vector — typical LLM shape.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let n = 32_000usize;
        let mut xs: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.00007) - 1.0).sin() * 0.4).collect();
        let want_idx: usize = 17_829;
        xs[want_idx] = 99.0;

        let arg = Value::Tensor(HostTensor::from_f32(vec![n], &xs).unwrap());
        let out = match leaf.execute(0, "argmax", std::slice::from_ref(&arg)).unwrap() {
            Value::Tensor(t) => t.as_i32().unwrap(),
            _ => panic!(),
        };
        assert_eq!(out, vec![want_idx as i32]);
    }

    #[test]
    fn metal_conv2d_3x3_matches_inline_reference() {
        // Tiny conv: N=1, H=4, W=4, C_in=2, C_out=3, K=3x3, stride=1, pad=1.
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let n = 1usize; let h = 4usize; let w = 4usize;
        let c_in = 2usize; let c_out = 3usize;
        let k_h = 3usize; let k_w = 3usize;
        let s_h = 1usize; let s_w = 1usize;
        let p_h = 1usize; let p_w = 1usize;
        let h_out = (h + 2*p_h - k_h) / s_h + 1;
        let w_out = (w + 2*p_w - k_w) / s_w + 1;
        assert_eq!((h_out, w_out), (4, 4));

        let input: Vec<f32> = (0..n*h*w*c_in).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..c_out*k_h*k_w*c_in).map(|i| ((i as f32 * 0.011) + 0.2).cos() * 0.3).collect();
        let bias: Vec<f32> = (0..c_out).map(|i| (i as f32) * 0.05 - 0.1).collect();

        // Inline reference.
        let mut want = vec![0.0f32; n*h_out*w_out*c_out];
        for nn in 0..n {
            for ho in 0..h_out {
                for wo in 0..w_out {
                    for co in 0..c_out {
                        let mut acc = bias[co];
                        for kh in 0..k_h {
                            let h_in = ho as isize * s_h as isize + kh as isize - p_h as isize;
                            if h_in < 0 || h_in as usize >= h { continue; }
                            for kw in 0..k_w {
                                let w_in = wo as isize * s_w as isize + kw as isize - p_w as isize;
                                if w_in < 0 || w_in as usize >= w { continue; }
                                for ci in 0..c_in {
                                    let in_idx = ((nn*h + h_in as usize)*w + w_in as usize)*c_in + ci;
                                    let w_idx = ((co*k_h + kh)*k_w + kw)*c_in + ci;
                                    acc += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                        want[((nn*h_out + ho)*w_out + wo)*c_out + co] = acc;
                    }
                }
            }
        }

        let args = vec![
            Value::Tensor(HostTensor::from_f32(vec![n, h, w, c_in], &input).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![c_out, k_h, k_w, c_in], &weight).unwrap()),
            Value::Tensor(HostTensor::from_f32(vec![c_out], &bias).unwrap()),
            Value::Scalar(ScalarValue::I32(n as i32)),
            Value::Scalar(ScalarValue::I32(h as i32)),
            Value::Scalar(ScalarValue::I32(w as i32)),
            Value::Scalar(ScalarValue::I32(c_in as i32)),
            Value::Scalar(ScalarValue::I32(c_out as i32)),
            Value::Scalar(ScalarValue::I32(k_h as i32)),
            Value::Scalar(ScalarValue::I32(k_w as i32)),
            Value::Scalar(ScalarValue::I32(s_h as i32)),
            Value::Scalar(ScalarValue::I32(s_w as i32)),
            Value::Scalar(ScalarValue::I32(p_h as i32)),
            Value::Scalar(ScalarValue::I32(p_w as i32)),
        ];
        let result = leaf.execute(0, "conv2d", &args).unwrap();
        let got = match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![n, h_out, w_out, c_out]);
                t.as_f32().unwrap()
            }
            _ => panic!(),
        };
        assert_eq!(want.len(), got.len());
        let max_err = want.iter().zip(got.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "metal conv2d max_abs_err {max_err}");
        eprintln!("metal conv2d 3x3 (N={n} H={h} W={w} C_in={c_in} C_out={c_out}): max_abs_err = {max_err:.3e}");
    }

    #[test]
    fn metal_silu_handles_long_input() {
        let leaf = match metal_or_skip() { Some(l) => l, None => return };
        let xs: Vec<f32> = (0..65536).map(|i| ((i as f32 / 1000.0) - 30.0).sin()).collect();
        let arg = t1(&xs);
        let result = leaf.execute(0, "silu", std::slice::from_ref(&arg)).unwrap();
        let out = match result { Value::Tensor(t) => t.as_f32().unwrap(), _ => panic!() };
        assert_eq!(out.len(), 65536);
    }
}
