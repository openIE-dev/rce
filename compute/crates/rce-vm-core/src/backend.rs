use crate::cost::Cost;
use crate::plan::{Plan, PlanError};
use crate::tensor::TensorHandle;
use crate::value::Value;
use thiserror::Error;

/// Stable identifier for a backend instance (e.g. `cpu`, `wgpu:apple-m3`,
/// `cuda:0`, `wasm:wasmtime`, `silicon:tier-a:DeepCausalModelError`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BackendId(pub String);

impl BackendId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Wgpu,
    Cuda,
    Metal,
    Rocm,
    Npu,
    WasmLeaf,
    OnnxLeaf,
    Silicon,
}

/// Static capabilities a backend advertises. The scheduler queries this
/// before dispatching a sub-graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Capabilities {
    pub kind: Option<BackendKind>,
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub supports_f32: bool,
    pub supports_f64: bool,
    pub supports_i8: bool,
    pub supports_i32: bool,
    pub supports_i64: bool,
    /// Whether the backend can host weights resident in device memory.
    pub device_memory: bool,
    /// Whether the backend honours `Determinism::Strict` mode (single-threaded
    /// reductions, fixed work order, IEEE-754-strict arithmetic).
    pub strict_deterministic: bool,
    /// Whether the backend can execute Wasm component imports as leaves.
    pub wasm_leaf: bool,
}

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("plan error: {0}")]
    Plan(#[from] PlanError),

    #[error("backend `{0:?}` does not support op {1}")]
    UnsupportedOp(BackendKind, String),

    #[error("device out of memory: needed {needed} bytes")]
    OutOfMemory { needed: u64 },

    #[error("capability denied: {0}")]
    CapabilityDenied(String),

    #[error("backend driver error: {0}")]
    Driver(String),
}

/// Object-safe trait every backend implements.
///
/// `compile` produces a `Plan` (a backend-specific representation of how
/// the sub-graph will execute). `dispatch` runs the plan against bound
/// inputs and returns outputs. `cost` returns an energy/time estimate the
/// scheduler uses for placement.
pub trait Backend: Send + Sync {
    fn id(&self) -> &BackendId;
    fn capabilities(&self) -> Capabilities;

    /// Compile a graph slice into a backend-specific plan.
    fn compile(&self, slice: &GraphSlice) -> Result<Plan, DispatchError>;

    /// Dispatch a compiled plan with bound inputs. Returns outputs in the
    /// order declared by the plan.
    fn dispatch(
        &self,
        plan: &Plan,
        inputs: &[TensorHandle],
    ) -> Result<Vec<Value>, DispatchError>;

    /// Cost estimate for a compiled plan. Used by the placement pass.
    fn cost(&self, plan: &Plan) -> Cost;

    /// Cancel an in-flight dispatch. Best-effort; backends without
    /// cancellation return `Ok(())` without action.
    fn cancel(&self, _handle: u64) -> Result<(), DispatchError> {
        Ok(())
    }
}

/// A subgraph slice handed to a backend for compilation. Opaque structure:
/// in v0.1 this is a serialized `rce-format::Graph` payload plus an
/// entry-node index. Backends decode it via `rce_format::section::decode_cbor`
/// or via the `with-rce-graph` feature.
#[derive(Debug, Clone)]
pub struct GraphSlice {
    pub graph_cbor: Vec<u8>,
    pub entry_node: u32,
    pub strict_deterministic: bool,
}
