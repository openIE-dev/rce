use thiserror::Error;

pub type Result<T> = std::result::Result<T, InterpretError>;

#[derive(Debug, Error)]
pub enum InterpretError {
    #[error("unbound parameter `{0}`")]
    UnboundParam(String),

    #[error("output node {0} out of range (graph has {1} nodes)")]
    OutputNodeOutOfRange(usize, usize),

    #[error("graph is cyclic — interpreter requires a DAG")]
    Cyclic,

    #[error("node {node_id} ({kind}) expected {expected} input(s), got {got}")]
    ArityMismatch {
        node_id: usize,
        kind: String,
        expected: usize,
        got: usize,
    },

    #[error("type mismatch in node {node_id}: {msg}")]
    TypeMismatch { node_id: usize, msg: String },

    #[error("unsupported op `{0}` (custom ops, while-loops, capabilities are out of v0.1 scope)")]
    Unsupported(String),

    #[error("division by zero in node {0}")]
    DivByZero(usize),

    /// A float element of node `{0}`'s value is NaN or an infinity. RCE's
    /// floats are finite (spec 6.3 N3), so the reference evaluation stops
    /// at the first such node in its fixed order with the trap
    /// `non-finite float`.
    #[error("non-finite float in node {0}")]
    NonFinite(usize),

    /// A trap of RCE's reference evaluation (spec 6.3): an integer result
    /// outside its type (`overflow: V is outside T`), an integer division by
    /// zero, an index out of range. The text is RCE's trap text.
    #[error("trap in node {0}: {1}")]
    Trap(usize, String),

    #[error("invalid wire: {0}")]
    InvalidWire(String),

    #[error("internal: {0}")]
    Internal(String),
}
