use thiserror::Error;

pub type Result<T> = std::result::Result<T, ScheduleError>;

#[derive(Debug, Error)]
pub enum ScheduleError {
    #[error("interpret error: {0}")]
    Interpret(#[from] rce_vm_backend_cpu::InterpretError),

    #[error("output node {0} out of range (graph has {1} nodes)")]
    OutputNodeOutOfRange(usize, usize),

    #[error("graph is cyclic — scheduler requires a DAG")]
    Cyclic,

    #[error("invalid wire: {0}")]
    InvalidWire(String),

    #[error("unbound parameter `{0}`")]
    UnboundParam(String),

    #[error("internal: input channel for node {0} closed before producer fired")]
    InputChannelClosed(usize),

    #[error("internal: tokio join error: {0}")]
    Join(String),
}
