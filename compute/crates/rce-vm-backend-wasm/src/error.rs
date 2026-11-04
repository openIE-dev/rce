use thiserror::Error;

pub type Result<T> = std::result::Result<T, WasmError>;

#[derive(Debug, Error)]
pub enum WasmError {
    #[error("wat parse: {0}")]
    Wat(String),

    #[error("wasm engine init: {0}")]
    Engine(String),

    #[error("wasm compile: {0}")]
    Compile(String),

    #[error("wasm instantiate: {0}")]
    Instantiate(String),

    #[error("missing export `{0}` in wasm module (v0.1 expects `rce_dispatch`)")]
    MissingExport(String),

    #[error("wasm function arity mismatch: expected {expected} args, got {got}")]
    ArityMismatch { expected: usize, got: usize },

    #[error("wasm function call: {0}")]
    Call(String),

    #[error("unsupported wasm value type: {0}")]
    UnsupportedValType(String),

    #[error("wasm function returned {got} values, rce leaf protocol expects exactly 1")]
    ResultArity { got: usize },

    #[error("hex decode: {0}")]
    Hex(String),
}
