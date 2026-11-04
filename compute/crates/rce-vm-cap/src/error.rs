use thiserror::Error;

pub type Result<T> = std::result::Result<T, CapError>;

#[derive(Debug, Error)]
pub enum CapError {
    #[error("capability `{0}` not registered in this CapTable")]
    Unregistered(String),

    #[error("capability `{capability}`/`{method}` does not exist (id known, method missing)")]
    UnknownMethod { capability: String, method: String },

    #[error("effect `{effect}` requested by capability `{capability}` is not in the host's allow-list")]
    EffectDenied { capability: String, effect: String },

    #[error("delegation chain rejected: {0}")]
    BadChain(String),

    #[error("capability handler returned an error: {0}")]
    Handler(String),

    #[error("type mismatch: {0}")]
    TypeMismatch(String),
}
