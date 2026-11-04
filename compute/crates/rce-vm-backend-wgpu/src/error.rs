use thiserror::Error;

pub type Result<T> = std::result::Result<T, WgpuError>;

#[derive(Debug, Error)]
pub enum WgpuError {
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    #[error("wgpu device request failed: {0}")]
    DeviceRequest(String),

    #[error("wgpu shader compile: {0}")]
    Shader(String),

    #[error("buffer mapping: {0}")]
    Mapping(String),

    #[error("invalid input: {0}")]
    Invalid(String),

    #[error("unsupported dtype: {0}")]
    UnsupportedDtype(String),
}
