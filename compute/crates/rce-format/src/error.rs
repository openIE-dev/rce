use thiserror::Error;

pub type Result<T> = std::result::Result<T, FgError>;

#[derive(Debug, Error)]
pub enum FgError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid magic bytes (expected {expected:?}, got {got:?})")]
    BadMagic { expected: [u8; 4], got: [u8; 4] },

    #[error("unsupported format version {0}")]
    UnsupportedVersion(u32),

    #[error("unknown hash algorithm code {0}")]
    UnknownHashAlg(u8),

    #[error("unknown section kind {0:#06x}")]
    UnknownSectionKind(u16),

    #[error("content hash mismatch: expected {expected}, got {got}")]
    HashMismatch { expected: String, got: String },

    #[error("CBOR decode: {0}")]
    Cbor(String),

    #[error("CBOR encode: {0}")]
    CborEncode(String),

    #[error("section decode ({kind:?}): {msg}")]
    SectionDecode { kind: u16, msg: String },

    #[error("invalid TOC: {0}")]
    InvalidToc(String),

    #[error("buffer too short: needed {needed} bytes, got {got}")]
    Truncated { needed: usize, got: usize },

    #[error("section too large: {got} bytes (max {max})")]
    SectionTooLarge { got: u64, max: u64 },

    #[error("toc entry count {0} exceeds limit ({1})")]
    TocEntryCountTooLarge(u32, u32),

    #[error("toc length {0} exceeds limit ({1})")]
    TocLenTooLarge(u64, u64),

    #[cfg(feature = "sign")]
    #[error("signature verification failed")]
    BadSignature,
}

impl<T: std::fmt::Debug> From<ciborium::de::Error<T>> for FgError {
    fn from(e: ciborium::de::Error<T>) -> Self {
        FgError::Cbor(format!("{e:?}"))
    }
}

impl<T: std::fmt::Debug> From<ciborium::ser::Error<T>> for FgError {
    fn from(e: ciborium::ser::Error<T>) -> Self {
        FgError::CborEncode(format!("{e:?}"))
    }
}
