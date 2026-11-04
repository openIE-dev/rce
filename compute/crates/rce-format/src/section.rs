use crate::error::{FgError, Result};

/// Section kinds. Stable wire identifiers (u16, little-endian on disk).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum SectionKind {
    /// CBOR-encoded `FunctionGraph` / `ProgramGraph`. Required.
    Graph = 0x0001,
    /// CBOR-encoded type table (named tuples, enums, structural types).
    TypeTable = 0x0002,
    /// CBOR-encoded capability registry index for the module.
    CapabilityIndex = 0x0003,
    /// Binary weight manifest: list of (name, dtype, shape, file-relative-offset, length)
    /// pointing into the sidecar `.fgw`. Or, with `EMBED_WEIGHTS`, into the inline blob.
    WeightManifest = 0x0004,
    /// Inline weight blob (only present if `EMBED_WEIGHTS` flag is set).
    WeightBlob = 0x0005,
    /// Detached Ed25519 signatures over `content_hash`.
    Signatures = 0x0006,
    /// Source maps, port names, capability traces, energy estimates per node.
    DebugInfo = 0x0007,
    /// Reserved for vendor extensions. Bytes 0xFF00–0xFFFF.
    VendorReserved = 0xFF00,
}

impl SectionKind {
    pub fn from_u16(v: u16) -> Result<Self> {
        Ok(match v {
            0x0001 => Self::Graph,
            0x0002 => Self::TypeTable,
            0x0003 => Self::CapabilityIndex,
            0x0004 => Self::WeightManifest,
            0x0005 => Self::WeightBlob,
            0x0006 => Self::Signatures,
            0x0007 => Self::DebugInfo,
            v if (0xFF00..=0xFFFF).contains(&v) => Self::VendorReserved,
            other => return Err(FgError::UnknownSectionKind(other)),
        })
    }
}

/// A section: kind + opaque payload bytes. Encoders/decoders for specific
/// section types live in dedicated modules and operate on `payload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub kind: SectionKind,
    pub payload: Vec<u8>,
}

impl Section {
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(self.payload.clone())
    }

    pub fn decode(kind: SectionKind, bytes: &[u8]) -> Result<Self> {
        Ok(Self {
            kind,
            payload: bytes.to_vec(),
        })
    }
}

/// Helper: encode any serde-serializable value as deterministic CBOR.
pub fn encode_cbor<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf).map_err(FgError::from)?;
    Ok(buf)
}

/// Helper: decode CBOR into a serde-deserializable type.
pub fn decode_cbor<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    ciborium::de::from_reader(bytes).map_err(FgError::from)
}
