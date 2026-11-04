use crate::error::{FgError, Result};
use crate::version::{FormatVersion, FORMAT_VERSION};
use byteorder::{ByteOrder, LittleEndian};

/// Magic bytes: `FG\0\1` — short, distinctive, version-tagged.
pub const MAGIC: [u8; 4] = *b"FG\0\x01";

/// Header is fixed-length 64 bytes. Readers may rely on this.
pub const HEADER_LEN: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HashAlg {
    Blake3 = 1,
}

impl HashAlg {
    pub fn from_u8(b: u8) -> Result<Self> {
        match b {
            1 => Ok(HashAlg::Blake3),
            _ => Err(FgError::UnknownHashAlg(b)),
        }
    }
}

/// Bit flags carried in the header.
pub mod flags {
    /// Weight blobs are inlined into a `WeightManifest` payload section
    /// rather than stored in a sidecar `.fgw`.
    pub const EMBED_WEIGHTS: u32 = 1 << 0;

    /// File contains a `Signatures` section. Verification is the consumer's
    /// responsibility.
    pub const SIGNED: u32 = 1 << 1;

    /// File contains a `DebugInfo` section.
    pub const HAS_DEBUG_INFO: u32 = 1 << 2;

    /// Strict-determinism marker: the producer asserts the graph compiles
    /// to a deterministic plan. Runtimes may use this to gate `Determinism::Strict`.
    pub const STRICT_DETERMINISTIC: u32 = 1 << 3;
}

#[derive(Debug, Clone)]
pub struct FgHeader {
    pub version: FormatVersion,
    pub hash_alg: HashAlg,
    /// BLAKE3-256 content hash over canonical-ordered sections.
    pub content_hash: [u8; 32],
    pub flags: u32,
    /// Offset of the TOC from the start of the header.
    pub toc_off: u64,
    /// Length of the TOC in bytes.
    pub toc_len: u64,
}

impl FgHeader {
    pub fn new_unhashed() -> Self {
        Self {
            version: FormatVersion::CURRENT,
            hash_alg: HashAlg::Blake3,
            content_hash: [0; 32],
            flags: 0,
            toc_off: 0,
            toc_len: 0,
        }
    }

    pub fn encode(&self) -> [u8; HEADER_LEN as usize] {
        let mut buf = [0u8; HEADER_LEN as usize];
        buf[0..4].copy_from_slice(&MAGIC);
        LittleEndian::write_u32(&mut buf[4..8], self.version.to_u32());
        buf[8] = self.hash_alg as u8;
        // 9..16 reserved (zero).
        buf[16..48].copy_from_slice(&self.content_hash);
        LittleEndian::write_u32(&mut buf[48..52], self.flags);
        LittleEndian::write_u64(&mut buf[52..60], self.toc_off);
        // bytes 60..64 hold low 4 bytes of toc_len; we steal an extra 4
        // by widening the layout below — actually we keep toc_len fully
        // by relocating it. See below.
        // Reserved layout was tight: re-route toc_len into the trailing 4 bytes
        // by storing only its u32-truncated form for v0.1; readers MUST
        // consult the TOC's count + per-entry lengths anyway. For safety
        // we cap toc_len at u32::MAX in v0.x.
        // toc_len is a u32 on disk in v0.x. Callers (FgFile::write) bounds-check
        // toc_len against MAX_TOC_LEN (~1.3 MiB) before calling encode, so this
        // saturate is unreachable in practice. Keeping it as a saturating cast
        // rather than an expect() so encode() is infallible — the writer's
        // pre-check is the real gate, and the reader's bounds check would
        // refuse a file with a corrupt toc_len even if both validators were
        // bypassed.
        let toc_len_u32: u32 = u32::try_from(self.toc_len).unwrap_or(u32::MAX);
        LittleEndian::write_u32(&mut buf[60..64], toc_len_u32);
        buf
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < HEADER_LEN as usize {
            return Err(FgError::Truncated {
                needed: HEADER_LEN as usize,
                got: buf.len(),
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != MAGIC {
            return Err(FgError::BadMagic {
                expected: MAGIC,
                got: magic,
            });
        }
        let version_raw = LittleEndian::read_u32(&buf[4..8]);
        let version = FormatVersion::from_u32(version_raw);
        if !FormatVersion::CURRENT.is_compatible_with(version) {
            return Err(FgError::UnsupportedVersion(version_raw));
        }
        let hash_alg = HashAlg::from_u8(buf[8])?;
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&buf[16..48]);
        let flags = LittleEndian::read_u32(&buf[48..52]);
        let toc_off = LittleEndian::read_u64(&buf[52..60]);
        let toc_len = LittleEndian::read_u32(&buf[60..64]) as u64;
        Ok(Self {
            version,
            hash_alg,
            content_hash,
            flags,
            toc_off,
            toc_len,
        })
    }

    pub fn current_format() -> u32 {
        FORMAT_VERSION
    }
}
