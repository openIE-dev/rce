//! `.fg` binary format — rce's on-disk module representation.
//!
//! The format is content-addressed, streaming-loadable, and signature-friendly.
//! See `docs/spec/fg-binary-format.md` for the normative specification.
//!
//! ```text
//! file := header | toc | sections...
//! header := magic("FG\0\1") | version_u32 | hash_alg_u8 | reserved_u8[7]
//!         | content_hash_32B | flags_u32 | toc_off_u64 | toc_len_u64
//! toc    := count_u32 | entries[count] of (kind_u16 | flags_u16 | off_u64 | len_u64)
//! sections: { graph (CBOR), type_table (CBOR), capability_index (CBOR),
//!             weight_manifest (binary), signatures (binary), debug_info (CBOR) }
//! ```
//!
//! The header is fixed at 64 bytes. The TOC is variable-length and lives at
//! `toc_off`. Sections are content-addressed; rearranging or rewriting them
//! changes the file's content hash.

#![cfg_attr(not(feature = "sign"), allow(unused))]

pub mod error;
pub mod header;
pub mod hash;
pub mod section;
pub mod toc;
pub mod version;
pub mod weight;

#[cfg(feature = "sign")]
pub mod sign;

#[cfg(feature = "with-rce-graph")]
pub mod graph_section;

pub use error::{FgError, Result};
pub use header::{FgHeader, HashAlg, HEADER_LEN, MAGIC};
pub use section::{Section, SectionKind};
pub use toc::{Toc, TocEntry};
pub use version::{FormatVersion, FORMAT_VERSION};

use std::io::{Read, Seek, SeekFrom, Write};

/// Hard cap on individual section size (1 GiB). The format spec allows
/// up to `u64::MAX`, but readers must defend against adversarial inputs
/// that claim huge sections to force OOM. Producers wanting larger
/// sections should split them. v0.2 may raise this cap; for now, a value
/// chosen so a single section comfortably holds a multi-GB tensor blob's
/// metadata while staying well clear of pathological allocator failures.
pub const MAX_SECTION_BYTES: u64 = 1 << 30; // 1 GiB

/// Hard cap on TOC entry count. The spec is silent; in practice, even a
/// many-section module has well under 100 entries. We pick 64 K to leave
/// generous headroom while preventing `count_u32 = u32::MAX` attacks.
pub const MAX_TOC_ENTRIES: u32 = 65_536;

/// Implied maximum TOC byte length: count prefix + entries-at-cap.
/// `4 + MAX_TOC_ENTRIES × TOC_ENTRY_LEN`. The header carries
/// `toc_len` directly; we reject larger values to prevent OOM allocations
/// before the TOC is ever read.
pub const MAX_TOC_LEN: u64 =
    4 + (MAX_TOC_ENTRIES as u64) * (toc::TOC_ENTRY_LEN as u64);

/// A complete `.fg` file in memory: header + TOC + sections.
///
/// For streaming/lazy access (large model checkpoints) prefer [`FgReader`]
/// which reads only the header + TOC eagerly and seeks to sections on demand.
#[derive(Debug, Clone)]
pub struct FgFile {
    pub header: FgHeader,
    pub sections: Vec<Section>,
}

impl FgFile {
    /// Construct a new file from a list of sections. The content hash is
    /// computed during [`FgFile::write`]; until then the header carries
    /// `content_hash = [0; 32]`.
    pub fn new(sections: Vec<Section>) -> Self {
        Self {
            header: FgHeader::new_unhashed(),
            sections,
        }
    }

    /// Serialize the file. Computes BLAKE3 content hash over canonical
    /// sections and writes it into the header.
    ///
    /// The `Signatures` section is excluded from the hash by design —
    /// signatures sign the content hash, so including them in the hash
    /// would create a circular dependency. The spec authorises this in §6.
    pub fn write<W: Write + Seek>(&mut self, mut w: W) -> Result<u64> {
        // 1. Lay out: header (fixed 64) | sections... | toc
        // 2. Compute content hash over (sorted-by-kind) section bytes,
        //    skipping Signatures.
        // 3. Patch header.content_hash and rewrite header.

        // Reserve header space.
        let header_pos = w.stream_position()?;
        w.write_all(&[0u8; HEADER_LEN as usize])?;

        // Write sections in TOC order, capture offsets/lengths.
        let mut entries = Vec::with_capacity(self.sections.len());
        let mut hasher = blake3::Hasher::new();
        for section in &self.sections {
            let off = w.stream_position()?;
            let bytes = section.encode()?;
            if section.kind != SectionKind::Signatures {
                hasher.update(&(section.kind as u16).to_le_bytes());
                hasher.update(&(bytes.len() as u64).to_le_bytes());
                hasher.update(&bytes);
            }
            w.write_all(&bytes)?;
            entries.push(TocEntry {
                kind: section.kind,
                flags: 0,
                offset: off,
                length: bytes.len() as u64,
            });
        }

        // Write TOC.
        let toc_off = w.stream_position()?;
        let toc = Toc { entries };
        let toc_bytes = toc.encode()?;
        w.write_all(&toc_bytes)?;
        let toc_len = toc_bytes.len() as u64;

        // Patch header. Bounds-check toc_len here so a malformed graph (e.g.
        // adversarially-crafted section list pushing past MAX_TOC_LEN) errors
        // cleanly at write time rather than producing a file the reader will
        // refuse.
        if toc_len > MAX_TOC_LEN {
            return Err(FgError::TocLenTooLarge(toc_len, MAX_TOC_LEN));
        }
        let content_hash: [u8; 32] = hasher.finalize().into();
        self.header.content_hash = content_hash;
        self.header.toc_off = toc_off - header_pos;
        self.header.toc_len = toc_len;
        let end = w.stream_position()?;
        w.seek(SeekFrom::Start(header_pos))?;
        w.write_all(&self.header.encode())?;
        w.seek(SeekFrom::Start(end))?;
        Ok(end - header_pos)
    }

    /// Parse a complete file from a buffer. For lazy/streaming reads use
    /// [`FgReader`].
    pub fn read<R: Read + Seek>(mut r: R) -> Result<Self> {
        let mut header_buf = [0u8; HEADER_LEN as usize];
        r.read_exact(&mut header_buf)?;
        let header = FgHeader::decode(&header_buf)?;

        // Bounds-check toc_len BEFORE allocating; otherwise an attacker
        // can claim a 4-GiB TOC and force OOM during the seek-and-read.
        if header.toc_len > MAX_TOC_LEN {
            return Err(FgError::TocLenTooLarge(header.toc_len, MAX_TOC_LEN));
        }

        // Seek to TOC.
        r.seek(SeekFrom::Start(header.toc_off))?;
        let mut toc_buf = vec![0u8; header.toc_len as usize];
        r.read_exact(&mut toc_buf)?;
        let toc = Toc::decode(&toc_buf)?;

        // Read each section. Signatures are read but excluded from the
        // hash check (spec §6: signatures sign the content hash, so they
        // can't be part of it).
        let mut sections = Vec::with_capacity(toc.entries.len());
        let mut hasher = blake3::Hasher::new();
        for entry in &toc.entries {
            if entry.length > MAX_SECTION_BYTES {
                return Err(FgError::SectionTooLarge {
                    got: entry.length,
                    max: MAX_SECTION_BYTES,
                });
            }
            r.seek(SeekFrom::Start(entry.offset))?;
            let mut buf = vec![0u8; entry.length as usize];
            r.read_exact(&mut buf)?;
            if entry.kind != SectionKind::Signatures {
                hasher.update(&(entry.kind as u16).to_le_bytes());
                hasher.update(&(buf.len() as u64).to_le_bytes());
                hasher.update(&buf);
            }
            sections.push(Section::decode(entry.kind, &buf)?);
        }

        // Verify content hash.
        let computed: [u8; 32] = hasher.finalize().into();
        if computed != header.content_hash {
            return Err(FgError::HashMismatch {
                expected: hex::encode(header.content_hash),
                got: hex::encode(computed),
            });
        }
        Ok(Self { header, sections })
    }

    /// Find the first section of a given kind.
    pub fn section(&self, kind: SectionKind) -> Option<&Section> {
        self.sections.iter().find(|s| s.kind == kind)
    }
}

/// Streaming reader: parses header + TOC eagerly, leaves sections on disk.
/// Use this for large model checkpoints where you only need a subset of
/// sections (e.g. just the graph, weights via mmap).
pub struct FgReader<R: Read + Seek> {
    inner: R,
    pub header: FgHeader,
    pub toc: Toc,
}

impl<R: Read + Seek> FgReader<R> {
    pub fn open(mut inner: R) -> Result<Self> {
        let mut header_buf = [0u8; HEADER_LEN as usize];
        inner.read_exact(&mut header_buf)?;
        let header = FgHeader::decode(&header_buf)?;
        if header.toc_len > MAX_TOC_LEN {
            return Err(FgError::TocLenTooLarge(header.toc_len, MAX_TOC_LEN));
        }
        inner.seek(SeekFrom::Start(header.toc_off))?;
        let mut toc_buf = vec![0u8; header.toc_len as usize];
        inner.read_exact(&mut toc_buf)?;
        let toc = Toc::decode(&toc_buf)?;
        Ok(Self { inner, header, toc })
    }

    /// Read a single section by kind. Returns `Ok(None)` if absent.
    pub fn read_section(&mut self, kind: SectionKind) -> Result<Option<Section>> {
        let Some(entry) = self.toc.entries.iter().find(|e| e.kind == kind).cloned() else {
            return Ok(None);
        };
        if entry.length > MAX_SECTION_BYTES {
            return Err(FgError::SectionTooLarge {
                got: entry.length,
                max: MAX_SECTION_BYTES,
            });
        }
        self.inner.seek(SeekFrom::Start(entry.offset))?;
        let mut buf = vec![0u8; entry.length as usize];
        self.inner.read_exact(&mut buf)?;
        Ok(Some(Section::decode(entry.kind, &buf)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn header_roundtrip() {
        let mut h = FgHeader::new_unhashed();
        h.flags = 0xDEADBEEF;
        h.toc_off = 1024;
        h.toc_len = 256;
        h.content_hash = [0x42; 32];
        let bytes = h.encode();
        assert_eq!(bytes.len(), HEADER_LEN as usize);
        let h2 = FgHeader::decode(&bytes).unwrap();
        assert_eq!(h.flags, h2.flags);
        assert_eq!(h.toc_off, h2.toc_off);
        assert_eq!(h.toc_len, h2.toc_len);
        assert_eq!(h.content_hash, h2.content_hash);
    }

    #[test]
    fn empty_file_roundtrip() {
        let mut file = FgFile::new(vec![]);
        let mut buf = Cursor::new(Vec::<u8>::new());
        file.write(&mut buf).unwrap();
        buf.set_position(0);
        let read = FgFile::read(buf).unwrap();
        assert_eq!(read.sections.len(), 0);
        assert_eq!(read.header.content_hash, file.header.content_hash);
    }

    #[test]
    fn graph_section_roundtrip() {
        let graph_cbor = b"hello-graph-cbor".to_vec();
        let section = Section {
            kind: SectionKind::Graph,
            payload: graph_cbor.clone(),
        };
        let mut file = FgFile::new(vec![section]);
        let mut buf = Cursor::new(Vec::<u8>::new());
        file.write(&mut buf).unwrap();
        buf.set_position(0);
        let read = FgFile::read(buf).unwrap();
        let got = read.section(SectionKind::Graph).unwrap();
        assert_eq!(got.payload, graph_cbor);
    }

    #[test]
    fn detects_corruption() {
        let section = Section {
            kind: SectionKind::Graph,
            payload: b"original".to_vec(),
        };
        let mut file = FgFile::new(vec![section]);
        let mut buf = Cursor::new(Vec::<u8>::new());
        file.write(&mut buf).unwrap();

        // Flip a byte deep in the section payload.
        let bytes = buf.get_mut();
        let target = (HEADER_LEN as usize) + 4;
        bytes[target] ^= 0xFF;
        buf.set_position(0);
        let err = FgFile::read(buf).unwrap_err();
        assert!(matches!(err, FgError::HashMismatch { .. }));
    }
}
