use crate::error::{FgError, Result};
use crate::section::SectionKind;
use crate::MAX_TOC_ENTRIES;
use byteorder::{ByteOrder, LittleEndian};

/// Per-section TOC entry. 24 bytes on disk: u16 kind + u16 flags + u64 off + u64 len.
#[derive(Debug, Clone, Copy)]
pub struct TocEntry {
    pub kind: SectionKind,
    pub flags: u16,
    pub offset: u64,
    pub length: u64,
}

pub const TOC_ENTRY_LEN: usize = 2 + 2 + 8 + 8; // 20 bytes

/// Table of contents.
///
/// On-disk layout: `count_u32 | entry_0 | entry_1 | ... | entry_{count-1}`.
#[derive(Debug, Clone, Default)]
pub struct Toc {
    pub entries: Vec<TocEntry>,
}

impl Toc {
    pub fn encode(&self) -> Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(4 + TOC_ENTRY_LEN * self.entries.len());
        let mut count_buf = [0u8; 4];
        LittleEndian::write_u32(&mut count_buf, self.entries.len() as u32);
        buf.extend_from_slice(&count_buf);
        for e in &self.entries {
            let mut entry_buf = [0u8; TOC_ENTRY_LEN];
            LittleEndian::write_u16(&mut entry_buf[0..2], e.kind as u16);
            LittleEndian::write_u16(&mut entry_buf[2..4], e.flags);
            LittleEndian::write_u64(&mut entry_buf[4..12], e.offset);
            LittleEndian::write_u64(&mut entry_buf[12..20], e.length);
            buf.extend_from_slice(&entry_buf);
        }
        Ok(buf)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            return Err(FgError::InvalidToc("missing count prefix".into()));
        }
        let count_u32 = LittleEndian::read_u32(&bytes[0..4]);
        if count_u32 > MAX_TOC_ENTRIES {
            return Err(FgError::TocEntryCountTooLarge(count_u32, MAX_TOC_ENTRIES));
        }
        let count = count_u32 as usize;
        let expected_len = 4 + count * TOC_ENTRY_LEN;
        if bytes.len() < expected_len {
            return Err(FgError::InvalidToc(format!(
                "truncated: expected {expected_len} bytes for {count} entries, got {}",
                bytes.len()
            )));
        }
        let mut entries = Vec::with_capacity(count);
        for i in 0..count {
            let off = 4 + i * TOC_ENTRY_LEN;
            let kind_raw = LittleEndian::read_u16(&bytes[off..off + 2]);
            let kind = SectionKind::from_u16(kind_raw)?;
            let flags = LittleEndian::read_u16(&bytes[off + 2..off + 4]);
            let offset = LittleEndian::read_u64(&bytes[off + 4..off + 12]);
            let length = LittleEndian::read_u64(&bytes[off + 12..off + 20]);
            entries.push(TocEntry {
                kind,
                flags,
                offset,
                length,
            });
        }
        Ok(Self { entries })
    }
}
