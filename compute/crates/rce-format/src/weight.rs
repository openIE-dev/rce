//! Weight manifest + sidecar (`.fgw`) format.
//!
//! Weights live in a sidecar file by default to keep the graph blob small
//! and streaming-friendly. The manifest is part of the main `.fg` and points
//! into the sidecar by `(offset, length)` keyed by tensor name.
//!
//! Sidecar layout (v1): `magic("FGW\1") | count_u32 | entries[count] | blob...`
//! where each entry is `name_len_u16 | name | offset_u64 | length_u64 |
//! shape_rank_u8 | shape[rank]_u64 | dtype_len_u8 | dtype`.
//!
//! Future: align with safetensors-2 / GGUF-3 by changing the magic + manifest
//! schema; the wrapper container survives those changes.

use crate::error::{FgError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightEntry {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub offset: u64,
    pub length: u64,
    /// BLAKE3-256 hash of the raw bytes; readers SHOULD verify on first load.
    pub blake3: [u8; 32],
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct WeightManifest {
    pub entries: Vec<WeightEntry>,
}

impl WeightManifest {
    /// Locate a weight entry by tensor name.
    pub fn lookup(&self, name: &str) -> Option<&WeightEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Total bytes of weight payload (sum of entry lengths).
    pub fn total_bytes(&self) -> u64 {
        self.entries.iter().map(|e| e.length).sum()
    }

    /// Validate that no two entries overlap and offsets are within bounds
    /// of `payload_len` (sidecar file or inline blob length).
    pub fn validate_layout(&self, payload_len: u64) -> Result<()> {
        let mut sorted: Vec<&WeightEntry> = self.entries.iter().collect();
        sorted.sort_by_key(|e| e.offset);
        let mut last_end: u64 = 0;
        for e in sorted {
            if e.offset < last_end {
                return Err(FgError::InvalidToc(format!(
                    "weight '{}' overlaps prior entry (offset {} < end {})",
                    e.name, e.offset, last_end
                )));
            }
            let end = e
                .offset
                .checked_add(e.length)
                .ok_or_else(|| FgError::InvalidToc("weight offset+length overflow".into()))?;
            if end > payload_len {
                return Err(FgError::InvalidToc(format!(
                    "weight '{}' extends past payload (end {} > len {})",
                    e.name, end, payload_len
                )));
            }
            last_end = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrip() {
        let m = WeightManifest {
            entries: vec![
                WeightEntry {
                    name: "model.embed.weight".into(),
                    dtype: "f16".into(),
                    shape: vec![32000, 768],
                    offset: 0,
                    length: 32000 * 768 * 2,
                    blake3: [1u8; 32],
                },
                WeightEntry {
                    name: "model.layer.0.mlp.gate".into(),
                    dtype: "f16".into(),
                    shape: vec![3072, 768],
                    offset: 32000 * 768 * 2,
                    length: 3072 * 768 * 2,
                    blake3: [2u8; 32],
                },
            ],
        };
        let bytes = crate::section::encode_cbor(&m).unwrap();
        let m2: WeightManifest = crate::section::decode_cbor(&bytes).unwrap();
        assert_eq!(m, m2);
        m2.validate_layout((32000 + 3072) * 768 * 2).unwrap();
    }

    #[test]
    fn detects_overlap() {
        let m = WeightManifest {
            entries: vec![
                WeightEntry {
                    name: "a".into(),
                    dtype: "f32".into(),
                    shape: vec![10],
                    offset: 0,
                    length: 40,
                    blake3: [0u8; 32],
                },
                WeightEntry {
                    name: "b".into(),
                    dtype: "f32".into(),
                    shape: vec![10],
                    offset: 20, // overlaps
                    length: 40,
                    blake3: [0u8; 32],
                },
            ],
        };
        assert!(m.validate_layout(1024).is_err());
    }
}
