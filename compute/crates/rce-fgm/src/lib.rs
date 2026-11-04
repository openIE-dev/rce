//! RCE's content-addressed store of tensor data (spec 6.13, GT-9).
//!
//! A *blob* is the bytes of one tensor: its elements in row-major order,
//! each in little-endian bytes (`f32` 4, `f64` 8, `i32` 4, `i8` 1, two's
//! complement). Its *digest* is the BLAKE3 hash of those bytes, written as
//! 64 lowercase hexadecimal digits. A program names a weight by its digest
//! (`tensor.blob_f32(DIGEST, SHAPE)`), never by a path, so the program's
//! hash covers its weights and a blob can be fetched from any store that
//! holds it. A [`Store`] is a directory holding each blob as the file
//! `DIGEST`; reading one checks its hash, so a store can be shared, copied
//! or cached without trust in where it came from.
//!
//! This crate was earlier a single-file model artifact (`.fgm`: a graph
//! and its weights in one file); in RCE the computation is the program's
//! source and only the tensor data lives here.

use std::io::Write;
use std::path::{Path, PathBuf};

/// The BLAKE3 hash of a blob.
pub type Digest = [u8; 32];

/// The digest of `bytes`.
pub fn digest(bytes: &[u8]) -> Digest {
    *blake3::hash(bytes).as_bytes()
}

/// A digest as 64 lowercase hexadecimal digits.
pub fn hex(d: &Digest) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// The digest 64 lowercase hexadecimal digits name.
pub fn parse_hex(s: &str) -> Option<Digest> {
    if s.len() != 64 || !s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
        return None;
    }
    let mut d = [0u8; 32];
    for (i, b) in d.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok()?;
    }
    Some(d)
}

/// Why a blob could not be read.
#[derive(Debug)]
pub enum StoreError {
    /// the store holds no blob of this digest
    Missing(String),
    /// the file named by the digest holds other bytes
    Mismatch { digest: String, found: String },
    Io(std::io::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StoreError::Missing(d) => write!(f, "the store holds no blob {d}"),
            StoreError::Mismatch { digest, found } => write!(f, "the blob {digest} holds bytes whose digest is {found}"),
            StoreError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StoreError {}

/// A directory of blobs, each the file named by its digest.
#[derive(Clone, Debug)]
pub struct Store {
    dir: PathBuf,
}

impl Store {
    pub fn new(dir: impl Into<PathBuf>) -> Store {
        Store { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Add `bytes` and give their digest. The file is written whole under
    /// a temporary name and renamed, so a reader never sees part of it; a
    /// blob already present is left as it is.
    pub fn put(&self, bytes: &[u8]) -> std::io::Result<Digest> {
        let d = digest(bytes);
        let path = self.dir.join(hex(&d));
        if path.exists() {
            return Ok(d);
        }
        std::fs::create_dir_all(&self.dir)?;
        let tmp = self.dir.join(format!(".{}.{}", hex(&d), std::process::id()));
        {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(d)
    }

    /// Whether the store has a file for `d` (not yet checked).
    pub fn has(&self, d: &Digest) -> bool {
        self.dir.join(hex(d)).is_file()
    }

    /// The bytes of the blob `d`, checked against their digest.
    pub fn get(&self, d: &Digest) -> Result<Vec<u8>, StoreError> {
        let path = self.dir.join(hex(d));
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(StoreError::Missing(hex(d))),
            Err(e) => return Err(StoreError::Io(e)),
        };
        let got = digest(&bytes);
        if &got != d {
            return Err(StoreError::Mismatch { digest: hex(d), found: hex(&got) });
        }
        Ok(bytes)
    }
}

/// The blob of an `f32` tensor.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The blob of an `f64` tensor.
pub fn f64_bytes(v: &[f64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The blob of an `i32` tensor.
pub fn i32_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The blob of an `i8` tensor.
pub fn i8_bytes(v: &[i8]) -> Vec<u8> {
    v.iter().map(|x| *x as u8).collect()
}

/// The element size of a tensor element type, in bytes.
pub fn elem_size(elem: &str) -> Option<usize> {
    match elem {
        "f32" | "i32" => Some(4),
        "f64" => Some(8),
        "i8" => Some(1),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blob_is_named_by_its_digest_and_checked_when_read() {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::new(dir.path().join("blobs"));
        let b = f32_bytes(&[1.0, -2.5, 0.0]);
        let d = s.put(&b).unwrap();
        assert_eq!(d, digest(&b));
        assert_eq!(parse_hex(&hex(&d)), Some(d));
        assert_eq!(s.put(&b).unwrap(), d, "a second put keeps the blob");
        assert_eq!(s.get(&d).unwrap(), b);
        // the file changed under its name: refused
        std::fs::write(s.dir().join(hex(&d)), [0u8; 12]).unwrap();
        assert!(matches!(s.get(&d), Err(StoreError::Mismatch { .. })));
        assert!(matches!(s.get(&digest(b"other")), Err(StoreError::Missing(_))));
        assert_eq!(parse_hex("ABC"), None);
        assert_eq!(i8_bytes(&[-1, 2]), vec![255, 2]);
    }
}
