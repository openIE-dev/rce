//! The BLAKE3 hash chain and its JSONL ledger file.
use crate::{Body, Draft, GENESIS, Receipt, SCHEMA, delta};
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VerifyError {
    #[error("receipt {0}: hash does not match body")]
    Hash(u64),
    #[error("receipt {0}: signed by a different key")]
    Signer(u64),
    #[error("receipt {0}: bad signature")]
    Signature(u64),
    #[error("receipt {0}: prev_hash does not link to the previous receipt")]
    Link(u64),
    #[error("receipt {0}: sequence number out of order")]
    Seq(u64),
    #[error("receipt {0}: delta_j does not equal e_meas - e_sim")]
    Delta(u64),
}

/// Check every receipt, its signature, its delta and its link to the one
/// before. `first_prev` is `GENESIS` for a full chain.
pub fn verify_chain(
    rs: &[Receipt],
    key: &VerifyingKey,
    first_prev: &str,
) -> Result<(), VerifyError> {
    let mut prev = first_prev.to_string();
    let first = rs.first().map(|r| r.body.seq).unwrap_or(0);
    for (seq, r) in (first..).zip(rs) {
        r.verify(key)?;
        if r.body.seq != seq {
            return Err(VerifyError::Seq(r.body.seq));
        }
        if r.body.prev_hash != prev {
            return Err(VerifyError::Link(r.body.seq));
        }
        if r.body.delta_j != delta(r.body.draft.e_sim_j, r.body.draft.e_meas) {
            return Err(VerifyError::Delta(r.body.seq));
        }
        prev = r.hash.clone();
    }
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum LedgerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("ledger line {0}: {1}")]
    Parse(usize, serde_json::Error),
    #[error("existing ledger fails verification: {0}")]
    Verify(#[from] VerifyError),
}

/// An append-only chain, optionally backed by a JSONL file (one receipt per line).
pub struct Ledger {
    key: SigningKey,
    path: Option<PathBuf>,
    last_hash: String,
    next_seq: u64,
}

impl Ledger {
    /// In-memory ledger starting at genesis.
    pub fn new(key: SigningKey) -> Self {
        Ledger {
            key,
            path: None,
            last_hash: GENESIS.into(),
            next_seq: 0,
        }
    }

    /// Open (or create) a JSONL ledger. An existing file is verified in full
    /// against `key` before anything is appended.
    pub fn open(path: &Path, key: SigningKey) -> Result<Self, LedgerError> {
        let mut l = Ledger {
            key,
            path: Some(path.to_path_buf()),
            last_hash: GENESIS.into(),
            next_seq: 0,
        };
        if path.exists() {
            let rs = read_jsonl(path)?;
            verify_chain(&rs, &l.key.verifying_key(), GENESIS)?;
            if let Some(last) = rs.last() {
                l.last_hash = last.hash.clone();
                l.next_seq = last.body.seq + 1;
            }
        }
        Ok(l)
    }

    pub fn verifying_key(&self) -> VerifyingKey {
        self.key.verifying_key()
    }

    pub fn head(&self) -> &str {
        &self.last_hash
    }

    /// Seal a draft into the next receipt and append it.
    pub fn append(&mut self, draft: Draft) -> Result<Receipt, LedgerError> {
        let body = Body {
            schema: SCHEMA.into(),
            seq: self.next_seq,
            delta_j: delta(draft.e_sim_j, draft.e_meas),
            draft,
            prev_hash: self.last_hash.clone(),
        };
        let r = Receipt::sign(body, &self.key);
        if let Some(p) = &self.path {
            let mut f = OpenOptions::new().create(true).append(true).open(p)?;
            let mut line = serde_json::to_vec(&r).expect("receipt serializes");
            line.push(b'\n');
            f.write_all(&line)?;
            f.sync_data()?;
        }
        self.last_hash = r.hash.clone();
        self.next_seq += 1;
        Ok(r)
    }
}

/// Read a JSONL ledger.
pub fn read_jsonl(path: &Path) -> Result<Vec<Receipt>, LedgerError> {
    let f = BufReader::new(File::open(path)?);
    let mut out = vec![];
    for (i, line) in f.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        out.push(serde_json::from_str(&line).map_err(|e| LedgerError::Parse(i + 1, e))?);
    }
    Ok(out)
}
