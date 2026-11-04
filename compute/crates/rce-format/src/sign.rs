//! Detached Ed25519 signatures over the file's content hash.
//!
//! A `Signatures` section holds a list of (publisher_id, signature) pairs
//! signing the 32-byte BLAKE3 content hash from the header. Multi-sig is
//! supported by including multiple entries; verification policy (M-of-N,
//! threshold, allowlist) is the consumer's responsibility.

use crate::error::{FgError, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// On-disk record. The fixed-size byte arrays are stored as `Vec<u8>` for
/// serde compatibility; constructors enforce the length.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureEntry {
    /// 32-byte Ed25519 public key.
    #[serde(with = "serde_bytes")]
    pub pubkey: Vec<u8>,
    /// 64-byte signature over `content_hash`.
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
    /// Optional human-readable key identifier (e.g. `did:key:...`).
    #[serde(default)]
    pub key_id: String,
}

impl SignatureEntry {
    fn pubkey_array(&self) -> Result<[u8; 32]> {
        self.pubkey
            .as_slice()
            .try_into()
            .map_err(|_| FgError::BadSignature)
    }

    fn signature_array(&self) -> Result<[u8; 64]> {
        self.signature
            .as_slice()
            .try_into()
            .map_err(|_| FgError::BadSignature)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Signatures {
    pub entries: Vec<SignatureEntry>,
}

impl Signatures {
    pub fn sign(&mut self, signing_key: &SigningKey, content_hash: &[u8; 32], key_id: &str) {
        let signature: Signature = signing_key.sign(content_hash);
        self.entries.push(SignatureEntry {
            pubkey: signing_key.verifying_key().to_bytes().to_vec(),
            signature: signature.to_bytes().to_vec(),
            key_id: key_id.to_string(),
        });
    }

    /// Verify all signatures over `content_hash`. Returns `Err` on the first
    /// invalid signature; callers wanting partial verification should iterate
    /// `entries` themselves.
    pub fn verify_all(&self, content_hash: &[u8; 32]) -> Result<()> {
        for e in &self.entries {
            let pk_bytes = e.pubkey_array()?;
            let pk = VerifyingKey::from_bytes(&pk_bytes).map_err(|_| FgError::BadSignature)?;
            let sig_bytes = e.signature_array()?;
            let sig = Signature::from_bytes(&sig_bytes);
            pk.verify(content_hash, &sig)
                .map_err(|_| FgError::BadSignature)?;
        }
        Ok(())
    }

    /// Verify that *any one* of the signatures is from a key in `allow`.
    pub fn verify_any_of(
        &self,
        content_hash: &[u8; 32],
        allow: &[[u8; 32]],
    ) -> Result<()> {
        for e in &self.entries {
            let pk_bytes = match e.pubkey_array() {
                Ok(b) => b,
                Err(_) => continue,
            };
            if !allow.iter().any(|k| k == &pk_bytes) {
                continue;
            }
            let pk = VerifyingKey::from_bytes(&pk_bytes).map_err(|_| FgError::BadSignature)?;
            let sig_bytes = e.signature_array()?;
            let sig = Signature::from_bytes(&sig_bytes);
            if pk.verify(content_hash, &sig).is_ok() {
                return Ok(());
            }
        }
        Err(FgError::BadSignature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::OsRng;

    #[test]
    fn sign_and_verify_roundtrip() {
        let mut rng = OsRng;
        let signing = SigningKey::generate(&mut rng);
        let content_hash = [0x42u8; 32];
        let mut sigs = Signatures::default();
        sigs.sign(&signing, &content_hash, "did:key:test");
        sigs.verify_all(&content_hash).unwrap();
    }

    #[test]
    fn detects_tamper() {
        let mut rng = OsRng;
        let signing = SigningKey::generate(&mut rng);
        let content_hash = [0x42u8; 32];
        let bad_hash = [0x99u8; 32];
        let mut sigs = Signatures::default();
        sigs.sign(&signing, &content_hash, "did:key:test");
        assert!(sigs.verify_all(&bad_hash).is_err());
    }
}
