//! Ed25519-signed delegation chains.
//!
//! A delegation: signer A grants subject B the right to exercise
//! capability `cap_id`, optionally restricted to `effects` and
//! `expires_at`. Each delegation is signed by A's private key over
//! `BLAKE3(cap_id || subject || effects || expires)`.
//!
//! A chain is a sequence of delegations rooted at a known issuer. The
//! end of the chain (the most-recent delegation's subject) is the
//! current bearer; the host validates root-of-trust by matching the
//! root delegation's signer against an allow-list.

use crate::error::{CapError, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delegation {
    /// Capability id this delegation is about (e.g. `joulesperbit.pay.ledger`).
    pub capability_id: String,
    /// Subject's Ed25519 public key — the holder of this delegation.
    #[serde(with = "serde_bytes_array")]
    pub subject: [u8; 32],
    /// Effects the subject is permitted (subset of the issuer's own).
    pub effects: Vec<String>,
    /// UNIX time after which the delegation is invalid (0 = no expiry).
    pub expires_at: u64,
    /// Issuer's Ed25519 public key.
    #[serde(with = "serde_bytes_array")]
    pub issuer: [u8; 32],
    /// Issuer's signature over `BLAKE3(capability_id || subject || effects || expires_at)`.
    #[serde(with = "serde_bytes_64")]
    pub signature: [u8; 64],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationChain {
    /// Root-most delegation first.
    pub links: Vec<Delegation>,
}

/// Effects-allow policy: a host can refuse a capability that requests
/// an effect not in this set.
#[derive(Debug, Clone, Default)]
pub struct EffectsAllow {
    pub allowed: Vec<String>,
}

impl EffectsAllow {
    pub fn permits_all(allowed: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            allowed: allowed.into_iter().map(Into::into).collect(),
        }
    }

    pub fn check(&self, capability: &str, requested: &[String]) -> Result<()> {
        for e in requested {
            if !self.allowed.iter().any(|a| a == e) {
                return Err(CapError::EffectDenied {
                    capability: capability.into(),
                    effect: e.clone(),
                });
            }
        }
        Ok(())
    }
}

impl Delegation {
    pub fn sign_new(
        signing: &SigningKey,
        capability_id: &str,
        subject: [u8; 32],
        effects: &[String],
        expires_at: u64,
    ) -> Self {
        let issuer = signing.verifying_key().to_bytes();
        let payload = chain_payload(capability_id, &subject, effects, expires_at);
        let signature: Signature = signing.sign(&payload);
        Self {
            capability_id: capability_id.into(),
            subject,
            effects: effects.to_vec(),
            expires_at,
            issuer,
            signature: signature.to_bytes(),
        }
    }

    pub fn verify(&self) -> Result<()> {
        let payload = chain_payload(
            &self.capability_id,
            &self.subject,
            &self.effects,
            self.expires_at,
        );
        let pk = VerifyingKey::from_bytes(&self.issuer)
            .map_err(|e| CapError::BadChain(format!("issuer key invalid: {e}")))?;
        let sig = Signature::from_bytes(&self.signature);
        pk.verify(&payload, &sig)
            .map_err(|e| CapError::BadChain(format!("signature: {e}")))?;
        Ok(())
    }
}

impl DelegationChain {
    /// Validate the chain end-to-end: every link's signature is correct,
    /// every link's issuer matches the previous link's subject, and
    /// effects of each link are a subset of the previous link's effects.
    /// The root link's issuer must be in `roots_of_trust`.
    pub fn validate(
        &self,
        roots_of_trust: &[[u8; 32]],
        capability_id: &str,
        now: u64,
    ) -> Result<()> {
        if self.links.is_empty() {
            return Err(CapError::BadChain("empty chain".into()));
        }
        for (i, link) in self.links.iter().enumerate() {
            if link.capability_id != capability_id {
                return Err(CapError::BadChain(format!(
                    "link {i}: capability_id mismatch ({} vs {})",
                    link.capability_id, capability_id
                )));
            }
            if link.expires_at != 0 && link.expires_at < now {
                return Err(CapError::BadChain(format!(
                    "link {i}: expired at {} (now {now})",
                    link.expires_at
                )));
            }
            link.verify()?;
        }
        // Root issuer must be in allow-list.
        let root_issuer = self.links[0].issuer;
        if !roots_of_trust.iter().any(|r| r == &root_issuer) {
            return Err(CapError::BadChain(format!(
                "root issuer {} not in allow-list",
                hex::encode(root_issuer)
            )));
        }
        // Each subsequent link's issuer must equal the previous link's subject.
        for window in self.links.windows(2) {
            if window[0].subject != window[1].issuer {
                return Err(CapError::BadChain(format!(
                    "chain break: {} → {}",
                    hex::encode(window[0].subject),
                    hex::encode(window[1].issuer)
                )));
            }
            // Effects monotonicity: child effects ⊆ parent effects.
            for e in &window[1].effects {
                if !window[0].effects.iter().any(|p| p == e) {
                    return Err(CapError::BadChain(format!(
                        "effect `{e}` granted to child but not held by parent"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The bearer of the chain — public key of the most-recent subject.
    pub fn bearer(&self) -> Option<[u8; 32]> {
        self.links.last().map(|l| l.subject)
    }

    /// Effects authorised by the leaf delegation.
    pub fn effects(&self) -> &[String] {
        self.links
            .last()
            .map(|l| l.effects.as_slice())
            .unwrap_or(&[])
    }
}

fn chain_payload(capability_id: &str, subject: &[u8; 32], effects: &[String], expires: u64) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(capability_id.as_bytes());
    h.update(subject);
    for e in effects {
        h.update(e.as_bytes());
        h.update(b"\0");
    }
    h.update(&expires.to_le_bytes());
    h.finalize().as_bytes().to_vec()
}

mod serde_bytes_array {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let v: Vec<u8> = serde_bytes::ByteBuf::deserialize(d)?.into_vec();
        v.try_into().map_err(|_| serde::de::Error::custom("expected 32-byte slice"))
    }
}

mod serde_bytes_64 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(v: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(v)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let v: Vec<u8> = serde_bytes::ByteBuf::deserialize(d)?.into_vec();
        v.try_into().map_err(|_| serde::de::Error::custom("expected 64-byte slice"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use getrandom::SysRng;
    use rand_core::UnwrapErr;

    #[test]
    fn round_trip_single_link_chain() {
        let mut rng = UnwrapErr(SysRng);
        let issuer = SigningKey::generate(&mut rng);
        let subject = SigningKey::generate(&mut rng);
        let d = Delegation::sign_new(
            &issuer,
            "test.cap",
            subject.verifying_key().to_bytes(),
            &["io:read".into()],
            0,
        );
        let chain = DelegationChain { links: vec![d] };
        chain
            .validate(&[issuer.verifying_key().to_bytes()], "test.cap", 0)
            .unwrap();
    }

    #[test]
    fn detects_chain_break() {
        let mut rng = UnwrapErr(SysRng);
        let issuer = SigningKey::generate(&mut rng);
        let alice = SigningKey::generate(&mut rng);
        let _bob = SigningKey::generate(&mut rng);
        let charlie = SigningKey::generate(&mut rng);
        // issuer → alice
        let d1 = Delegation::sign_new(
            &issuer,
            "test.cap",
            alice.verifying_key().to_bytes(),
            &["io:read".into()],
            0,
        );
        // BUT charlie tries to claim to be a delegate of alice without alice's signature
        // We forge a delegation issued by *bob* (not alice) to charlie.
        let d2 = Delegation::sign_new(
            &SigningKey::generate(&mut rng),
            "test.cap",
            charlie.verifying_key().to_bytes(),
            &["io:read".into()],
            0,
        );
        let chain = DelegationChain { links: vec![d1, d2] };
        let res = chain.validate(&[issuer.verifying_key().to_bytes()], "test.cap", 0);
        assert!(matches!(res, Err(CapError::BadChain(_))));
    }

    #[test]
    fn detects_effect_escalation() {
        let mut rng = UnwrapErr(SysRng);
        let issuer = SigningKey::generate(&mut rng);
        let alice = SigningKey::generate(&mut rng);
        let bob = SigningKey::generate(&mut rng);
        // Alice has io:read only.
        let d1 = Delegation::sign_new(
            &issuer,
            "test.cap",
            alice.verifying_key().to_bytes(),
            &["io:read".into()],
            0,
        );
        // Alice tries to grant Bob io:write, which she doesn't have.
        let d2 = Delegation::sign_new(
            &alice,
            "test.cap",
            bob.verifying_key().to_bytes(),
            &["io:read".into(), "io:write".into()],
            0,
        );
        let chain = DelegationChain { links: vec![d1, d2] };
        let res = chain.validate(&[issuer.verifying_key().to_bytes()], "test.cap", 0);
        assert!(matches!(res, Err(CapError::BadChain(_))));
    }

    #[test]
    fn rejects_unknown_root() {
        let mut rng = UnwrapErr(SysRng);
        let real_issuer = SigningKey::generate(&mut rng);
        let attacker = SigningKey::generate(&mut rng);
        let alice = SigningKey::generate(&mut rng);
        let d = Delegation::sign_new(
            &attacker,
            "test.cap",
            alice.verifying_key().to_bytes(),
            &[],
            0,
        );
        let chain = DelegationChain { links: vec![d] };
        // Host trusts only real_issuer.
        let res = chain.validate(&[real_issuer.verifying_key().to_bytes()], "test.cap", 0);
        assert!(matches!(res, Err(CapError::BadChain(_))));
    }

    #[test]
    fn rejects_expired() {
        let mut rng = UnwrapErr(SysRng);
        let issuer = SigningKey::generate(&mut rng);
        let alice = SigningKey::generate(&mut rng);
        let d = Delegation::sign_new(
            &issuer,
            "test.cap",
            alice.verifying_key().to_bytes(),
            &[],
            100,
        );
        let chain = DelegationChain { links: vec![d] };
        // now = 200 → expired.
        let res = chain.validate(&[issuer.verifying_key().to_bytes()], "test.cap", 200);
        assert!(matches!(res, Err(CapError::BadChain(_))));
    }
}
