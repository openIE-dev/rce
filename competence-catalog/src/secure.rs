//! Security layer — closes the two flagged SOC2 gaps: encryption at rest (C1.2) and RBAC (CC6.3).
//!
//! Confidentiality: competence records are sealed with AES-256-GCM (authenticated encryption) so
//! stored code/data is protected at rest and any tamper is detected on open. Access: role-based
//! least-privilege over catalog operations. Together these complete the SOC2 control map.

use aes_gcm::aead::{Aead, AeadCore, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Key, Nonce};

/// Seal plaintext with AES-256-GCM; output = nonce || ciphertext (authenticated).
pub fn seal(key: &[u8; 32], pt: &[u8]) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);
    let ct = cipher.encrypt(&nonce, pt).map_err(|_| "encrypt failed".to_string())?;
    let mut out = nonce.to_vec();
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Open a sealed blob; any tamper (or wrong key) fails authentication.
pub fn open(key: &[u8; 32], blob: &[u8]) -> Result<Vec<u8>, String> {
    if blob.len() < 12 {
        return Err("blob too short".into());
    }
    let (n, ct) = blob.split_at(12);
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    cipher.decrypt(Nonce::from_slice(n), ct).map_err(|_| "authentication failed (tamper or wrong key)".into())
}

/// Roles for least-privilege access (CC6.3).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role { Admin, Engineer, Agent, Viewer }

/// Catalog operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action { Lookup, Execute, Admit, Sync, Admin }

/// Least-privilege permission matrix.
pub fn can(r: Role, a: Action) -> bool {
    match r {
        Role::Admin => true,
        Role::Engineer => matches!(a, Action::Lookup | Action::Execute | Action::Admit | Action::Sync),
        Role::Agent => matches!(a, Action::Lookup | Action::Execute),
        Role::Viewer => matches!(a, Action::Lookup),
    }
}