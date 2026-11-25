//! Security falsifiers: AES-256-GCM authenticated sealing (tamper-evident) + RBAC least privilege.

use competence_catalog::secure::{can, open, seal, Action, Role};

#[test]
fn seal_open_roundtrip_and_tamper_fails() {
    let key = [7u8; 32];
    let pt = b"competence record: secret code";
    let blob = seal(&key, pt).unwrap();
    assert_eq!(open(&key, &blob).unwrap(), pt);
    let mut tampered = blob.clone();
    let last = tampered.len() - 1;
    tampered[last] ^= 0xFF;
    assert!(open(&key, &tampered).is_err());
    assert!(open(&[9u8; 32], &blob).is_err()); // wrong key
}

#[test]
fn rbac_least_privilege() {
    assert!(can(Role::Viewer, Action::Lookup));
    assert!(!can(Role::Viewer, Action::Execute));
    assert!(!can(Role::Viewer, Action::Admit));
    assert!(can(Role::Agent, Action::Execute));
    assert!(!can(Role::Agent, Action::Admit));
    assert!(can(Role::Engineer, Action::Admit));
    assert!(!can(Role::Engineer, Action::Admin));
    assert!(can(Role::Admin, Action::Admin));
}