//! Sovereignty + repo-ingestion falsifiers: correction isolated from AI/listeners; sovereign
//! repo import with dedup and admit-by-replay.

use competence_catalog::repo::ingest_repo;
use competence_catalog::sovereignty::{Policy, Sovereign};

#[test]
fn default_policy_is_fully_sovereign() {
    let mut s = Sovereign::new(Policy::default());
    assert!(s.assert_isolated().is_ok());
    assert!(s.register_listener("telemetry").is_err()); // listeners blocked
    let out = s.correct_isolated("2 + 3", 6).unwrap();
    assert!(out.is_some());
    assert!(s.audit.iter().any(|a| a.contains("isolated")));
}

#[test]
fn permissive_policy_is_not_sovereign() {
    let mut p = Policy::default();
    p.allow_model = true;
    let s = Sovereign::new(p);
    assert!(s.assert_isolated().is_err());
}

#[test]
fn repo_ingestion_sovereign_dedup_admit() {
    let dir = std::env::temp_dir().join(format!("rce-repo-test-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(dir.join("a.py"), "2 + 3").unwrap();
    std::fs::write(dir.join("b.rs"), "2 + 3").unwrap(); // same behavior -> dedup
    std::fs::write(dir.join("c.py"), "2 * 3").unwrap();
    std::fs::write(dir.join("d.py"), "this is not arithmetic").unwrap(); // refused
    let rep = ingest_repo(&dir, Policy::default());
    assert_eq!(rep.files, 4);
    assert_eq!(rep.lifted, 3);
    assert_eq!(rep.dedup, 1);
    assert_eq!(rep.admitted, 2);
    assert_eq!(rep.refused, 1);
    let _ = std::fs::remove_dir_all(&dir);
}