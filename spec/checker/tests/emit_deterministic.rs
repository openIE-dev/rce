//! Deterministic projection (spec 10.5; research paper 2.3): two emits of
//! one program to one target give the same bytes. Each emit is its own
//! `rce` process, so state that differs between processes (hash-map
//! iteration order, addresses, time) would show here. Every example of
//! spec/examples that `rce emit` accepts is emitted twice to every target:
//! the Form, the core, and the four program emitters.
use std::path::Path;
use std::process::Command;

const TARGETS: [&str; 6] = ["form", "core", "rust", "python", "js", "scala"];

fn emit(target: &str, file: &Path) -> Option<Vec<u8>> {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["emit", target]).arg(file).output().unwrap();
    o.status.success().then_some(o.stdout)
}

#[test]
fn two_emits_give_the_same_bytes() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    files.sort();
    let (mut compared, mut differ) = (0, Vec::new());
    for f in &files {
        for t in TARGETS {
            let Some(a) = emit(t, f) else { continue };
            let b = emit(t, f).expect("the second emit of a program the first accepted");
            compared += 1;
            if a != b {
                differ.push(format!("{} {t}", f.display()));
            }
        }
    }
    assert!(compared >= files.len(), "only {compared} emits compared over {} examples", files.len());
    assert!(differ.is_empty(), "two emits differ: {differ:?}");
}
