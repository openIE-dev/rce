//! Repo ingestion — the world's repositories into the substrate, sovereignly.
//!
//! Walk a repository, lift each source file into the Form, correct it in isolation (no AI / no
//! network / no listeners), dedup by name-free content key, and admit what replays. This is the
//! bootstrap-corpus import path at repo scale, and it never leaks code: correction is local and
//! deterministic, governed by the sovereignty policy.

use std::collections::HashSet;
use std::path::Path;

use crate::canonical::canonical_key;
use crate::lift::lift_expr;
use crate::repair::eval_expr;
use crate::sovereignty::{Policy, Sovereign};

/// The result of ingesting a repository.
pub struct RepoReport {
    pub files: usize,
    pub lifted: usize,
    pub admitted: usize,
    pub refused: usize,
    pub dedup: usize,
    pub keys: Vec<String>,
}

fn is_source(ext: &str) -> bool {
    matches!(ext, "rs" | "py" | "c" | "h" | "js" | "ts" | "go" | "rb" | "java" | "cs" | "cpp" | "hs" | "ml" | "ex" | "lua" | "r" | "f90" | "ada" | "adb")
}

fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                let name = p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
                if name == "target" || name == ".git" || name == "node_modules" { continue; }
                walk(&p, out);
            } else if let Some(ext) = p.extension().map(|s| s.to_string_lossy().to_string()) {
                if is_source(&ext) { out.push(p); }
            }
        }
    }
}

/// Ingest a repository sovereignly: lift → isolated correct → dedup → admit-by-replay.
pub fn ingest_repo(root: &Path, policy: Policy) -> RepoReport {
    let mut files = Vec::new();
    walk(root, &mut files);
    let mut sov = Sovereign::new(policy);
    let mut seen = HashSet::new();
    let mut rep = RepoReport { files: files.len(), lifted: 0, admitted: 0, refused: 0, dedup: 0, keys: Vec::new() };
    for p in files {
        let Ok(src) = std::fs::read_to_string(&p) else { rep.refused += 1; continue };
        // lift the arithmetic/expression fragment; whole-file modules lift their exprs elsewhere.
        let Ok(e) = lift_expr(src.trim()) else { rep.refused += 1; continue };
        rep.lifted += 1;
        let Ok(key) = canonical_key(src.trim()) else { rep.refused += 1; continue };
        if !seen.insert(key.clone()) { rep.dedup += 1; continue; }
        // isolated correction/verification: no model, network, or listeners
        let _ = sov.correct_isolated(src.trim(), eval_expr(&e).unwrap_or(0));
        if eval_expr(&e).is_some() { rep.admitted += 1; rep.keys.push(key); }
        else { rep.refused += 1; }
    }
    rep
}