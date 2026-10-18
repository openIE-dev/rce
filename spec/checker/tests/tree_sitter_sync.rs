//! The tree-sitter grammar (spec/tree-sitter) is kept in sync with the EBNF
//! and the Rust parser: for every `.rce` file under spec/, the tree-sitter
//! CLI parses it without error exactly when `parse::parse` accepts it.
//! There is one language: both sides reserve the same single word list (2.4)
//! and ignore a `# rce-version:` line, which is only a release label (2.7).
//! The CLI compiles the generated C parser itself; this crate stays pure
//! Rust. When the CLI is not installed the test prints a notice and
//! passes, except under CI (`CI` set), where a missing CLI fails it.
use std::path::{Path, PathBuf};
use std::process::Command;

fn rce_files(d: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(d).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            if !p.ends_with("target") && !p.ends_with("node_modules") {
                rce_files(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "rce") {
            out.push(p);
        }
    }
}

#[test]
fn tree_sitter_agrees_with_rust_parser() {
    let spec = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let ts = spec.join("tree-sitter");
    if Command::new("tree-sitter").arg("--version").output().is_err() {
        assert!(std::env::var_os("CI").is_none(), "tree-sitter CLI required under CI");
        println!("tree-sitter CLI not installed: skipped");
        return;
    }
    let mut files = vec![];
    rce_files(&spec, &mut files);
    files.sort();
    assert!(files.len() > 150, "{}", files.len());
    let mut bad = vec![];
    let mut n_ok = 0;
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        // the scanner reserves the one list of words in every module
        let rust_ok = match rce_spec_check::parse::lex(&src) {
            Ok(t) => {
                let mut p = rce_spec_check::parse::Parser::new(t);
                p.module().is_ok() && rce_spec_check::v09::stray_words(&p.t, &p.kw_used).is_ok()
            }
            _ => false,
        };
        let out = Command::new("tree-sitter")
            .current_dir(&ts)
            .args(["parse", "-q"])
            .arg(f)
            .output()
            .unwrap();
        let ts_ok = out.status.success();
        if rust_ok != ts_ok {
            bad.push(format!("{}: rust {rust_ok}, tree-sitter {ts_ok}", f.display()));
        } else {
            n_ok += 1;
        }
    }
    println!("tree-sitter and Rust parser agree on {n_ok}/{} files", files.len());
    assert!(bad.is_empty(), "{} disagreements:\n{}", bad.len(), bad.join("\n"));
}
