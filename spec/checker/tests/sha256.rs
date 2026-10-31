//! 5.15 `sha256`: SHA-256 (FIPS 180-4) of the UTF-8 bytes of a text as 64
//! lowercase hex digits. The reference evaluator and the emitted Rust,
//! Python and JavaScript runtimes print the same digests, checked against
//! the FIPS 180-4 / NIST CSRC example vectors and against a sha2 crate
//! digest for multi-block and non-ASCII inputs.
use rce_spec_check::cli;
use sha2::{Digest, Sha256};
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

const VECTORS: [(&str, &str); 3] = [
    ("", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    (
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    ),
];

fn inputs() -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = VECTORS.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    for s in ["héllo wörld ✓", &"x".repeat(55), &"y".repeat(56), &"z".repeat(64), &"q".repeat(200)] {
        let h: String = Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        v.push((s.to_string(), h));
    }
    v
}

fn prog() -> String {
    let mut s = "fn main()\n".to_string();
    for (t, _) in inputs() {
        s += &format!("    emit \"{t}\".sha256()\n");
    }
    s
}

fn want() -> Vec<String> {
    inputs().into_iter().map(|(_, h)| h).collect()
}

fn lines(out: &str) -> Vec<String> {
    let re = regex::Regex::new(r"\b[0-9a-f]{64}\b").unwrap();
    out.lines().filter_map(|l| re.find(l).map(|m| m.as_str().to_string())).collect()
}

fn dir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rce-sha256-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn reference_evaluator_matches_fips_vectors() {
    let f = dir().join("ref.rce");
    std::fs::write(&f, prog()).unwrap();
    let (mut out, mut err) = (vec![], vec![]);
    let a = vec!["run".to_string(), f.to_string_lossy().into_owned()];
    assert_eq!(cli::main(&a, &mut &b""[..], &mut out, &mut err), 0, "{}", String::from_utf8_lossy(&err));
    assert_eq!(lines(&String::from_utf8(out).unwrap()), want());
}

#[test]
fn every_emitted_runtime_matches_the_reference() {
    let d = dir();
    let f = d.join("emit.rce");
    std::fs::write(&f, prog()).unwrap();
    for (target, ext, tool) in [("rust", "rs", "rustc"), ("python", "py", "python3"), ("js", "mjs", "node")] {
        if !targets::covers(target, tool) {
            continue;
        }
        let (mut out, mut err) = (vec![], vec![]);
        let a = vec!["emit".to_string(), target.to_string(), f.to_string_lossy().into_owned()];
        assert_eq!(cli::main(&a, &mut &b""[..], &mut out, &mut err), 0, "{}", String::from_utf8_lossy(&err));
        let p = d.join(format!("s_{target}.{ext}"));
        std::fs::write(&p, out).unwrap();
        let o = if target == "rust" {
            let exe = d.join("s_rust");
            let c = Command::new("rustc").args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&p).output().unwrap();
            assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
            Command::new(&exe).arg("run").output().unwrap()
        } else {
            Command::new(tool).arg(&p).arg("run").output().unwrap()
        };
        let s = String::from_utf8_lossy(&o.stdout).into_owned();
        assert_eq!(lines(&s), want(), "{target}: {s}{}", String::from_utf8_lossy(&o.stderr));
    }
}

#[test]
fn sha256_is_unary_and_typed_text() {
    let f = dir().join("bad.rce");
    std::fs::write(&f, "fn main()\n    let n: i64 = \"a\".sha256()\n    emit n\n").unwrap();
    let (mut out, mut err) = (vec![], vec![]);
    let a = vec!["check".to_string(), f.to_string_lossy().into_owned()];
    assert_ne!(cli::main(&a, &mut &b""[..], &mut out, &mut err), 0);
}
