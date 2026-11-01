fn git_commit() -> String {
    let out = std::process::Command::new("git").args(["rev-parse", "--short=12", "HEAD"]).output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}

include!("build_stdlib.rs");

fn main() {
    // CC-4: emitted code names the checker commit that printed it.
    println!("cargo:rustc-env=RCE_GIT_COMMIT={}", git_commit());
    embed_stdlib();
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads");
    emit_targets();
    tls_cfg();
    tree_sitter();
}

/// `rce_tls`: the target has the TLS crates (Cargo.toml lists rustls, rcgen
/// and ring for every target except wasm32-unknown-unknown). Without it the
/// TLS host commands report that TLS is not built for the target.
fn tls_cfg() {
    println!("cargo:rustc-check-cfg=cfg(rce_tls)");
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if !(arch == "wasm32" && os == "unknown") {
        println!("cargo:rustc-cfg=rce_tls");
    }
}

/// The emit targets that `tests/emit_targets.txt` names (spec 10.5,
/// tests/common/targets.rs), each passed to the tests as
/// `--cfg rce_emit="WORD"`: a test that runs one target alone is marked
/// ignored, never passed, when the file does not name its target. Without
/// the file no target is passed: the build goes on with a warning, and the
/// tests that read the file fail when they run.
fn emit_targets() {
    const FILE: &str = "tests/emit_targets.txt";
    const ALL: [&str; 4] = ["rust", "python", "js", "scala"];
    println!("cargo:rerun-if-changed={FILE}");
    println!("cargo:rustc-check-cfg=cfg(rce_emit, values(\"rust\", \"python\", \"js\", \"scala\"))");
    let text = match std::fs::read_to_string(FILE) {
        Ok(t) => t,
        Err(e) => {
            println!("cargo:warning={FILE}: {e}; no emit target is covered");
            return;
        }
    };
    for l in text.lines() {
        let w = l.split('#').next().unwrap_or("").trim();
        if w.is_empty() {
            continue;
        }
        assert!(ALL.contains(&w), "{FILE}: `{w}` is not an emit target (rust, python, js, scala)");
        println!("cargo:rustc-cfg=rce_emit=\"{w}\"");
    }
}

#[cfg(feature = "tree-sitter")]
fn tree_sitter() {
    let dir = std::path::Path::new("../../tree-sitter/src");
    cc::Build::new()
        .include(dir)
        .file(dir.join("parser.c"))
        .file(dir.join("scanner.c"))
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-Wno-trigraphs")
        .compile("tree-sitter-rce");
    println!("cargo:rerun-if-changed=../../tree-sitter/src/parser.c");
    println!("cargo:rerun-if-changed=../../tree-sitter/src/scanner.c");
}

#[cfg(not(feature = "tree-sitter"))]
fn tree_sitter() {}
