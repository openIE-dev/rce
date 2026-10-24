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
    tree_sitter();
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
