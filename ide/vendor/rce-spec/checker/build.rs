#[cfg(feature = "tree-sitter")]
fn main() {
    let dir = std::path::Path::new("../tree-sitter/src");
    cc::Build::new()
        .include(dir)
        .file(dir.join("parser.c"))
        .file(dir.join("scanner.c"))
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-but-set-variable")
        .flag_if_supported("-Wno-trigraphs")
        .compile("tree-sitter-rce");
    println!("cargo:rerun-if-changed=../tree-sitter/src/parser.c");
    println!("cargo:rerun-if-changed=../tree-sitter/src/scanner.c");
}

#[cfg(not(feature = "tree-sitter"))]
fn main() {}
