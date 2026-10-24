// Shared by spec/checker/build.rs and spec/checker-wasm/build.rs (include!):
// both crates sit two levels below the repository root, so the library path
// is the same for both.

/// The standard library is part of the language: every `.rce` module under
/// ../../stdlib/lib is compiled into the checker and resolves by `use`
/// (4.1). This writes OUT_DIR/stdlib_gen.rs: one (module path, source) per file,
/// and the library's manifest.
fn embed_stdlib() {
    let root = std::path::Path::new("../../stdlib/lib");
    let mut files = vec![];
    fn walk(d: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut es: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        es.sort();
        for p in es {
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "rce") {
                out.push(p);
            }
        }
    }
    walk(root, &mut files);
    println!("cargo:rerun-if-changed=../../stdlib/lib");
    let mut text = String::from("pub static LIB: &[(&str, &str)] = &[\n");
    for f in &files {
        println!("cargo:rerun-if-changed={}", f.display());
        let src = std::fs::read_to_string(f).unwrap_or_default();
        let name = src
            .lines()
            .find_map(|l| l.strip_prefix("module ").map(|n| n.trim().to_string()))
            .unwrap_or_else(|| f.strip_prefix(root).unwrap().with_extension("").to_string_lossy().replace('/', "."));
        let abs = std::fs::canonicalize(f).unwrap();
        text.push_str(&format!("    ({:?}, include_str!({:?})),\n", name, abs.to_string_lossy()));
    }
    text.push_str("];\n");
    // the library's own manifest (F.5.3), so a tool can name the library it
    // carries by hash and refuse a module that differs from its pin
    let man = root.join("MANIFEST.json");
    println!("cargo:rerun-if-changed={}", man.display());
    match std::fs::canonicalize(&man) {
        Ok(abs) => text.push_str(&format!("pub static MANIFEST: &str = include_str!({:?});\n", abs.to_string_lossy())),
        Err(_) => text.push_str("pub static MANIFEST: &str = \"\";\n"),
    }
    let out = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("stdlib_gen.rs");
    std::fs::write(out, text).unwrap();
}
