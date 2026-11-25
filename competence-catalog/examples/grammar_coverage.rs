//! Measured per-language grammar coverage from real tree-sitter node-types.json files.
//! Usage: cargo run --example grammar_coverage -- /path/nt-rust.json /path/nt-python.json …

fn main() {
    for path in std::env::args().skip(1) {
        let s = std::fs::read_to_string(&path).expect("read node-types");
        let v: serde_json::Value = serde_json::from_str(&s).expect("parse node-types");
        let (m, t, f) = competence_catalog::ingest::grammar_coverage(&v);
        println!("{path}: {m}/{t} = {f:.4}");
    }
}
