//! Dump a generated tree-sitter highlights.scm from a real node-types.json (construct IR).
//! Usage: cargo run --example queries_dump -- /tmp/nt-rust.json
use competence_catalog::adapter_gen::tree_sitter_queries;

fn main() {
    let path = std::env::args().nth(1).expect("node-types path");
    let nt: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    print!("{}", tree_sitter_queries(&nt));
}
