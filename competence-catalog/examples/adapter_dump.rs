//! Dump generated editor-ecosystem adapters (adapter_gen) for a language as JSON,
//! so RCEdit plugins can be materialized per ecosystem from the one competence server.
use competence_catalog::adapter_gen;

fn main() {
    let lang = std::env::args().nth(1).unwrap_or_else(|| "rust".into());
    let arts = adapter_gen::generate_all(&lang).expect("language lut");
    println!("{}", serde_json::to_string_pretty(&arts).unwrap());
}
