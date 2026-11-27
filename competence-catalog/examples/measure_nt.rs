//! Honest per-grammar coverage probe. Usage: measure_nt <key> <node-types.json> [...]
//! Prints `key mapped total frac` per pair using ingest::grammar_coverage (real named nodes).

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < a.len() {
        let (key, path) = (&a[i], &a[i + 1]);
        i += 2;
        match std::fs::read_to_string(path) {
            Ok(s) => {
                let v: serde_json::Value = serde_json::from_str(&s).expect("node-types json");
                let (m, t, f) = competence_catalog::ingest::grammar_coverage(&v);
                println!("{key} {m} {t} {f:.4}");
            }
            Err(e) => println!("{key} ERR {e}"),
        }
    }
}
