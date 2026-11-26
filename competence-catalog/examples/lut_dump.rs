//! Dump the emission LUT templates (core constructs) for every built-in language as JSON,
//! so the interactive constellation can translate a Form between languages in the browser.
use competence_catalog::lut_emit::builtin_luts;
use competence_catalog::structure::CONSTRUCTS;

fn main() {
    let core: Vec<&str> = CONSTRUCTS.iter().copied().take(49).collect();
    let mut out = serde_json::Map::new();
    for l in builtin_luts() {
        let mut m = serde_json::Map::new();
        for c in &core {
            if let Some(t) = l.template(c) {
                m.insert(c.to_string(), serde_json::Value::String(t.to_string()));
            }
        }
        out.insert(l.name.clone(), serde_json::Value::Object(m));
    }
    println!("{}", serde_json::Value::Object(out));
}
