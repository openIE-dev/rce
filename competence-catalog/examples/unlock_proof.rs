//! Proof receipts for the unlock: the competence-backed editor loop (Rust) + generated adapters
//! + synthesized native languages. Prints a JSON receipt for the constellation "Unlock" view.
use competence_catalog::editor_loop::{EditorLoop, EditOutcome};
use competence_catalog::lang_synth;
use competence_catalog::langpredict::FABRICS;
use std::collections::BTreeMap;

fn main() {
    let mut loop_ = EditorLoop::rust();
    let good = "let x = 1; return x + 1;";
    let probes: Vec<(BTreeMap<String, i128>, i128)> = vec![(BTreeMap::new(), 2)];
    let bad = "return 3;";
    let _ = loop_.edit(bad, &probes);
    let key = match loop_.edit(good, &probes) { EditOutcome::Served { key, .. } => key, _ => unreachable!() };
    for _ in 0..50 { loop_.edit(good, &probes); }
    let _ = loop_.serve(&key, &probes);
    let synth: Vec<serde_json::Value> = FABRICS.iter().map(|f| {
        let s = lang_synth::synthesize(f);
        serde_json::json!({
            "name": s.name, "fabric": s.fabric, "paradigm": s.paradigm,
            "constructs": s.constructs.len(), "coverage": s.coverage(),
            "sample_replays": s.sample_replays(),
            "native": f.native_constructs,
        })
    }).collect();
    let receipt = serde_json::json!({
        "editor_loop": {
            "target": "rust",
            "edits": loop_.stats.edits,
            "caught_at_edit": loop_.stats.caught_at_edit,
            "admitted": loop_.stats.admitted,
            "served_hits": loop_.stats.served_hits,
            "escaped_errors": loop_.stats.escaped_errors,
            "debug_sessions": loop_.stats.debug_sessions,
            "cost_per_act_uj": loop_.cost_per_act(),
        },
        "adapters_generated_for": "rust",
        "synthesized_languages": synth,
    });
    println!("{}", serde_json::to_string(&receipt).unwrap());
}
