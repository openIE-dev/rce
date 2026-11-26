use competence_catalog::constellation::{self, UniverseEntry};
use competence_catalog::ingest;
use serde_json::json;

fn mini_nt() -> serde_json::Value {
    json!([
        {"type": "integer_literal", "named": true},
        {"type": "string_literal", "named": true},
        {"type": "identifier", "named": true},
        {"type": "if_statement", "named": true},
        {"type": "call_expression", "named": true},
        {"type": "binary_expression", "named": true},
        {"type": "comment", "named": true},
        {"type": ";", "named": false}
    ])
}

#[test]
fn universe_is_exhaustive() {
    let u = constellation::universe();
    assert!(u.len() > 500, "linguist universe should be exhaustive, got {}", u.len());
    assert!(u.iter().any(|e| e.name == "Rust"));
    assert!(u.iter().any(|e| e.name == "MoonBit"));
}

#[test]
fn structural_metrics_are_real_and_bounded() {
    let nt = mini_nt();
    let (n, mapped, c, h, red, surf) = ingest::structural_metrics(&nt);
    assert_eq!(n, 7, "only named nodes count");
    assert_eq!(mapped, 7, "mini grammar fully mapped");
    assert!(c > 1);
    assert!(h > 0.0 && h <= (c as f64).log2() + 1e-9, "entropy bounded by log2(C)");
    assert!((0.0..=1.0).contains(&red), "redundancy is a fraction");
    assert!(surf > 0.0, "surface-per-construct is positive (can be <1 when a node spans many constructs)");
}

#[test]
fn unmeasured_languages_have_no_invented_numbers() {
    let u = constellation::universe();
    let g = constellation::build(&u, &std::collections::BTreeMap::new());
    assert_eq!(g.measured_count, 0);
    for n in &g.nodes {
        if n.kind == "language" {
            assert!(n.grammar.is_none(), "no grammar profile without a real grammar");
            assert!(n.law.is_none(), "no law numbers without measurement");
        }
    }
}

#[test]
fn measured_node_carries_law_frame() {
    let mut u = vec![UniverseEntry {
        name: "Rust".into(), key: "rust".into(), kind: "programming".into(),
        group: "Rust".into(), extensions: vec![".rs".into()], aliases: vec![],
        language_id: None, color: "#dea584".into(), measured: true,
    }];
    let _ = &mut u;
    let mut m = std::collections::BTreeMap::new();
    m.insert("rust".to_string(), mini_nt());
    let g = constellation::build(&u, &m);
    let n = g.nodes.iter().find(|n| n.id == "rust").unwrap();
    assert!(n.measured);
    let law = n.law.as_ref().unwrap();
    assert_eq!(law.ratio, constellation::PROPOSER_UJ / constellation::LOOKUP_UJ);
    assert!(law.statement.contains("I = H_task - H_corpus -> 0"));
    assert!(n.grammar.as_ref().unwrap().coverage > 0.99);
}

#[test]
fn influence_and_family_edges_exist() {
    let u = constellation::universe();
    let g = constellation::build(&u, &std::collections::BTreeMap::new());
    assert!(g.edges.iter().any(|e| e.kind == "influence" && e.from == "c" && e.to == "cplusplus"));
    assert!(g.nodes.iter().any(|n| n.id == "cplusplus") && g.nodes.iter().any(|n| n.id == "csharp"));
    assert!(g.edges.iter().any(|e| e.kind == "family"));
    assert!(g.nodes.iter().any(|n| n.kind == "family"));
    assert!(g.nodes.iter().any(|n| n.kind == "paradigm"));
}
