use competence_catalog::ingest::grammar_coverage;
use serde_json::json;

#[test]
fn coverage_counts_named_node_types_only() {
    let nt = json!([
        {"type": "identifier", "named": true},
        {"type": "binary_expression", "named": true},
        {"type": "comment", "named": true},
        {"type": "(", "named": false}
    ]);
    let (m, t, f) = grammar_coverage(&nt);
    assert_eq!((m, t), (2, 3));
    assert!((f - 2.0 / 3.0).abs() < 1e-12);
}

#[test]
fn empty_grammar_is_zero_not_nan() {
    let (m, t, f) = grammar_coverage(&json!([]));
    assert_eq!((m, t), (0, 0));
    assert_eq!(f, 0.0);
}
