use competence_catalog::adapter_gen::tree_sitter_queries;
use competence_catalog::lowering_trace::{compare, trace};
use serde_json::json;

#[test]
fn invented_decisions_flagged_across_languages() {
    let c = compare("rust", "python");
    assert!(c.len() >= 5, "calling conv/abi/regs/memory/ownership/width/overflow diverge");
    assert!(c.iter().any(|(cat, _, _)| *cat == "ownership"));
    assert!(c.iter().any(|(cat, _, _)| *cat == "overflow"));
    assert_eq!(trace("rust").len(), 7);
}

#[test]
fn tree_sitter_queries_generated_from_construct_ir() {
    let nt = json!([
        {"type":"integer_literal","named":true},
        {"type":"string_literal","named":true},
        {"type":"identifier","named":true},
        {"type":"if_statement","named":true},
        {"type":"call_expression","named":true},
        {"type":"comment","named":true},
        {"type":";","named":false}
    ]);
    let q = tree_sitter_queries(&nt);
    assert!(q.contains("(integer_literal) @number"));
    assert!(q.contains("(string_literal) @string"));
    assert!(q.contains("(comment) @comment"));
    assert!(q.contains("(call_expression) @function"));
    assert!(!q.contains("(;)"), "anonymous nodes excluded");
}
