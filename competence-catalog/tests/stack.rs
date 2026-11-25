//! The all-encompassing vertical stack: an intent flows down the layers — mine a
//! categorical form, crystallize it, then reuse it as competence — and the build
//! layer is content-addressed. One substrate, every layer, one discipline.

use competence_catalog::stack::{Layer, Stack};
use competence_catalog::workflow::probe;
use serde_json::json;

const LIST_SHAPE: &str = "(&[i64],&Op)->List[i64]";

#[test]
fn intent_flows_form_then_competence() {
    let mut s = Stack::new();
    let probes = vec![probe(json!([[-1, 2, -3]]), "[1, 2, 3]")];

    // First resolve: no competence yet → mine the map-Abs form.
    let r1 = s.resolve(LIST_SHAPE, &probes).expect("the form covers the intent");
    assert_eq!(r1.layer, Layer::Form);
    assert_eq!(r1.result, "[1, 2, 3]");

    // Second resolve: the mined form is now crystallized competence → reuse.
    let r2 = s.resolve(LIST_SHAPE, &probes).expect("competence covers it");
    assert_eq!(r2.layer, Layer::Competence, "the stack reused crystallized competence");
    assert_eq!(r2.result, "[1, 2, 3]");
    assert!(!s.hindsight.is_empty(), "the stack learned");
}

#[test]
fn build_layer_is_content_addressed_and_incremental() {
    let mut s = Stack::new();
    s.build("a + b");
    s.build("b + a"); // accidental representation change → hit
    assert_eq!(s.build.hits, 1);
    assert_eq!(s.build.misses, 1);
}

#[test]
fn novel_intent_mines_nothing() {
    let mut s = Stack::new();
    let probes = vec![probe(json!([[1, 2, 3]]), "[99, 99, 99]")];
    assert!(s.resolve(LIST_SHAPE, &probes).is_none());
}
