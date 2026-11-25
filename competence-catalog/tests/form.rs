//! Categorical form-layer falsifiers (Conal / DéjàVu / Hydra ingested): an intent
//! is recognized as an instantiation of a categorical form, the filling is mined
//! by replay, and the mined form crystallizes as verified competence.

use competence_catalog::form::{probe, FormMiner};
use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::workflow::WorkflowLibrary;
use serde_json::json;

const LIST_SHAPE: &str = "(&[i64],&Op)->List[i64]";

#[test]
fn mines_the_map_abs_form() {
    let miner = FormMiner::new(RuntimeBridge::new_inprocess());
    let probes = vec![
        probe(json!([[-1, 2, -3]]), "[1, 2, 3]"),
        probe(json!([[0, -5]]), "[0, 5]"),
    ];
    let m = miner.mine(LIST_SHAPE, &probes).expect("map-Abs covers the intent");
    assert_eq!(m.form, "map", "the functor form");
    assert_eq!(m.op, "Abs", "the mined filling");
}

#[test]
fn mines_filter_even_and_crystallizes() {
    let miner = FormMiner::new(RuntimeBridge::new_inprocess());
    let mut lib = WorkflowLibrary::new(RuntimeBridge::new_inprocess());
    let probes = vec![probe(json!([[1, 2, 3, 4]]), "[2, 4]")];
    let m = miner.mine_and_crystallize(&mut lib, LIST_SHAPE, &probes)
        .expect("filter-Even covers the intent");
    assert_eq!(m.form, "filter");
    assert_eq!(m.op, "Even");
    assert_eq!(lib.len(), 1, "the mined form crystallized as verified competence");
    // The crystallized workflow resolves a novel input by behavior.
    let novel = vec![probe(json!([[5, 6, 7, 8]]), "[6, 8]")];
    assert!(lib.resolve(LIST_SHAPE, &novel).is_some());
}

#[test]
fn uncoverable_intent_mines_nothing() {
    let miner = FormMiner::new(RuntimeBridge::new_inprocess());
    // No map/filter Op filling yields a constant non-empty list from these inputs.
    let probes = vec![probe(json!([[1, 2, 3]]), "[99, 99, 99]")];
    assert!(miner.mine(LIST_SHAPE, &probes).is_none());
}
