//! Crystallized-workflow falsifiers (AWM/ReasoningBank ingested): multi-step
//! competence is admitted only by replay, indexed by shape, reused by behavior;
//! a non-reproducing workflow is refused at the gate.

use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::workflow::{pipeline_program, probe, Workflow, WorkflowLibrary};
use serde_json::json;

fn sqrtabs() -> Workflow {
    let ops = vec![
        ("math".to_string(), "fabs".to_string(), "(f64)->f64".to_string()),
        ("math".to_string(), "sqrt".to_string(), "(f64)->f64".to_string()),
    ];
    Workflow {
        name: "sqrtabs".into(),
        shape: "(f64)->f64".into(),
        ops: vec!["math.fabs".into(), "math.sqrt".into()],
        program: pipeline_program("sqrtabs", "f64", &ops),
    }
}

fn spec() -> Vec<competence_catalog::semantic::Probe> {
    vec![probe(json!([-4.0]), "2"), probe(json!([9.0]), "3"), probe(json!([0.0]), "0")]
}

#[test]
fn workflow_crystallizes_by_replay_and_resolves() {
    let mut lib = WorkflowLibrary::new(RuntimeBridge::new_inprocess());
    assert!(lib.is_empty());
    assert!(lib.crystallize(sqrtabs(), &spec()), "a reproducing workflow is admitted");
    assert_eq!(lib.len(), 1);
    // A novel intent spec (no name) resolves to the crystallized workflow.
    let hit = lib.resolve("(f64)->f64", &spec()).expect("workflow covers the spec");
    assert_eq!(hit.name, "sqrtabs");
    assert_eq!(hit.ops.len(), 2, "multi-step competence");
}

#[test]
fn non_reproducing_workflow_is_refused() {
    let mut lib = WorkflowLibrary::new(RuntimeBridge::new_inprocess());
    // fabs alone does not reproduce sqrt(|x|): fabs(-4)=4, not 2.
    let ops = vec![("math".to_string(), "fabs".to_string(), "(f64)->f64".to_string())];
    let bad = Workflow {
        name: "bad".into(),
        shape: "(f64)->f64".into(),
        ops: vec!["math.fabs".into()],
        program: pipeline_program("bad", "f64", &ops),
    };
    assert!(!lib.crystallize(bad, &spec()), "a non-reproducing workflow is refused");
    assert_eq!(lib.len(), 0);
}

#[test]
fn workflow_reuse_is_name_free() {
    let mut lib = WorkflowLibrary::new(RuntimeBridge::new_inprocess());
    lib.crystallize(sqrtabs(), &spec());
    // Different probe inputs, same behavior → still resolves (behavior, not name).
    let novel = vec![probe(json!([-16.0]), "4"), probe(json!([25.0]), "5")];
    assert!(lib.resolve("(f64)->f64", &novel).is_some());
}
