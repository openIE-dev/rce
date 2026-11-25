//! Completion-threshold falsifiers: the gated dependency order that avoids the
//! chicken-and-egg. A dependent is workable only after its prerequisites are
//! fully completed to threshold; an unmeasured prerequisite is never assumed
//! complete; independent stages form the parallel frontier.

use competence_catalog::threshold::{
    acyclic, blockers, complete, frontier, order_report, rce_dag, unlocked, Direction,
    Measurements, Stage, Threshold,
};

#[test]
fn the_egg_is_workable_first() {
    let dag = rce_dag();
    let m = Measurements::new();
    // With nothing measured, only the egg (RCE-SPEC, no prerequisites) is workable.
    assert!(unlocked(&dag, &m, "RCE-SPEC"));
    assert_eq!(frontier(&dag, &m), vec!["RCE-SPEC"]);
}

#[test]
fn dependent_is_blocked_until_prereqs_complete() {
    let dag = rce_dag();
    let m = Measurements::new();
    // JEPA cannot be really worked on before the engine + substrate exist.
    assert!(!unlocked(&dag, &m, "RCE-JEPA"));
    assert_eq!(blockers(&dag, &m, "RCE-JEPA"), vec!["RCE-Engine", "RCE"]);
}

#[test]
fn own_metric_met_does_not_bypass_prereqs() {
    let dag = rce_dag();
    let mut m = Measurements::new();
    // Even if JEPA's own corpus threshold is met, it is not workable while its
    // prerequisites are incomplete — the cart-before-horse guard.
    m.set("verified_corpus_size", 1000.0);
    assert!(complete(&dag, &m).contains("RCE-JEPA"));
    assert!(!unlocked(&dag, &m, "RCE-JEPA"));
}

#[test]
fn completing_prereqs_unlocks_the_dependent() {
    let dag = rce_dag();
    let mut m = Measurements::new();
    m.set("spec_conformance", 1.0)
        .set("gate_tests", 1.0)
        .set("canonical_stability", 1.0)
        .set("emit_replay_identity", 1.0)
        .set("executor_coverage", 0.92);
    // Now the engine is complete and JEPA's prerequisites are done.
    assert!(unlocked(&dag, &m, "RCE-JEPA"));
    assert!(blockers(&dag, &m, "RCE-JEPA").is_empty());
}

#[test]
fn independent_stages_form_the_parallel_frontier() {
    let dag = rce_dag();
    let mut m = Measurements::new();
    m.set("spec_conformance", 1.0).set("gate_tests", 1.0);
    // Once Form + gate are at threshold, substrate and compile proceed in parallel;
    // the engine still waits on them.
    let f = frontier(&dag, &m);
    assert!(f.contains(&"RCE"));
    assert!(f.contains(&"RCE-Compile"));
    assert!(!f.contains(&"RCE-Engine"));
    assert!(!unlocked(&dag, &m, "RCE-Engine"));
}

#[test]
fn unmeasured_prereq_is_never_assumed_complete() {
    let dag = rce_dag();
    let mut m = Measurements::new();
    m.set("spec_conformance", 1.0); // gate_tests absent
    assert!(!complete(&dag, &m).contains("RCE-VC"));
    assert!(!unlocked(&dag, &m, "RCE")); // needs the gate, which is unmeasured
}

#[test]
fn all_complete_means_no_frontier() {
    let dag = rce_dag();
    let mut m = Measurements::new();
    m.set("spec_conformance", 1.0)
        .set("gate_tests", 1.0)
        .set("canonical_stability", 1.0)
        .set("emit_replay_identity", 1.0)
        .set("executor_coverage", 0.92)
        .set("verified_corpus_size", 100.0);
    assert_eq!(complete(&dag, &m).len(), 6);
    assert!(frontier(&dag, &m).is_empty());
    assert!(order_report(&dag, &m).iter().all(|(_, done, open, _)| *done && *open));
}

#[test]
fn the_order_is_acyclic() {
    assert!(acyclic(&rce_dag()));
}

#[test]
fn at_most_threshold_guards_a_mismatch_count() {
    let dag = vec![Stage {
        id: "emit",
        requires: &[],
        threshold: Threshold { metric: "replay_mismatches", required: 0.0, direction: Direction::AtMost },
    }];
    let mut m = Measurements::new();
    m.set("replay_mismatches", 0.0);
    assert!(complete(&dag, &m).contains("emit"));
    m.set("replay_mismatches", 1.0);
    assert!(!complete(&dag, &m).contains("emit"));
}