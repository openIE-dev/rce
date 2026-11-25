//! The falsifier: as the catalog grows, reuse rate rises and cost per competent
//! act falls toward the G0 interpreter floor. Plus: a hit costs zero model calls,
//! a miss escalates and crystallizes, and the gate refuses negative transfer.

use competence_catalog::{input_key, Catalog, CompetenceRecord, Lookup, Provenance};
use rcevc_core::Reason;

fn lcg(seed: &mut u64) -> usize {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 33) as usize
}

#[test]
fn hit_costs_zero_model_calls() {
    let mut cat = Catalog::new().unwrap();
    let (res, cost, hit) = cat.resolve("hello", "upper", "probe");
    assert_eq!(res, "HELLO");
    assert!(!hit, "first encounter must miss");
    assert!(cost >= 5_000_000, "a miss pays the proposer: {cost}");

    let (res2, cost2, hit2) = cat.resolve("hello", "upper", "probe");
    assert_eq!(res2, "HELLO");
    assert!(hit2, "second encounter must hit");
    assert_eq!(cost2, cat.costs().g0, "a hit costs the G0 floor, no model");
    assert_eq!(cat.meter().proposer_calls, 0, "no conflict proposer needed");
}

#[test]
fn gate_refuses_negative_transfer() {
    let mut cat = Catalog::new().unwrap();
    let bad = CompetenceRecord {
        input_key: input_key("x"),
        handler_kind: "upper".into(),
        probe_input: "a".into(),
        probe_output: "WRONG".into(), // handler("a") = "A", not "WRONG"
        result: "X".into(),
        cost_uj: 1,
        provenance: Provenance::Modelled,
    };
    let out = cat.admit(bad);
    assert!(!out.is_accepted(), "a failing spectest must be refused");
    assert!(matches!(out.reason(), Some(Reason::Gate { .. })));
    assert!(matches!(cat.lookup("x"), Lookup::Miss), "refused record not indexed");
    assert_eq!(cat.meter().catalog_size, 0);
}

#[test]
fn content_addressing_is_name_free() {
    assert_eq!(input_key("abc"), input_key("abc"));
    assert_ne!(input_key("abc"), input_key("abd"));
    assert_eq!(input_key("abc").len(), 64);
}

#[test]
fn falsifier_cost_per_act_falls_as_corpus_grows() {
    let mut cat = Catalog::new().unwrap();
    let inputs: Vec<String> = (0..8).map(|i| format!("task-{i}")).collect();
    let mut seed = 0x1234_5678_9abc_def0u64;

    let mut first_half_cost = 0u64;
    let acts = 240u64;
    for act in 0..acts {
        let idx = lcg(&mut seed) % inputs.len();
        let (_, cost, _) = cat.resolve(&inputs[idx], "upper", "probe");
        if act < acts / 2 {
            first_half_cost += cost;
        }
    }
    let m = cat.meter();
    let second_half_cost = m.workload_est_uj - first_half_cost;
    let first_cpa = first_half_cost as f64 / (acts / 2) as f64;
    let second_cpa = second_half_cost as f64 / (acts / 2) as f64;

    // Each distinct task misses exactly once, then is owned.
    assert_eq!(m.misses, m.catalog_size as u64);
    assert!(m.catalog_size >= 6, "corpus should cover the task family");
    assert!(m.reuse_rate > 0.9, "reuse rate must rise: {}", m.reuse_rate);
    // The declining curve: the second half is far cheaper than the first.
    assert!(
        second_cpa < first_cpa / 50.0,
        "cost per act must fall: first {first_cpa} µJ, second {second_cpa} µJ"
    );
    // It converges toward the G0 interpreter floor.
    assert!(second_cpa < 10.0, "second half near the floor: {second_cpa} µJ");
}

#[test]
fn admission_is_hash_chained_and_replayable() {
    let mut cat = Catalog::new().unwrap();
    for i in 0..5 {
        cat.resolve(&format!("t{i}"), "reverse", "probe");
    }
    let log = cat.repo().log();
    assert!(log.len() >= 5, "every admission is a log entry");
    let genesis = cat.repo().genesis();
    assert!(
        rcevc_core::verify_chain(&genesis, log).is_ok(),
        "the log must verify as a hash chain"
    );
}