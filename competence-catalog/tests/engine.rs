//! The gear-ladder engine falsifiers: competence precedes models, every gear
//! above G0 passes the replay gate, and cost per act decays as reuse grows.

use competence_catalog::engine::{probe, Engine, FixedProposer, NoProposer, Rule, GEAR_COST};
use competence_catalog::runtime::RuntimeBridge;
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("time.civil.is_leap".into(), "time.civil".into(), "is_leap".into(), "(i64)->bool".into()),
        ("hash.fnv1a32".into(), "hash".into(), "fnv1a32".into(), "(&[u8])->u32".into()),
        ("math.sin".into(), "math".into(), "sin".into(), "(f64)->f64".into()),
    ]
}

fn leap_spec() -> Vec<competence_catalog::semantic::Probe> {
    vec![probe(json!([2024]), "true"), probe(json!([1900]), "false"), probe(json!([2000]), "true")]
}

fn engine(rules: Vec<Rule>, proposer: Box<dyn competence_catalog::engine::Proposer>) -> Engine {
    Engine::new(RuntimeBridge::new_inprocess(), &entries(), rules, proposer)
}

#[test]
fn extraction_crystallizes_then_decays_to_g0() {
    let mut e = engine(vec![], Box::new(NoProposer));
    // G0 empty: the first novel spec is extracted from the basis at G2.
    let first = e.resolve("(i64)->bool", &leap_spec()).unwrap();
    assert_eq!(first.gear, 2, "first coverage is cheapest extraction");
    assert!(first.handler.contains("is_leap"));
    // Crystallized: the next instance hits G0 at the floor.
    let second = e.resolve("(i64)->bool", &leap_spec()).unwrap();
    assert_eq!(second.gear, 0, "after crystallization, reuse hits G0");
    assert_eq!(second.cost_uj, 1);
    // Many G0 reuses drive cost per act toward the floor.
    for _ in 0..50 { e.resolve("(i64)->bool", &leap_spec()).unwrap(); }
    assert!(e.cost_per_act() < 100.0, "cost per act decays: {}", e.cost_per_act());
    assert_eq!(e.gear_acts[0], 51);
    assert_eq!(e.gear_acts[2], 1);
}

#[test]
fn proven_rule_resolves_at_g1() {
    let rule = Rule {
        shape: "(i64)->bool".into(),
        applies: Box::new(|_| true),
        handler: "time.civil.is_leap".into(),
    };
    let mut e = engine(vec![rule], Box::new(NoProposer));
    let r = e.resolve("(i64)->bool", &leap_spec()).unwrap();
    assert_eq!(r.gear, 1, "a proven rule resolves before extraction");
    assert_eq!(r.cost_uj, GEAR_COST[1]);
}

#[test]
fn gate_refuses_a_bad_proposal() {
    // The proposer names math.sin for a 3-integer spec it cannot cover.
    let mut e = engine(vec![], Box::new(FixedProposer { handler: "math.sin".into(), gear: 3 }));
    let novel = vec![probe(json!([1, 2, 3]), "6")];
    let r = e.resolve("(i64,i64,i64)->i64", &novel);
    assert!(r.is_none(), "a proposal that fails replay is refused, not used");
    assert_eq!(e.acts(), 0);
}

#[test]
fn no_proposer_stops_at_the_basis() {
    let mut e = engine(vec![], Box::new(NoProposer));
    // A shape the basis does not cover and no proposer: resolution fails honestly.
    let r = e.resolve("(i64,i64,i64)->i64", &[probe(json!([1, 2, 3]), "6")]);
    assert!(r.is_none());
}

#[test]
fn engine_synthesizes_then_reuses_at_g0() {
    let math_entries = vec![
        ("math.fabs".to_string(), "math".to_string(), "fabs".to_string(), "(f64)->f64".to_string()),
        ("math.sqrt".to_string(), "math".to_string(), "sqrt".to_string(), "(f64)->f64".to_string()),
        ("math.exp".to_string(), "math".to_string(), "exp".to_string(), "(f64)->f64".to_string()),
        ("math.sin".to_string(), "math".to_string(), "sin".to_string(), "(f64)->f64".to_string()),
    ];
    let mut e = Engine::new(RuntimeBridge::new_inprocess(), &math_entries, vec![], Box::new(NoProposer));
    let spec = vec![probe(json!([-4.0]), "2"), probe(json!([9.0]), "3"), probe(json!([0.0]), "0")];
    // No single op is sqrt(|x|): G2 synthesizes the composition.
    let first = e.resolve("(f64)->f64", &spec).unwrap();
    assert_eq!(first.gear, 2, "synthesis at G2");
    assert!(first.handler.contains("sqrt") && first.handler.contains("fabs"));
    // The synthesized program is reused at G0 next time.
    let second = e.resolve("(f64)->f64", &spec).unwrap();
    assert_eq!(second.gear, 0, "synthesized competence reused at the floor");
    assert_eq!(second.cost_uj, 1);
}

#[test]
fn gear_costs_are_competence_before_model() {
    assert!(GEAR_COST[0] < GEAR_COST[1]);
    assert!(GEAR_COST[1] < GEAR_COST[2]);
    assert!(GEAR_COST[2] < GEAR_COST[3]);
    assert_eq!(GEAR_COST[0], 1, "G0 at the interpreter floor");
}
