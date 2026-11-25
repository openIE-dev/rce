//! Public synthesis falsifiers: compose basis operations into new code that
//! covers a spec no single operation does, verified by replay. Synthesis
//! proposes; the gate disposes.

use competence_catalog::runtime::RuntimeBridge;
use competence_catalog::semantic::Probe;
use competence_catalog::synthesis::Synthesizer;
use serde_json::json;

fn entries() -> Vec<(String, String, String, String)> {
    vec![
        ("math.fabs".into(), "math".into(), "fabs".into(), "(f64)->f64".into()),
        ("math.sqrt".into(), "math".into(), "sqrt".into(), "(f64)->f64".into()),
        ("math.exp".into(), "math".into(), "exp".into(), "(f64)->f64".into()),
        ("math.sin".into(), "math".into(), "sin".into(), "(f64)->f64".into()),
    ]
}

fn synth() -> Synthesizer {
    Synthesizer::new(RuntimeBridge::new_inprocess(), &entries())
}

#[test]
fn synthesizes_sqrt_of_abs_from_parts() {
    let s = synth();
    // No single basis op is sqrt(|x|); the composition covers it.
    let probes = vec![
        Probe::new(json!([-4.0]), "2"),
        Probe::new(json!([9.0]), "3"),
        Probe::new(json!([0.0]), "0"),
    ];
    let comp = s.synthesize("(f64)->f64", &probes).expect("sqrt(fabs(x)) is synthesizable");
    assert_eq!(comp.ops.len(), 2, "a two-operation composition");
    assert!(comp.expr.contains("sqrt") && comp.expr.contains("fabs"), "expr: {}", comp.expr);
    // The synthesized program re-executes (the gate can replay it).
    let b = RuntimeBridge::new_inprocess();
    let got = b.run_caller("synth", &comp.program, "Calc(x: -16.0)").unwrap();
    assert_eq!(got, "4", "sqrt(fabs(-16)) = 4");
}

#[test]
fn uncoverable_spec_returns_none() {
    let s = synth();
    // No composition of these ops yields 999 at x=1.
    let probes = vec![Probe::new(json!([1.0]), "999")];
    assert!(s.synthesize("(f64)->f64", &probes).is_none());
}

#[test]
fn learned_synthesis_prefers_experienced_parts() {
    use competence_catalog::learning::Hindsight;
    let s = synth();
    let mut h = Hindsight::new();
    // Experience: sqrt and fabs have been reused; exp/sin have not.
    for _ in 0..3 {
        h.record("k", "(f64)->f64", "math.sqrt", 0);
        h.record("k", "(f64)->f64", "math.fabs", 0);
    }
    let probes = vec![
        Probe::new(json!([-4.0]), "2"),
        Probe::new(json!([9.0]), "3"),
        Probe::new(json!([0.0]), "0"),
    ];
    let comp = s.synthesize_learned("(f64)->f64", &probes, &h).expect("learned synthesis covers it");
    assert!(comp.ops.contains(&"math.sqrt".to_string()) && comp.ops.contains(&"math.fabs".to_string()));
}

#[test]
fn multi_arg_is_out_of_scope() {
    let s = synth();
    let probes = vec![Probe::new(json!([1.0, 2.0]), "3")];
    assert!(s.synthesize("(f64,f64)->f64", &probes).is_none());
}

#[test]
fn deep_three_op_composition() {
    let s = synth();
    // sqrt(sqrt(sqrt(256)))=2 and sqrt(sqrt(sqrt(65536)))=4 — needs three ops.
    let probes = vec![Probe::new(json!([256.0]), "2"), Probe::new(json!([65536.0]), "4")];
    let comp = s.synthesize_deep("(f64)->f64", &probes).expect("a 3-op composition covers it");
    assert_eq!(comp.ops.len(), 3, "deeper pipeline");
    // The 2-op search cannot reach it.
    assert!(s.synthesize("(f64)->f64", &probes).is_none());
}
