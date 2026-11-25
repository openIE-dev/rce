//! Falsifiers for the four obscure-frontier ingestions: calibrated cost (Horowitz/Agner),
//! deterministic record/replay (rr/QEMU), structural rewrite (ast-grep/comby), and library
//! learning (DreamCoder/stitch).

use competence_catalog::costdata::{cpu_model, fma_saving_calibrated, instruction_energy};
use competence_catalog::library::{compressed_length, description_length, learn_library};
use competence_catalog::lift::lift_expr;
use competence_catalog::placement::Prim;
use competence_catalog::replay::{record, replay, verify_chain};
use competence_catalog::structural::{node_count, simplify};

#[test]
fn calibrated_cost_fma_beats_mul_add_and_dram_dominates() {
    assert!(fma_saving_calibrated() > 0.0);
    assert!(instruction_energy(Prim::Fma) < instruction_energy(Prim::Mul) + instruction_energy(Prim::Add));
    let m = cpu_model();
    assert!(m.b > m.a * 100.0, "data movement must dominate (b≫a)");
}

#[test]
fn record_replay_is_byte_identical() {
    let tasks = vec![("2 + 3".to_string(), 5), ("2 * 3".to_string(), 6)];
    let rec = record(&tasks);
    assert!(verify_chain(&rec));
    let outs = replay(&rec).unwrap();
    assert_eq!(outs, vec![5, 6]);
}

#[test]
fn tampered_recording_is_detected() {
    let tasks = vec![("2 + 3".to_string(), 5)];
    let mut rec = record(&tasks);
    rec.events[0].out = 999; // tamper
    assert!(!verify_chain(&rec));
    assert!(replay(&rec).is_err());
}

#[test]
fn structural_rewrite_reduces_description_length() {
    let e = lift_expr("(a + 0) * 1").unwrap();
    let s = simplify(&e);
    assert_eq!(node_count(&s), 1); // collapses to `a`
    assert!(node_count(&s) < node_count(&e));
    let z = simplify(&lift_expr("a - a").unwrap());
    assert_eq!(node_count(&z), 1); // 0
    let d = simplify(&lift_expr("-(-a)").unwrap());
    assert_eq!(node_count(&d), 1); // a
}

#[test]
fn library_learning_compresses_the_corpus() {
    let corpus: Vec<_> = (0..3).map(|_| lift_expr("(a * b) + c").unwrap()).collect();
    let lib = learn_library(&corpus, 2);
    assert!(!lib.is_empty(), "a repeated abstraction must be learned");
    let orig = description_length(&corpus);
    let comp = compressed_length(&corpus, &lib);
    assert!(comp < orig, "library must shrink description length: {comp} < {orig}");
}