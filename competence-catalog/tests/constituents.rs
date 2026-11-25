//! Falsifiers for the new constituents: RCE-BENCH (fixed gate + joules), the RCE Compiler
//! (native lowering to the floor, placement-derived fusion), RCE-Conform (verified legacy
//! conversion), and living placement (capability lookup, no flags).

use std::collections::HashSet;

use competence_catalog::bench::{filter, run_suite, suite, summary};
use competence_catalog::worldmodel::plan;
use competence_catalog::compiler::{compile_src, Arch, Target};
use competence_catalog::conform::convert;
use competence_catalog::fabric::{builtin_fabrics, lower_muladd, place, Kind};
use competence_catalog::placement::{builtin_targets, fma_saving, place_muladd, plan_muladd, Prim};

#[test]
fn bench_gate_is_fixed_and_physical() {
    let t = &suite()[0]; // add: 2+3 = 5
    let empty = HashSet::new();
    let good = filter(t, "2 + 3", &empty);
    assert!(good.correct);
    assert_eq!(good.est_uj, 1_000); // synthesized, not reuse
    let bad = filter(t, "2 + 4", &empty);
    assert!(!bad.correct); // the gate refuses; no goalpost moves
    assert_eq!(bad.est_uj, 5_000_000);
}

#[test]
fn bench_reuse_prices_at_the_floor() {
    let t = &suite()[0];
    let mut corpus = HashSet::new();
    let first = filter(t, "2 + 3", &corpus);
    corpus.insert(first.key.clone());
    // A different surface, same behaviour -> same key -> G0 reuse at 1 µJ.
    let again = filter(t, "3 + 2", &corpus);
    assert!(again.reused);
    assert_eq!(again.est_uj, 1);
    assert_eq!(again.key, first.key);
}

#[test]
fn bench_suite_summary_is_physical() {
    let empty = HashSet::new();
    let rs = run_suite(|t| t.intent.to_string(), &empty);
    let (pass, joules) = summary(&rs);
    assert_eq!(pass, 1.0); // every intent meets its own acceptance
    assert!(joules > 0);
}

#[test]
fn compiler_emits_fused_primitive_only_where_capable() {
    let fma = Target { arch: Arch::Arm64, has_fma: true };
    let no = Target { arch: Arch::Arm64, has_fma: false };
    let with = compile_src("(a * b) + c", &fma).unwrap();
    assert!(with.contains("madd"), "fusion derived from capability: {with}");
    let without = compile_src("(a * b) + c", &no).unwrap();
    assert!(without.contains("mul") && without.contains("add"));
    assert!(!without.contains("madd"));
}

#[test]
fn compiler_lowers_to_x86_without_external_compiler() {
    let x86 = Target { arch: Arch::X86, has_fma: false };
    let out = compile_src("(a * b) + c", &x86).unwrap();
    assert!(out.contains("imul") && out.contains("add") && out.contains("ret"));
    let lit = compile_src("2 + 3", &Target { arch: Arch::Arm64, has_fma: true }).unwrap();
    assert!(lit.contains("mov") && lit.contains("ret"));
}

#[test]
fn conform_converts_legacy_verified() {
    let c = convert("2 + 3", "python", None).unwrap();
    assert!(c.verified); // behaviour preserved
    assert!(!c.repaired);
    assert_eq!(c.form, "5");
    // Same competence, different modern language -> same name-free key.
    let r = convert("2 + 3", "rust", None).unwrap();
    assert_eq!(c.key, r.key);
}

#[test]
fn conform_repairs_to_modern_spec() {
    let c = convert("2 + 3", "rust", Some(6)).unwrap();
    assert!(c.repaired);
    assert!(c.verified);
    assert_eq!(c.form, "6");
}

#[test]
fn placement_is_derived_not_flagged() {
    let targets = builtin_targets();
    let fma_t = targets.iter().find(|t| t.name == "cpu-fma").unwrap();
    let scalar = targets.iter().find(|t| t.name == "cpu-scalar").unwrap();
    assert_eq!(plan_muladd(fma_t).ops, vec![Prim::Fma]);
    assert_eq!(plan_muladd(scalar).ops, vec![Prim::Mul, Prim::Add]);
    assert!(fma_saving(fma_t) > 0.0); // the waste the curse leaves, measurable
    let (best, _) = place_muladd(&targets).unwrap();
    assert_eq!(best.name, "amx-tensor"); // argmin joules
}

#[test]
fn heterogeneous_breadth_covers_all_fabric_kinds() {
    let fs = builtin_fabrics();
    assert!(fs.len() >= 20, "global fabric breadth, got {}", fs.len());
    for k in [Kind::Cpu, Kind::Gpu, Kind::Tensor, Kind::Dataflow, Kind::Wasm, Kind::MlIr] {
        assert!(fs.iter().any(|f| f.kind == k), "missing fabric kind {k:?}");
    }
    // Non-Western fabrics present (global, not Western-first).
    assert!(fs.iter().any(|f| f.name == "ascend-davinci"));
    assert!(fs.iter().any(|f| f.name == "cambricon"));
}

#[test]
fn one_form_lowers_to_every_fabric() {
    for f in builtin_fabrics() {
        let out = lower_muladd(&f);
        assert!(!out.is_empty(), "{} lowered empty", f.name);
    }
    let arm = builtin_fabrics().into_iter().find(|f| f.name == "cpu-arm64").unwrap();
    assert!(lower_muladd(&arm).contains("fmadd"));
    let wasm = builtin_fabrics().into_iter().find(|f| f.name == "wasm").unwrap();
    assert!(lower_muladd(&wasm).contains("f32.mul")); // no FMA assumed -> split
}

#[test]
fn heterogeneous_placement_picks_cheapest_fabric() {
    let fs = builtin_fabrics();
    let best = place(&fs).unwrap();
    assert!(matches!(best.kind, Kind::Dataflow | Kind::Tensor), "cheapest should be tensor/dataflow, got {}", best.name);
}

#[test]
fn worldmodel_plans_with_exact_physics() {
    // The forward model is exact (eval); only the inverse (which edit) is searched.
    let p = plan("2 + 3", 6, 8, 2).expect("a plan");
    assert!(p.verified);
    assert_eq!(p.form, "6");
    assert!(p.edits.iter().any(|e| e.contains("operator")), "edits {:?}", p.edits);
}

#[test]
fn worldmodel_already_at_goal_is_floor() {
    let p = plan("2 + 3", 5, 8, 2).unwrap();
    assert!(p.edits.is_empty());
    assert_eq!(p.cost_uj, 1);
}

#[test]
fn worldmodel_unreachable_goal_returns_none() {
    assert!(plan("2 + 3", 999, 4, 1).is_none());
}