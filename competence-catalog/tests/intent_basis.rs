//! The basis-set falsifier: an intent-keyed reference database turns *novel
//! instances* of known operations into G0 hits (zero model calls), where
//! exact-input memoization misses every novel input. A novel operation
//! crystallizes once, then the miss rate decays.

use competence_catalog::Catalog;

fn basis() -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    vec![
        ("sort", "List[Int]", "sort", "3 1 2"),
        ("upper", "Text", "upper", "a"),
        ("lower", "Text", "lower", "A"),
        ("reverse", "Text", "reverse", "ab"),
        ("sum", "List[Int]", "sum", "1 2"),
    ]
}

fn lcg(seed: &mut u64) -> u64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *seed >> 33
}

#[test]
fn novel_instances_of_known_intents_hit_g0() {
    let mut cat = Catalog::new().unwrap();
    assert_eq!(cat.seed_basis(&basis()), 5);

    let ops = [
        ("sort", "List[Int]"),
        ("upper", "Text"),
        ("lower", "Text"),
        ("reverse", "Text"),
        ("sum", "List[Int]"),
    ];
    let mut seed = 0xdead_beefu64;
    for _ in 0..200 {
        let (op, shape) = ops[(lcg(&mut seed) % 5) as usize];
        let args = format!("{}", lcg(&mut seed)); // a never-before-seen input
        let (_, cost, hit) = cat.resolve_intent(op, shape, &args, "probe");
        assert!(hit, "a known intent must hit even for a novel input");
        assert_eq!(cost, cat.costs().g0);
    }
    let m = cat.meter();
    assert_eq!(m.misses, 0, "the basis spans these intents; no proposer needed");
    assert_eq!(m.reuse_rate, 1.0);
    assert_eq!(m.cost_per_act_uj, 1.0);
    assert_eq!(m.catalog_size, 5, "basis unchanged — reuse, not growth");
}

#[test]
fn exact_input_memoization_misses_novel_inputs() {
    let mut cat = Catalog::new().unwrap();
    cat.resolve("hello", "upper", "probe");
    let (_, _, hit) = cat.resolve("hello", "upper", "probe");
    assert!(hit, "exact repeat hits");
    let (_, _, hit2) = cat.resolve("world", "upper", "probe");
    assert!(!hit2, "novel exact input misses — the limitation the intent key solves");
}

#[test]
fn novel_operation_crystallizes_once_then_decays() {
    let mut cat = Catalog::new().unwrap();
    cat.seed_basis(&basis()); // no "len"
    let (_, c1, h1) = cat.resolve_intent("len", "Text", "hello world", "probe");
    assert!(!h1, "a novel operation misses first");
    assert!(c1 >= 5_000_000, "the first miss pays the proposer");
    assert_eq!(cat.meter().catalog_size, 6, "the basis grew by one");

    for i in 0..50 {
        let (_, c, h) = cat.resolve_intent("len", "Text", &format!("text {i}"), "probe");
        assert!(h, "after crystallization, novel instances hit");
        assert_eq!(c, 1);
    }
    let m = cat.meter();
    assert_eq!(m.misses, 1, "only the first novel operation missed");
    assert!(m.reuse_rate > 0.98, "exponential decay toward the floor");
}

#[test]
fn gate_refuses_unverifiable_competence() {
    let mut cat = Catalog::new().unwrap();
    // "factorial" is not an executable handler → the spectest cannot be re-run.
    let out = cat.admit_intent("factorial", "Int", "factorial", "5");
    assert!(!out.is_accepted(), "unverifiable competence is refused, not ranked");
    assert_eq!(cat.meter().catalog_size, 0);
}