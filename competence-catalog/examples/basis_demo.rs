//! The basis-set demonstration: an intent-keyed reference database vs exact-input
//! memoization, on the same workload of never-before-seen inputs.

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

fn main() {
    let ops = [
        ("sort", "List[Int]"),
        ("upper", "Text"),
        ("lower", "Text"),
        ("reverse", "Text"),
        ("sum", "List[Int]"),
    ];
    let acts = 1000;

    // Intent-keyed reference database (the strong starting basis set).
    let mut cat = Catalog::new().unwrap();
    let seeded = cat.seed_basis(&basis());
    let mut seed = 0x1234_5678u64;
    for _ in 0..acts {
        let (op, shape) = ops[(lcg(&mut seed) % 5) as usize];
        let args = format!("{}", lcg(&mut seed));
        cat.resolve_intent(op, shape, &args, "probe");
    }
    let m = cat.meter();

    // Exact-input memoization on the same novel workload.
    let mut exact = Catalog::new().unwrap();
    let mut seed2 = 0x1234_5678u64;
    for _ in 0..acts {
        let (op, _) = ops[(lcg(&mut seed2) % 5) as usize];
        let args = format!("{}", lcg(&mut seed2));
        exact.resolve(&args, op, "probe");
    }
    let em = exact.meter();

    println!("The starting basis set — intent key vs exact-input memoization");
    println!("  workload: {acts} never-before-seen inputs over 5 known operations");
    println!();
    println!("  INTENT-KEYED reference database (seeded basis = {seeded}):");
    println!("    hits / misses:   {} / {}", m.hits, m.misses);
    println!("    reuse rate:      {:.4}", m.reuse_rate);
    println!("    cost / act:      {:.1} µJ  (G0 interpreter floor)", m.cost_per_act_uj);
    println!("    proposer calls:  {}", m.proposer_calls);
    println!();
    println!("  EXACT-INPUT memoization (same workload):");
    println!("    hits / misses:   {} / {}", em.hits, em.misses);
    println!("    reuse rate:      {:.4}", em.reuse_rate);
    println!("    cost / act:      {:.0} µJ", em.cost_per_act_uj);
    println!();
    println!(
        "  → the intent key turns novel instances into G0 hits: {:.0}× cheaper per act.",
        em.cost_per_act_uj / m.cost_per_act_uj
    );
    println!("  → a strong starting basis spans the task space; the residual is novel operations only.");
}