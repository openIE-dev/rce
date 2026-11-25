//! Prints the Law-of-Coding falsifier curve: cost per competent act vs corpus size.

use competence_catalog::Catalog;

fn main() {
    let mut cat = Catalog::new().unwrap();
    let inputs: Vec<String> = (0..8).map(|i| format!("task-{i}")).collect();
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };

    let acts = 1000u64;
    let mut first_half = 0u64;
    for act in 0..acts {
        let idx = next() % inputs.len();
        let (_, cost, _) = cat.resolve(&inputs[idx], "upper", "probe");
        if act < acts / 2 {
            first_half += cost;
        }
    }
    let m = cat.meter();
    let second_half = m.workload_est_uj - first_half;
    let cpa1 = first_half as f64 / (acts / 2) as f64;
    let cpa2 = second_half as f64 / (acts / 2) as f64;

    println!("Law of Coding — competence catalog falsifier");
    println!("  acts:                 {}", m.acts);
    println!("  catalog size |H_corpus|: {}", m.catalog_size);
    println!("  hits / misses:        {} / {}", m.hits, m.misses);
    println!("  reuse rate:           {:.4}", m.reuse_rate);
    println!("  G0 floor / G4 proposer: {} µJ / {} µJ", cat.costs().g0, cat.costs().g4);
    println!("  cost/act first half:  {cpa1:.1} µJ");
    println!("  cost/act second half: {cpa2:.3} µJ");
    println!("  cost/act overall:     {:.1} µJ", m.cost_per_act_uj);
    println!("  reduction factor:     {:.0}x", cpa1 / cpa2);
}