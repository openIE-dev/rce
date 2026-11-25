use competence_catalog::realtime::{chunk_coherent, control_gears, select, Chunk, Decision, Gear, Loop};

fn khz(period_us: u64, memory_kib: u64) -> Loop {
    Loop { period_us, deadline_us: period_us, memory_kib }
}

#[test]
fn one_khz_loop_admits_lookup_not_jit_or_proposer() {
    let d = select(&khz(1000, 4096), &control_gears());
    assert_eq!(d, Decision::Admit { gear: "g0-lookup", value: 1.0 });
}

#[test]
fn jit_compile_spike_excluded_from_fast_loop() {
    let gears = vec![
        Gear { name: "g2-jit-cold", wcet_us: 20_000, memory_kib: 2048, benefit: 2.0, joules: 20.0 },
        Gear { name: "g0-lookup", wcet_us: 5, memory_kib: 64, benefit: 1.0, joules: 1.0 },
    ];
    let d = select(&khz(1000, 4096), &gears);
    assert_eq!(d, Decision::Admit { gear: "g0-lookup", value: 1.0 });
}

#[test]
fn memory_bound_loop_refuses_thor_class_gear() {
    let gears = vec![
        Gear { name: "thor-mind", wcet_us: 100, memory_kib: 1_048_576, benefit: 8.0, joules: 100.0 },
        Gear { name: "small-mind", wcet_us: 100, memory_kib: 512, benefit: 1.0, joules: 100.0 },
    ];
    let d = select(&khz(1000, 4096), &gears);
    assert_eq!(d, Decision::Admit { gear: "small-mind", value: 0.01 });
}

#[test]
fn no_gear_fits_refuses_fail_safe() {
    let d = select(&khz(10, 16), &control_gears());
    match d {
        Decision::Refuse { reason } => assert!(reason.contains("no gear fits")),
        other => panic!("expected refusal, got {:?}", other),
    }
}

#[test]
fn incoherent_budget_refuses() {
    let lp = Loop { period_us: 1000, deadline_us: 2000, memory_kib: 4096 };
    assert!(matches!(select(&lp, &control_gears()), Decision::Refuse { .. }));
    let zero = Loop { period_us: 0, deadline_us: 0, memory_kib: 4096 };
    assert!(matches!(select(&zero, &control_gears()), Decision::Refuse { .. }));
}

#[test]
fn ties_break_deterministically_by_name() {
    let gears = vec![
        Gear { name: "b", wcet_us: 5, memory_kib: 64, benefit: 1.0, joules: 1.0 },
        Gear { name: "a", wcet_us: 5, memory_kib: 64, benefit: 1.0, joules: 1.0 },
    ];
    let d1 = select(&khz(1000, 4096), &gears);
    let mut rev = gears.clone();
    rev.reverse();
    let d2 = select(&khz(1000, 4096), &rev);
    assert_eq!(d1, d2);
    assert_eq!(d1, Decision::Admit { gear: "a", value: 1.0 });
}

#[test]
fn invalid_gear_figures_refuse() {
    let gears = vec![Gear { name: "bad", wcet_us: 5, memory_kib: 64, benefit: 1.0, joules: 0.0 }];
    assert!(matches!(select(&khz(1000, 4096), &gears), Decision::Refuse { .. }));
}

#[test]
fn chunk_coherence_visits_waypoints_in_order() {
    let ok = Chunk { plan: vec![0.0, 1.0, 2.0], ticks: vec![0.0, 0.4, 1.0, 1.6, 2.0] };
    assert!(chunk_coherent(&ok, 0.01));
    let skipped = Chunk { plan: vec![0.0, 1.0, 2.0], ticks: vec![0.0, 2.0] };
    assert!(!chunk_coherent(&skipped, 0.01));
    let out_of_order = Chunk { plan: vec![0.0, 1.0], ticks: vec![1.0, 0.0] };
    assert!(!chunk_coherent(&out_of_order, 0.01));
    let empty_plan = Chunk { plan: vec![], ticks: vec![9.0] };
    assert!(chunk_coherent(&empty_plan, 0.01));
    assert!(!chunk_coherent(&ok, f64::NAN));
}
