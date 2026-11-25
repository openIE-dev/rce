use competence_catalog::rtos::{cyclic_table, rm_bound, schedulable, utilization, Task};

#[test]
fn liu_layland_bound_and_utilization() {
    assert!((rm_bound(2) - 0.8284271247461903).abs() < 1e-9);
    let ts = vec![
        Task { name: "ctrl", period_us: 1000, wcet_us: 500 },
        Task { name: "plan", period_us: 2000, wcet_us: 500 },
    ];
    assert!((utilization(&ts) - 0.75).abs() < 1e-12);
    assert!(schedulable(&ts));
}

#[test]
fn above_bound_refuses_unknown() {
    let ts = vec![
        Task { name: "a", period_us: 1000, wcet_us: 600 },
        Task { name: "b", period_us: 2000, wcet_us: 300 },
        Task { name: "c", period_us: 4000, wcet_us: 300 },
    ];
    assert!(utilization(&ts) > rm_bound(3));
    assert!(!schedulable(&ts));
}

#[test]
fn wcet_over_period_refuses() {
    let ts = vec![Task { name: "a", period_us: 1000, wcet_us: 1500 }];
    assert!(!schedulable(&ts));
    assert!(!schedulable(&[]));
}

#[test]
fn cyclic_table_sorted_offsets() {
    let ts = vec![
        Task { name: "ctrl", period_us: 1000, wcet_us: 100 },
        Task { name: "plan", period_us: 2000, wcet_us: 200 },
    ];
    let t = cyclic_table(&ts).unwrap();
    assert_eq!(t, vec![("ctrl", 0), ("plan", 0), ("ctrl", 1000)]);
}

#[test]
fn exploding_hyperperiod_refuses() {
    let ts = vec![
        Task { name: "a", period_us: 4093, wcet_us: 1 },
        Task { name: "b", period_us: 4099, wcet_us: 1 },
    ];
    assert!(cyclic_table(&ts).is_none());
}
