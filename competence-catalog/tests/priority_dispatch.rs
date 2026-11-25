use competence_catalog::rtos::{priority_dispatch, Task};

#[test]
fn shorter_period_runs_first() {
    let ts = vec![
        Task { name: "plan", period_us: 4, wcet_us: 1 },
        Task { name: "ctrl", period_us: 2, wcet_us: 1 },
    ];
    let order = priority_dispatch(&ts, 4).unwrap();
    assert_eq!(order, vec!["ctrl", "ctrl"]);
}

#[test]
fn equal_period_ties_break_by_name() {
    let ts = vec![
        Task { name: "b", period_us: 4, wcet_us: 1 },
        Task { name: "a", period_us: 4, wcet_us: 1 },
    ];
    let order = priority_dispatch(&ts, 2).unwrap();
    assert_eq!(order, vec!["a"]);
}

#[test]
fn unschedulable_set_refuses() {
    let ts = vec![
        Task { name: "a", period_us: 2, wcet_us: 2 },
        Task { name: "b", period_us: 2, wcet_us: 2 },
    ];
    assert!(priority_dispatch(&ts, 4).is_none());
}
