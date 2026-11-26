use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::program::{tokio_io_run, Env};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

#[test]
fn real_os_event_loop_delivers_completions_deterministically() {
    let tasks = vec![
        ("a".to_string(), Stmt::Seq(vec![
            Stmt::AwaitEvent(1),
            Stmt::Return(e("io + 1")),
        ]), Env::new()),
        ("b".to_string(), Stmt::Seq(vec![
            Stmt::AwaitEvent(2),
            Stmt::Return(e("io * 2")),
        ]), Env::new()),
    ];
    let events = vec![(20, "b".to_string(), 5), (10, "a".to_string(), 41)];
    let out = tokio_io_run(tasks, events).unwrap();
    assert_eq!(out, vec![("a".to_string(), Some(42)), ("b".to_string(), Some(10))]);
}

#[test]
fn tokio_io_is_reproducible_across_runs() {
    let build = || {
        let tasks = vec![
            ("t0".to_string(), Stmt::Seq(vec![Stmt::AwaitEvent(1), Stmt::Return(e("io"))]), Env::new()),
            ("t1".to_string(), Stmt::Seq(vec![Stmt::Let("z".into(), e("3")), Stmt::Return(e("z"))]), Env::new()),
        ];
        tokio_io_run(tasks, vec![(5, "t0".to_string(), 9)]).unwrap()
    };
    assert_eq!(build(), build());
}
