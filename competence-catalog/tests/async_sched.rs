use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::program::{Env, FnTable, Scheduler};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

fn task() -> Stmt {
    Stmt::Seq(vec![
        Stmt::Let("x".into(), e("1")),
        Stmt::Await(Box::new(Stmt::Return(e("0")))),
        Stmt::Assign("x".into(), e("x + 1")),
    ])
}

#[test]
fn tasks_interleave_deterministically_round_robin() {
    let mut s = Scheduler::new();
    s.spawn("a", task(), Env::new());
    s.spawn("b", task(), Env::new());
    s.run_to_completion(&mut FnTable::new(), 100).unwrap();
    assert_eq!(s.log, vec![
        "a:step", "b:step", "a:yield", "b:yield", "a:step", "b:step",
    ]);
}

#[test]
fn spawn_order_fixes_interleaving() {
    let mut s1 = Scheduler::new();
    s1.spawn("a", task(), Env::new());
    s1.spawn("b", task(), Env::new());
    s1.run_to_completion(&mut FnTable::new(), 100).unwrap();
    let mut s2 = Scheduler::new();
    s2.spawn("a", task(), Env::new());
    s2.spawn("b", task(), Env::new());
    s2.run_to_completion(&mut FnTable::new(), 100).unwrap();
    assert_eq!(s1.log, s2.log);
}

#[test]
fn await_yields_without_executing_inner() {
    let mut s = Scheduler::new();
    s.spawn("t", Stmt::Seq(vec![
        Stmt::Await(Box::new(Stmt::Return(e("99")))),
        Stmt::Let("y".into(), e("1")),
    ]), Env::new());
    s.run_to_completion(&mut FnTable::new(), 100).unwrap();
    assert_eq!(s.log, vec!["t:yield", "t:step"]);
}

#[test]
fn round_cap_refuses_runaway() {
    let mut s = Scheduler::new();
    s.spawn("loop", Stmt::Seq(vec![Stmt::While(e("1"), Box::new(Stmt::Seq(vec![])))]), Env::new());
    assert!(s.run_to_completion(&mut FnTable::new(), 5).is_err());
}
