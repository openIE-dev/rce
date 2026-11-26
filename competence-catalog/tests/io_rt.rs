use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::program::{Env, FnTable, IoRuntime};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

#[test]
fn higher_priority_preempts_each_tick() {
    let mut rt = IoRuntime::new();
    rt.spawn("low", 9, Stmt::Seq(vec![Stmt::Let("a".into(), e("1")), Stmt::Let("a".into(), e("2"))]), Env::new());
    rt.spawn("high", 1, Stmt::Seq(vec![Stmt::Let("b".into(), e("1")), Stmt::Let("b".into(), e("2"))]), Env::new());
    rt.run(&mut FnTable::new(), 50).unwrap();
    // high runs first in every tick it is ready
    assert_eq!(rt.log[..4], ["high:step", "low:step", "high:step", "low:step"]);
}

#[test]
fn await_event_suspends_until_completion_then_resumes() {
    let mut rt = IoRuntime::new();
    rt.spawn("t", 1, Stmt::Seq(vec![
        Stmt::AwaitEvent(7),
        Stmt::Let("y".into(), e("io + 1")),
        Stmt::Return(e("y")),
    ]), Env::new());
    rt.event(3, "t", 41);
    rt.run(&mut FnTable::new(), 50).unwrap();
    assert_eq!(rt.log, vec!["t:wait", "t:io", "t:step", "t:return"]);
}

#[test]
fn io_interleaving_is_deterministic() {
    let build = || {
        let mut rt = IoRuntime::new();
        rt.spawn("a", 1, Stmt::Seq(vec![Stmt::AwaitEvent(1), Stmt::Return(e("io"))]), Env::new());
        rt.spawn("b", 2, Stmt::Seq(vec![Stmt::Let("z".into(), e("1")), Stmt::Return(e("z"))]), Env::new());
        rt.event(2, "a", 5);
        rt.run(&mut FnTable::new(), 50).unwrap();
        rt.log
    };
    assert_eq!(build(), build());
}

#[test]
fn missing_completion_hits_tick_cap() {
    let mut rt = IoRuntime::new();
    rt.spawn("t", 1, Stmt::Seq(vec![Stmt::AwaitEvent(99)]), Env::new());
    assert!(rt.run(&mut FnTable::new(), 5).is_err());
}
