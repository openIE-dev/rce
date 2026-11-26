use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::program::{eval_program, threaded_run, Env, FnTable};

fn task(n: i128) -> Stmt {
    Stmt::Seq(vec![
        Stmt::Let("x".into(), lift_expr(&format!("{n}")).unwrap()),
        Stmt::Return(lift_expr("x * 2").unwrap()),
    ])
}

#[test]
fn real_threads_match_sequential_and_commit_in_id_order() {
    let tasks: Vec<(String, Stmt, Env)> = (0..8)
        .map(|i| (format!("t{i}"), task(i), Env::new()))
        .collect();
    let par = threaded_run(&tasks).unwrap();
    let mut seq = Vec::new();
    for (id, stmt, env) in &tasks {
        let mut e = env.clone();
        seq.push((id.clone(), eval_program(stmt, &mut e, &mut FnTable::new()).unwrap()));
    }
    seq.sort_by(|a, b| a.0.cmp(&b.0));
    assert_eq!(par, seq);
    assert_eq!(par[3].1, Some(6));
}

#[test]
fn threaded_is_deterministic_across_runs() {
    let tasks: Vec<(String, Stmt, Env)> = (0..6)
        .map(|i| (format!("t{i}"), task(i), Env::new()))
        .collect();
    let a = threaded_run(&tasks).unwrap();
    let b = threaded_run(&tasks).unwrap();
    assert_eq!(a, b);
}
