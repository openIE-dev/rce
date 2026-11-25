use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::program::{program_key, replay_admit, Env};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

// prog: let y = x + 1; return y * 2;
fn prog() -> Stmt {
    Stmt::Seq(vec![
        Stmt::Let("y".into(), e("x + 1")),
        Stmt::Return(e("y * 2")),
    ])
}

fn seed(x: i128) -> Env {
    let mut env = Env::new();
    env.insert("x".into(), x);
    env
}

#[test]
fn whole_program_admitted_by_replay_of_every_probe() {
    let probes = vec![(seed(1), 4), (seed(2), 6), (seed(10), 22)];
    assert!(replay_admit(&prog(), &probes));
}

#[test]
fn one_bad_probe_refuses_admission() {
    let probes = vec![(seed(1), 4), (seed(2), 999)];
    assert!(!replay_admit(&prog(), &probes));
}

#[test]
fn empty_probes_do_not_admit_vacuously() {
    // An empty spec carries no information; admission requires at least one probe.
    assert!(!replay_admit(&prog(), &[]));
}

#[test]
fn program_key_is_deterministic_and_name_free() {
    let k1 = program_key(&prog()).unwrap();
    let k2 = program_key(&prog()).unwrap();
    assert_eq!(k1, k2);
    assert_eq!(k1.len(), 64);
    let other = Stmt::Seq(vec![Stmt::Return(e("0"))]);
    assert_ne!(k1, program_key(&other).unwrap());
}
