use competence_catalog::program::{eval_stmt, lift_program, Env};

fn run(src: &str) -> Result<Option<i128>, String> {
    let p = lift_program(src)?;
    eval_stmt(&p, &mut Env::new())
}

#[test]
fn for_loop_sums() {
    assert_eq!(run("let s = 0; for i in 1..6 { s = s + i; } return s;").unwrap(), Some(15));
}

#[test]
fn while_loop_sums() {
    assert_eq!(run("let s = 0; let i = 1; while i < 6 { s = s + i; i = i + 1; } return s;").unwrap(), Some(15));
}

#[test]
fn if_else_branches() {
    assert_eq!(run("let x = 3; if x < 2 { return 1; } else { return 2; }").unwrap(), Some(2));
}

#[test]
fn assign_to_undeclared_refused() {
    assert!(run("s = 1;").is_err());
}

#[test]
fn infinite_loop_refused_by_fuel() {
    assert!(run("while 1 { }").is_err());
}

#[test]
fn import_and_type_are_declarations() {
    assert_eq!(run("import math; type T; return 7;").unwrap(), Some(7));
}
