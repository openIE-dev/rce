use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::{builtin_luts, render_stmt, LanguageLut, Stmt};
use competence_catalog::program::{eval_program, Env, FnTable};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

#[test]
fn generic_fn_monomorphizes_and_executes() {
    let prog = Stmt::Seq(vec![
        Stmt::GenericFn("id".into(), vec!["T".into()], vec!["x".into()], Box::new(Stmt::Return(e("x")))),
        Stmt::LetCall("a".into(), "id".into(), vec![e("41")]),
        Stmt::Return(e("a + 1")),
    ]);
    assert_eq!(eval_program(&prog, &mut Env::new(), &mut FnTable::new()).unwrap(), Some(42));
}

#[test]
fn async_fn_and_await_execute_cooperatively() {
    let prog = Stmt::Seq(vec![
        Stmt::AsyncFn("work".into(), vec!["x".into()], Box::new(Stmt::Return(e("x * 2")))),
        Stmt::Await(Box::new(Stmt::LetCall("r".into(), "work".into(), vec![e("21")]))),
        Stmt::Return(e("r")),
    ]);
    assert_eq!(eval_program(&prog, &mut Env::new(), &mut FnTable::new()).unwrap(), Some(42));
}

#[test]
fn generic_type_is_a_declaration() {
    let prog = Stmt::Seq(vec![
        Stmt::GenericType("Box".into(), vec!["T".into()]),
        Stmt::Return(e("7")),
    ]);
    assert_eq!(eval_program(&prog, &mut Env::new(), &mut FnTable::new()).unwrap(), Some(7));
}

fn lut(name: &str) -> LanguageLut {
    builtin_luts().into_iter().find(|l| l.name == name).unwrap()
}

#[test]
fn generic_and_async_render_per_family() {
    let g = Stmt::GenericFn("id".into(), vec!["T".into()], vec!["x".into()], Box::new(Stmt::Return(e("x"))));
    assert!(render_stmt(&g, &lut("rust")).unwrap().contains("<T>"));
    let a = Stmt::AsyncFn("w".into(), vec![], Box::new(Stmt::Return(e("1"))));
    assert!(render_stmt(&a, &lut("rust")).unwrap().contains("async"));
    let aw = Stmt::Await(Box::new(Stmt::Return(e("1"))));
    assert!(render_stmt(&aw, &lut("rust")).unwrap().contains("await"));
}

#[test]
fn every_language_has_generic_async_layer() {
    for l in builtin_luts() {
        for k in ["generic_fn", "generic_type", "async_fn", "await"] {
            assert!(l.template(k).is_some(), "{} missing {k}", l.name);
        }
    }
}
