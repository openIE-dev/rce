use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::{builtin_luts, render_stmt, Stmt};
use competence_catalog::program::{eval_stmt, Env};

#[test]
fn attribute_is_declaration_metadata() {
    let prog = Stmt::Seq(vec![
        Stmt::Attribute("inline".into()),
        Stmt::Return(lift_expr("7").unwrap()),
    ]);
    assert_eq!(eval_stmt(&prog, &mut Env::new()).unwrap(), Some(7));
}

#[test]
fn macro_execution_refuses_open_frontier() {
    let prog = Stmt::Macro("vec".into(), vec![lift_expr("1").unwrap()]);
    assert!(eval_stmt(&prog, &mut Env::new()).is_err());
}

#[test]
fn attribute_and_macro_render_per_family() {
    let l = builtin_luts().into_iter().find(|l| l.name == "rust").unwrap();
    let a = Stmt::Attribute("inline".into());
    assert!(render_stmt(&a, &l).unwrap().contains("#[inline]"));
    let m = Stmt::Macro("vec".into(), vec![lift_expr("1").unwrap()]);
    assert!(render_stmt(&m, &l).unwrap().contains("vec!("));
}

#[test]
fn every_language_has_attribute_macro_layer() {
    for l in builtin_luts() {
        for k in ["attribute", "macro"] {
            assert!(l.template(k).is_some(), "{} missing {k}", l.name);
        }
    }
}
