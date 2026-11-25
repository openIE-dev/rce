use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::Stmt;
use competence_catalog::program::{eval_program, expand_macro, Env, FnTable};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

#[test]
fn macro_expands_deterministically_and_executes() {
    let mut t = FnTable::new();
    t.add_macro("double", 1, "({0} + {0})");
    t.add_macro("sum3", 3, "({0} + {1} + {2})");
    let prog = Stmt::Seq(vec![
        Stmt::Macro("double".into(), vec![e("21")]),
    ]);
    // Macro yields its expanded value as the statement result.
    assert_eq!(eval_program(&prog, &mut Env::new(), &mut t).unwrap(), Some(42));
    let prog3 = Stmt::Macro("sum3".into(), vec![e("1"), e("2"), e("3")]);
    assert_eq!(eval_program(&prog3, &mut Env::new(), &mut t).unwrap(), Some(6));
}

#[test]
fn unknown_macro_and_arity_mismatch_decline() {
    let mut t = FnTable::new();
    t.add_macro("double", 1, "({0} + {0})");
    assert!(eval_program(&Stmt::Macro("nope".into(), vec![e("1")]), &mut Env::new(), &mut t).is_err());
    assert!(eval_program(&Stmt::Macro("double".into(), vec![e("1"), e("2")]), &mut Env::new(), &mut t).is_err());
}

#[test]
fn expand_macro_is_substitution_plus_lift() {
    let mut macros = std::collections::BTreeMap::new();
    macros.insert("double".to_string(), (1usize, "({0} + {0})".to_string()));
    let a = expand_macro("double", &[e("3")], &macros).unwrap();
    let b = expand_macro("double", &[e("3")], &macros).unwrap();
    assert_eq!(competence_catalog::canonical::emit(&a), competence_catalog::canonical::emit(&b));
}
