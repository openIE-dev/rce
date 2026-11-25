use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::{builtin_luts, render_stmt, LanguageLut, Pattern, Stmt};
use competence_catalog::program::{eval_program, eval_stmt, Env, FnTable};

fn e(s: &str) -> rce_spec_check::ast::Expr {
    lift_expr(s).unwrap()
}

#[test]
fn match_literal_arm_wins() {
    let s = Stmt::Match(
        e("2"),
        vec![
            (Pattern::Lit(1), Box::new(Stmt::Return(e("10")))),
            (Pattern::Lit(2), Box::new(Stmt::Return(e("20")))),
            (Pattern::Wild, Box::new(Stmt::Return(e("30")))),
        ],
    );
    assert_eq!(eval_stmt(&s, &mut Env::new()).unwrap(), Some(20));
}

#[test]
fn match_wildcard_fallback() {
    let s = Stmt::Match(
        e("9"),
        vec![
            (Pattern::Lit(1), Box::new(Stmt::Return(e("10")))),
            (Pattern::Wild, Box::new(Stmt::Return(e("30")))),
        ],
    );
    assert_eq!(eval_stmt(&s, &mut Env::new()).unwrap(), Some(30));
}

#[test]
fn match_bind_captures_scrutinee() {
    let s = Stmt::Match(
        e("7"),
        vec![(Pattern::Bind("x".into()), Box::new(Stmt::Return(e("x + 1"))))],
    );
    assert_eq!(eval_stmt(&s, &mut Env::new()).unwrap(), Some(8));
}

#[test]
fn local_fn_recursion_under_fuel() {
    // fact(n) = if n < 2 { return 1; } else { let r = fact(n-1); return n * r; }
    let prog = Stmt::Seq(vec![
        Stmt::LetFn("fact".into(), vec!["n".into()], Box::new(Stmt::If(
            e("n < 2"),
            Box::new(Stmt::Return(e("1"))),
            Box::new(Stmt::Seq(vec![
                Stmt::LetCall("r".into(), "fact".into(), vec![e("n - 1")]),
                Stmt::Return(e("n * r")),
            ])),
        ))),
        Stmt::LetCall("out".into(), "fact".into(), vec![e("5")]),
        Stmt::Return(e("out")),
    ]);
    let mut env = Env::new();
    let mut fns = FnTable::new();
    let r = eval_program(&prog, &mut env, &mut fns).unwrap();
    assert_eq!(r, Some(120));
}

#[test]
fn call_arity_mismatch_refused() {
    let prog = Stmt::Seq(vec![
        Stmt::LetFn("f".into(), vec!["a".into()], Box::new(Stmt::Return(e("a")))),
        Stmt::Call("f".into(), vec![e("1"), e("2")]),
    ]);
    assert!(eval_program(&prog, &mut Env::new(), &mut FnTable::new()).is_err());
}

fn lut(name: &str) -> LanguageLut {
    builtin_luts().into_iter().find(|l| l.name == name).unwrap()
}

#[test]
fn match_renders_per_family() {
    let s = Stmt::Match(e("x"), vec![(Pattern::Wild, Box::new(Stmt::Return(e("1"))))]);
    assert!(render_stmt(&s, &lut("rust")).unwrap().contains("match"));
    assert!(render_stmt(&s, &lut("ocaml")).unwrap().contains("match"));
    assert!(render_stmt(&s, &lut("clojure")).unwrap().contains("case"));
}

#[test]
fn every_language_has_match_layer() {
    for l in builtin_luts() {
        for k in ["match", "arm", "letfn", "pat_wild", "pat_lit", "pat_bind"] {
            assert!(l.template(k).is_some(), "{} missing {k}", l.name);
        }
    }
}
