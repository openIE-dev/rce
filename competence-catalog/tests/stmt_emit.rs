use competence_catalog::lift::lift_expr;
use competence_catalog::lut_emit::{builtin_luts, render_fn, render_stmt, Fn, LanguageLut, Stmt};

fn lut(name: &str) -> LanguageLut {
    builtin_luts().into_iter().find(|l| l.name == name).unwrap()
}

#[test]
fn c_family_let_renders() {
    let s = Stmt::Let("x".into(), lift_expr("1+2").unwrap());
    let out = render_stmt(&s, &lut("rust")).unwrap();
    assert!(out.contains("let x ="), "{out}");
}

#[test]
fn lisp_for_renders() {
    let s = Stmt::For("i".into(), lift_expr("1").unwrap(), lift_expr("6").unwrap(),
        Box::new(Stmt::Return(lift_expr("i").unwrap())));
    let out = render_stmt(&s, &lut("clojure")).unwrap();
    assert!(out.contains("loop for"), "{out}");
}

#[test]
fn ml_while_renders() {
    let s = Stmt::While(lift_expr("1").unwrap(), Box::new(Stmt::Return(lift_expr("0").unwrap())));
    let out = render_stmt(&s, &lut("ocaml")).unwrap();
    assert!(out.contains("while") && out.contains("done"), "{out}");
}

#[test]
fn fn_renders_with_params() {
    let f = Fn {
        name: "add".into(),
        params: vec![("a".into(), "i64".into()), ("b".into(), "i64".into())],
        ret: "i64".into(),
        body: Stmt::Return(lift_expr("a+b").unwrap()),
    };
    let out = render_fn(&f, &lut("rust")).unwrap();
    assert!(out.contains("fn add("), "{out}");
}

#[test]
fn every_language_has_the_statement_layer() {
    for l in builtin_luts() {
        for k in ["let", "assign", "return", "if_stmt", "while", "for", "import", "type_def", "fn"] {
            assert!(l.template(k).is_some(), "{} missing {k}", l.name);
        }
    }
}
