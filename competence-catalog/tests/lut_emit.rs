//! LUT-driven emission falsifiers: emitting to many languages is a lookup table
//! + a deterministic operation, not hand-written emitters. Adding a language is
//! adding data; same IR + same LUT → byte-identical output.

use competence_catalog::lut_emit::{builtin_luts, emit_expr, render_fn, render_stmt, stmt_luts, Fn, LanguageLut, Stmt};
use rce_spec_check::ast::Expr;

fn expr(s: &str) -> Expr {
    let toks = rce_spec_check::parse::lex(s).unwrap();
    let mut p = rce_spec_check::parse::Parser::new(toks);
    p.expr().unwrap()
}

#[test]
fn emits_to_all_builtin_languages() {
    let luts = builtin_luts();
    assert!(luts.len() >= 45, "broad language coverage from data tables");
    for lut in &luts {
        // Every target renders the fragment; syntax is data, so families differ.
        let out = emit_expr("(a + b) * c", lut).unwrap_or_else(|e| panic!("{}: {e}", lut.name));
        assert!(!out.is_empty(), "{} rendered empty", lut.name);
    }
    // Infix families render infix; the Lisp family renders prefix — driven by the LUT.
    let get = |n: &str| luts.iter().find(|l| l.name == n).unwrap();
    assert_eq!(emit_expr("(a + b) * c", get("rust")).unwrap(), "((a + b) * c)");
    assert_eq!(emit_expr("(a + b) * c", get("clojure")).unwrap(), "(* (+ a b) c)");
}

#[test]
fn language_specific_syntax_is_data() {
    let luts = builtin_luts();
    let get = |n: &str| luts.iter().find(|l| l.name == n).unwrap();
    // Logical AND differs by language — driven by the LUT, not code.
    assert_eq!(emit_expr("a and b", get("rust")).unwrap(), "(a && b)");
    assert_eq!(emit_expr("a and b", get("python")).unwrap(), "(a and b)");
    assert_eq!(emit_expr("a and b", get("zig")).unwrap(), "(a and b)");
    // Haskell uses /= and div.
    assert_eq!(emit_expr("a != b", get("haskell")).unwrap(), "(a /= b)");
    assert_eq!(emit_expr("a / b", get("haskell")).unwrap(), "(div a b)");
    // Python conditional expression order differs.
    assert_eq!(emit_expr("if a then b else c", get("python")).unwrap(), "(b if a else c)");
    assert_eq!(emit_expr("if a then b else c", get("rust")).unwrap(), "(a ? b : c)");
}

#[test]
fn emission_is_deterministic() {
    let luts = builtin_luts();
    let lut = &luts[0];
    let a = emit_expr("(a + b) * (c - d)", lut).unwrap();
    let b = emit_expr("(a + b) * (c - d)", lut).unwrap();
    assert_eq!(a, b, "same IR + same LUT → byte-identical");
}

#[test]
fn adding_a_language_is_adding_data() {
    // A brand-new language is just a new LUT — no emitter code written.
    let lol = LanguageLut::new("lolcode", &[
        ("int_lit", "{value}"), ("var", "{name}"),
        ("add", "SUM OF {a} AN {b}"), ("mul", "PRODUCT OF {a} AN {b}"),
    ]);
    assert_eq!(emit_expr("a + b", &lol).unwrap(), "SUM OF a AN b");
    assert_eq!(emit_expr("a * b", &lol).unwrap(), "PRODUCT OF a AN b");
}

#[test]
fn whole_program_emission_across_languages() {
    // let x = a + b; if x > 0 then return x else return 0 — same IR, many targets.
    let prog = Stmt::Seq(vec![
        Stmt::Let("x".into(), expr("a + b")),
        Stmt::If(expr("x > 0"),
            Box::new(Stmt::Return(expr("x"))),
            Box::new(Stmt::Return(expr("0")))),
    ]);
    let luts = stmt_luts();
    let get = |n: &str| luts.iter().find(|l| l.name == n).unwrap();

    let py = render_stmt(&prog, get("python")).unwrap();
    assert!(py.contains("x = (a + b)"), "{py}");
    assert!(py.contains("if (x > 0):"), "{py}");

    let rs = render_stmt(&prog, get("rust")).unwrap();
    assert!(rs.contains("let x = (a + b);"), "{rs}");
    assert!(rs.contains("if (x > 0) { return x; } else { return 0; }"), "{rs}");

    let go = render_stmt(&prog, get("go")).unwrap();
    assert!(go.contains("x := (a + b)"), "{go}");

    let js = render_stmt(&prog, get("js")).unwrap();
    assert!(js.contains("let x = (a + b);"), "{js}");
    assert!(js.contains("if ((x > 0)) { return x; } else { return 0; }"), "{js}");
}

#[test]
fn function_and_loop_emission() {
    let luts = stmt_luts();
    let get = |n: &str| luts.iter().find(|l| l.name == n).unwrap();
    let f = Fn {
        name: "add".into(),
        params: vec![("a".into(), "i64".into()), ("b".into(), "i64".into())],
        ret: "i64".into(),
        body: Stmt::Return(expr("a + b")),
    };
    assert!(render_fn(&f, get("python")).unwrap().contains("def add(a, b):"));
    assert!(render_fn(&f, get("rust")).unwrap().contains("fn add(a: i64, b: i64) -> i64"));
    assert!(render_fn(&f, get("go")).unwrap().contains("func add(a i64, b i64) i64"));
    assert!(render_fn(&f, get("c")).unwrap().contains("i64 add(i64 a, i64 b)"));
    // Loops: Go uses `for`, the C-family uses `while` — driven by the LUT.
    let w = Stmt::While(expr("x > 0"), Box::new(Stmt::Return(expr("x"))));
    assert!(render_stmt(&w, get("go")).unwrap().contains("for (x > 0)"));
    assert!(render_stmt(&w, get("rust")).unwrap().contains("while (x > 0)"));
    assert!(render_stmt(&w, get("python")).unwrap().contains("while (x > 0):"));
}
