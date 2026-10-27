//! W2b: `parse(unparse_source(m)) == m` up to source line numbers, on every
//! `.rce` file of the spec examples and the checker test corpora, and the
//! printed source is a fixpoint of print-then-parse.
use rce_spec_check::parse::parse;
use rce_spec_check::unparse::unparse_source;
use std::path::{Path, PathBuf};

fn files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut out = vec![];
    let mut stack = vec![root.parent().unwrap().join("examples"), root.join("tests"), root.parent().unwrap().join("conc06"), root.parent().unwrap().join("conc06b")];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rce") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// The AST's debug text with every source line number replaced by `_`.
pub fn shape(m: &rce_spec_check::ast::Module) -> String {
    let s = format!("{:?}", m.items);
    let re1 = regex::Regex::new(r"line: \d+").unwrap();
    let re2 = regex::Regex::new(r", \d+, (true|false)\)").unwrap();
    let re3 = regex::Regex::new(r"\b(Unit|None|SelfRef|Break|Continue)\(\d+\)").unwrap();
    let re4 = regex::Regex::new(r", \d+\)").unwrap();
    let s = re1.replace_all(&s, "line: _");
    let s = re2.replace_all(&s, ", _, $1)");
    let s = re3.replace_all(&s, "$1(_)");
    re4.replace_all(&s, ", _)").into_owned()
}

#[test]
fn every_module_round_trips_through_source() {
    let mut n = 0;
    let mut bad = vec![];
    for f in files() {
        let src = std::fs::read_to_string(&f).unwrap();
        let Ok(m) = parse(&src) else { continue };
        n += 1;
        let s1 = unparse_source(&m);
        match parse(&s1) {
            Ok(m2) => {
                if shape(&m) != shape(&m2) || m.pubs != m2.pubs || m.path != m2.path {
                    bad.push(format!("{}: AST differs", f.display()));
                } else if unparse_source(&m2) != s1 {
                    bad.push(format!("{}: not a fixpoint", f.display()));
                }
            }
            Err(e) => bad.push(format!("{}: {e}\n{s1}", f.display())),
        }
    }
    eprintln!("round-tripped {n} modules");
    assert!(n > 100, "only {n} modules parsed");
    assert!(bad.is_empty(), "{} of {n} failed:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn shape_sees_a_changed_literal_but_not_a_moved_line() {
    let a = parse("fn f() -> i64\n    return 1 + 2\n").unwrap();
    let b = parse("\n\nfn f() -> i64\n    return 1 + 2\n").unwrap();
    let c = parse("fn f() -> i64\n    return 1 + 3\n").unwrap();
    assert_eq!(shape(&a), shape(&b));
    assert_ne!(shape(&a), shape(&c));
}

/// Through the emitted Form: `rce emit form` then `rce unform` gives source
/// whose AST, verdict and observations equal the original's.
#[test]
fn emitted_form_terms_round_trip_and_check() {
    let mut n = 0;
    let mut bad = vec![];
    for f in files() {
        let src = std::fs::read_to_string(&f).unwrap();
        let Ok(m) = parse(&src) else { continue };
        let verdict = rce_spec_check::check_program(&[("t.rce".into(), src.clone())], false)[0].1.is_ok();
        if !verdict {
            continue;
        }
        let Ok(form) = rce_spec_check::cli::emit("form", &src) else {
            bad.push(format!("{}: emit form failed", f.display()));
            continue;
        };
        let v: serde_json::Value = serde_json::from_str(&form).unwrap();
        assert!(v["terms"]["items"].is_array(), "{}", f.display());
        let back = rce_spec_check::cli::unform(&form).unwrap();
        let m2 = parse(&back).unwrap();
        if shape(&m) != shape(&m2) {
            bad.push(format!("{}: terms differ", f.display()));
            continue;
        }
        let v2 = rce_spec_check::check_program(&[("t.rce".into(), back.clone())], false)[0].1.is_ok();
        if !v2 {
            bad.push(format!("{}: unformed source is refused", f.display()));
            continue;
        }
        // observations name `check` items by source line; lines are not terms
        let re = regex::Regex::new(r"^(check|example) \d+|line \d+").unwrap();
        let obs = |m: &rce_spec_check::ast::Module| -> Vec<String> { rce_spec_check::emit::observe(m).iter().map(|l| re.replace_all(l, "#").into_owned()).collect() };
        if obs(&m) != obs(&m2) {
            bad.push(format!("{}: observations differ", f.display()));
            continue;
        }
        n += 1;
    }
    eprintln!("form round trip: {n} admitted modules");
    assert!(n > 50, "only {n}");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn fn_main_form_is_not_a_skeleton() {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("examples/37_actor_spawn.rce")).unwrap();
    let form = rce_spec_check::cli::emit("form", &src).unwrap();
    let v: serde_json::Value = serde_json::from_str(&form).unwrap();
    assert_eq!(v["main"]["outputs"], serde_json::json!([7]));
    assert!(form.contains("\"Send\""), "fn main body terms present");
    assert!(form.len() > 1000);
}
