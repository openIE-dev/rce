//! The standard library table of 5.15 and the cost table of C.6.4 are the
//! reference evaluator's, and every emitted runtime charges the same
//! costs. A method outside the closed library is refused at check time.
use rce_spec_check::{cli, conc};
use regex::Regex;
use std::collections::{BTreeMap, BTreeSet};

fn read(p: &str) -> String {
    std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(p)).unwrap()
}

fn interp_methods() -> BTreeMap<String, BTreeSet<String>> {
    let s = read("src/interp.rs");
    let i = s.find("fn builtin(&mut self").unwrap();
    let j = i + s[i..].find("has no method {m} at comptime").unwrap();
    let arm = Regex::new(r#"(?m)^\s{12}\(\s*(?:mut\s+)?(\w+)\([^)]*\),\s*((?:\|?\s*"\w+"\s*)+),?\s*\)\s*(?:if [^=]*)?=>"#).unwrap();
    let name = Regex::new(r#""(\w+)""#).unwrap();
    let mut m: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for c in arm.captures_iter(&s[i..j]) {
        for n in name.captures_iter(&c[2]) {
            m.entry(c[1].to_string()).or_default().insert(n[1].to_string());
        }
    }
    m
}

#[test]
fn stdlib_table_is_the_evaluators() {
    let book = read("../book/06-regions-effects-extensions.md");
    let row = Regex::new(r"(?m)^\| `(\w+)` \| [^|]+ \| (.+) \|$").unwrap();
    let mut table: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for c in row.captures_iter(&book) {
        let ms = c[2].split(", ").map(|x| x.trim_matches('`').to_string()).collect();
        table.insert(c[1].to_string(), ms);
    }
    assert_eq!(table, interp_methods(), "5.15 table and interp::builtin differ");
}

#[test]
fn cost_table_is_the_evaluators_and_every_runtimes() {
    let book = read("../book/14-draft06-concurrency.md");
    let cost = |label: &str| -> i128 {
        let l = book.lines().find(|l| l.starts_with(&format!("| {label}"))).unwrap_or_else(|| panic!("no row {label}"));
        l.rsplit('|').nth(1).unwrap().trim().parse().unwrap()
    };
    assert_eq!(cost("`c_call`"), conc::C_CALL);
    assert_eq!(cost("`c_chan`"), conc::C_CHAN);
    assert_eq!(cost("`c_spawn`"), conc::C_SPAWN);
    assert_eq!(cost("`c_fork`"), conc::C_FORK);
    use rce_spec_check::interp::V;
    let (i, f) = (V::Int(1, "i64"), V::F(1.0));
    assert_eq!(cost("binary operator on integer"), conc::op_cost("+", &i));
    assert_eq!(cost("float (f64, f32) `+`"), conc::op_cost("*", &f));
    assert_eq!(cost("float (f64, f32) `/`"), conc::op_cost("%", &f));
    // The emitted runtimes use the same numbers at the same points.
    for (rt, n) in [("src/emit/rt.rs", ""), ("src/emit/rt.py", ""), ("src/emit/rt.js", "n")] {
        let s = read(rt);
        for c in [conc::C_CALL, conc::C_CHAN, conc::C_SPAWN] {
            assert!(s.contains(&format!("charge({c}{n})")), "{rt} never charges {c}");
        }
        for c in ["charge(1", "2", "8"] {
            assert!(s.contains(c), "{rt}: {c}");
        }
    }
    assert!(read("src/emit/conc06.rs").contains(&format!("{} * k", conc::C_FORK)));
}

#[test]
fn a_method_outside_the_library_is_refused_at_check_time() {
    let d = std::env::temp_dir().join(format!("rce-stdlib-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join("m.rce");
    std::fs::write(&p, "fn f(x: i64) -> i64\n    x.frobnicate()\n").unwrap();
    let a = vec!["check".to_string(), p.to_string_lossy().into_owned()];
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    let all = String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap();
    assert_ne!(code, 0);
    assert!(all.contains("W-NAME-1"), "{all}");
}
