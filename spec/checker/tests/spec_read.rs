//! The checker reads its method tables from spec-data.json (`specread`):
//! every method 5.15 lists is known to the type checker on its receiver,
//! a method it does not list is W-NAME-1, and `conc::method_work` follows
//! the C.6.4 classes of the same file.
use rce_spec_check::conc::method_work;
use rce_spec_check::interp::V;
use rce_spec_check::specread::{TABLES, Work, work};

fn diags(src: &str) -> String {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    match &d[0].1 {
        Ok(()) => String::new(),
        Err(e) => e.clone(),
    }
}

fn use_method(recv: &str, m: &str) -> String {
    let (ty, pre) = match recv {
        "Int" => ("i64", ""),
        "F" => ("f64", ""),
        "Bool" => ("bool", ""),
        "Text" => ("text", ""),
        "F32" => ("f32", ""),
        "List" => ("List[i64]", ""),
        "Map" => ("Map[text, i64]", ""),
        "Set" => ("Set[i64]", ""),
        "Opt" => ("i64?", ""),
        "Gen" => return format!("fn g() -> iter i64\n    yield 1\n\nfn f() -> i64\n    let y = g().{m}()\n    return 0\n"),
        r => panic!("receiver {r} has no sample type"),
    };
    format!("{pre}fn f(x: {ty}) -> i64\n    let y = x.{m}()\n    return 0\n")
}

#[test]
fn every_listed_method_is_known_and_others_are_not() {
    let mut n = 0;
    for (recv, ms) in &TABLES.methods {
        for m in ms {
            // `raw` exists only in a 0.8 module on `Text[C]` (5.15, 5.16);
            // `tests/indexed.rs` covers it.
            if recv == "Text" && m == "raw" {
                continue;
            }
            let e = diags(&use_method(recv, m));
            assert!(!e.contains("has no method"), "{recv}.{m}: {e}");
            n += 1;
        }
    }
    assert!(n > 0);
    for (recv, m) in [("F", "pow"), ("Text", "format"), ("List", "extend"), ("Int", "sqrt"), ("Map", "sort")] {
        let e = diags(&use_method(recv, m));
        assert!(e.contains("W-NAME-1") && e.contains("has no method"), "{recv}.{m}: {e}");
    }
}

#[test]
fn method_work_follows_the_classes() {
    let list = V::List((0..8).map(|i| V::Int(i, "i64")).collect());
    let text = V::Text("abcdefgh".into());
    for ((recv, m), w) in &TABLES.work {
        let r = match recv.as_str() {
            // a Set[τ] value is a list in canonical order
            "list" | "set" => list.clone(),
            "text" => text.clone(),
            "map" => continue,
            x => panic!("{x}"),
        };
        let got = method_work(&r, m, &[]);
        let want = match w {
            Work::Linear => 8,
            Work::NLogN => 24,
            Work::Constant => 0,
        };
        assert_eq!(got, want, "{recv}.{m}");
    }
    assert_eq!(work("list", "len"), Work::Constant);
    assert_eq!(method_work(&list, "len", &[]), 0);
}
