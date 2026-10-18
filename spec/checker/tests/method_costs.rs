//! C.6.4 method costs track work: c_call plus the method's work on the
//! sizes of its receiver and first argument, by complexity class. The
//! budget boundary is the same operation in the reference evaluator and
//! in every emitted runtime (Rust, Python, JavaScript), and the cost is
//! monotone in input size. The size-scaled work is the one rule of the one
//! language (C.15): a `# rce-version:` label never changes a cost.
use rce_spec_check::conc::{C_CALL, ceil_log2, method_work, size_of};
use rce_spec_check::interp::V;
use rce_spec_check::cli;
use std::process::Command;

fn list(n: usize) -> V {
    V::List((0..n as i128).map(|i| V::Int(i, "i64")).collect())
}

/// (setup lines, the one call inside the budget, its cost by the table)
fn cases() -> Vec<(String, String, i128)> {
    let xs = "    var xs: List[i64] = [5, 3, 9, 1, 7, 2, 8, 4]\n".to_string();
    let t = "    let t: text = \"hello\"\n".to_string();
    let c = |r: V, m: &str, a: Vec<V>| C_CALL + method_work(&r, m, &a);
    vec![
        (xs.clone(), "xs.sort()".into(), c(list(8), "sort", vec![])),
        (xs.clone(), "let _k = xs.len()".into(), c(list(8), "len", vec![])),
        (xs.clone(), "let _b = xs.contains(3)".into(), c(list(8), "contains", vec![V::Int(3, "i64")])),
        (xs.clone(), "let _s = xs.sum()".into(), c(list(8), "sum", vec![])),
        ("    var ys: List[i64] = [1]\n".into(), "ys.sort()".into(), c(list(1), "sort", vec![])),
        ("    var ys: List[i64] = [3, 1, 2]\n".into(), "ys.sort()".into(), c(list(3), "sort", vec![])),
        (t.clone(), "let _u = t.upper()".into(), c(V::Text("hello".into()), "upper", vec![])),
        (t.clone(), "let _b = t.contains(\"ll\")".into(), c(V::Text("hello".into()), "contains", vec![V::Text("ll".into())])),
        (t.clone(), "let _k = t.len_bytes()".into(), c(V::Text("hello".into()), "len_bytes", vec![])),
        (t.clone(), "let _h = t.sha256()".into(), c(V::Text("hello".into()), "sha256", vec![])),
    ]
}

fn prog(setup: &str, call: &str, b: i128) -> String {
    format!("fn main()\n{setup}    with budget {b}\n        {call}\n    emit 1\n")
}

fn trapped(out: &str) -> bool {
    out.contains("budget")
}

fn rce_run(src: &str, stem: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-mcost-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.rce"));
    std::fs::write(&f, src).unwrap();
    let (mut out, mut err) = (vec![], vec![]);
    let a = vec!["run".to_string(), f.to_string_lossy().into_owned()];
    cli::main(&a, &mut &b""[..], &mut out, &mut err);
    String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap()
}

fn emitted(src: &str, stem: &str, target: &str) -> Option<String> {
    let d = std::env::temp_dir().join(format!("rce-mcost-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.rce"));
    std::fs::write(&f, src).unwrap();
    let (mut out, mut err) = (vec![], vec![]);
    let a = vec!["emit".to_string(), target.to_string(), f.to_string_lossy().into_owned()];
    assert_eq!(cli::main(&a, &mut &b""[..], &mut out, &mut err), 0, "{}", String::from_utf8_lossy(&err));
    let (ext, tool) = match target {
        "python" => ("py", "python3"),
        "js" => ("mjs", "node"),
        _ => ("rs", "rustc"),
    };
    if Command::new(tool).arg("--version").output().is_err() {
        return None;
    }
    let p = d.join(format!("{stem}.{ext}"));
    std::fs::write(&p, out).unwrap();
    let o = if target == "rust" {
        let exe = d.join(stem);
        let c = Command::new("rustc").args(["--edition", "2024", "-o"]).arg(&exe).arg(&p).output().unwrap();
        assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
        Command::new(&exe).arg("--run").output().unwrap()
    } else {
        Command::new(tool).arg(&p).arg("--run").output().unwrap()
    };
    Some(String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr))
}

#[test]
fn budget_boundary_is_the_table_cost_in_every_runtime() {
    for (i, (setup, call, cost)) in cases().into_iter().enumerate() {
        for (b, trap) in [(cost - 1, true), (cost, false)] {
            let src = prog(&setup, &call, b);
            let stem = format!("c{i}_{b}");
            assert_eq!(trapped(&rce_run(&src, &stem)), trap, "reference: `{call}` with budget {b} (cost {cost})");
            for t in ["rust", "python", "js"] {
                if let Some(out) = emitted(&src, &stem, t) {
                    assert_eq!(trapped(&out), trap, "{t}: `{call}` with budget {b} (cost {cost}): {out}");
                }
            }
        }
    }
}

#[test]
fn table_values_and_classes() {
    assert_eq!(ceil_log2(0), 0);
    assert_eq!(ceil_log2(1), 0);
    assert_eq!(ceil_log2(2), 1);
    assert_eq!(ceil_log2(8), 3);
    assert_eq!(ceil_log2(9), 4);
    assert_eq!(method_work(&list(8), "sort", &[]), 24);
    assert_eq!(method_work(&list(8), "len", &[]), 0);
    assert_eq!(method_work(&list(8), "map", &[]), 8);
    assert_eq!(method_work(&V::Text("héllo".into()), "upper", &[]), 6, "UTF-8 bytes");
    assert_eq!(size_of(&V::Int(5, "i64")), 0);
    // Overflow safe: a huge size saturates, never wraps.
    assert_eq!(i128::MAX.saturating_mul(ceil_log2(i128::MAX)), i128::MAX);
}

#[test]
fn cost_is_monotone_in_input_size() {
    let methods = ["len", "sort", "contains", "reverse", "sum", "map", "filter", "fold", "get", "pop", "upper", "split", "join", "keys", "insert"];
    for m in methods {
        for mk in [0usize, 1, 2] {
            let mut prev = -1i128;
            for n in 0..600usize {
                let r = match mk {
                    0 => list(n),
                    1 => V::Text("a".repeat(n).into()),
                    _ => V::Map((0..n as i128).map(|i| (V::Int(i, "i64"), V::Int(i, "i64"))).collect()),
                };
                for k in [0usize, 3] {
                    let a = vec![V::Text("b".repeat(k).into())];
                    let w = method_work(&r, m, &a);
                    assert!(w >= 0);
                    if k == 0 {
                        assert!(w >= prev, "{m} on kind {mk}: cost fell from {prev} to {w} at n = {n}");
                        prev = w;
                    } else {
                        assert!(w >= method_work(&r, m, &[V::Text(String::new().into())]), "{m}: cost fell as the argument grew");
                    }
                }
            }
        }
    }
}

#[test]
fn class_table_in_the_book_is_method_work() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../book/14-draft06-concurrency.md");
    let book = std::fs::read_to_string(p).unwrap();
    let five = |kind: &str| match kind {
        "text" => V::Text("abcde".into()),
        "list" => list(5),
        "set" => V::Set((0..5).map(|i| V::Int(i, "i64")).collect()), // a Set[τ] is its own value (V::Set)
        _ => V::Map((0..5).map(|i| (V::Int(i, "i64"), V::Int(i, "i64"))).collect()),
    };
    let mut rows = 0;
    for l in book.lines().filter(|l| l.starts_with("| linear, ") || l.starts_with("| n log n, ")) {
        let cols: Vec<&str> = l.split('|').map(str::trim).collect();
        let kind = ["text", "list", "map", "set"].into_iter().find(|k| cols[1].contains(&format!("{k} receiver"))).unwrap();
        for m in cols[2].split(", ").map(|m| m.trim_matches('`')) {
            let want = if cols[1].starts_with("n log n") { 5 * 3 } else { 5 };
            assert_eq!(method_work(&five(kind), m, &[]), want, "{kind}.{m}");
        }
        rows += 1;
    }
    assert_eq!(rows, 5); // text, list, map, set linear rows and the n log n list row
    // Constant class: a few methods outside the table cost nothing extra.
    for (k, m) in [("list", "len"), ("list", "pop"), ("text", "len"), ("map", "len"), ("list", "get")] {
        assert_eq!(method_work(&five(k), m, &[]), 0, "{k}.{m}");
    }
}

/// BUGS.md RCE-004: `x op= y` charges the operator's cost (C.6.4) like
/// `x = x op y`, in the reference and every runtime. Before the fix a
/// compound assignment cost 0, so a loop of `+=` ran unbounded inside a budget.
#[test]
fn compound_assignment_charges_the_operator_in_every_runtime() {
    // int `+=` costs 1 (ten of them plus the range: 11); float `/=` costs 8
    let cases = [
        ("    var x = 0\n", "for _i in 0..10\n            x += 1", 11, "x"),
        ("    var y = 1.0\n", "y /= 3.0", 8, "y"),
    ];
    for (i, (setup, call, cost, var)) in cases.into_iter().enumerate() {
        for (b, trap) in [(cost - 1, true), (cost, false)] {
            let src = prog(setup, call, b) + &format!("    emit {var}\n");
            let stem = format!("ca{i}_{b}");
            assert_eq!(trapped(&rce_run(&src, &stem)), trap, "reference: `{call}` budget {b}");
            for t in ["rust", "python", "js"] {
                if let Some(out) = emitted(&src, &stem, t) {
                    assert_eq!(trapped(&out), trap, "{t}: `{call}` budget {b}: {out}");
                }
            }
        }
    }
}
