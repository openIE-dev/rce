//! `rce index` / `rce search` and the D.7 lowering of a module function to a
//! Form IR term. The vectors in tests/brick/vectors.json are checked here
//! against the reference interpreter (as `check` lines of the fixture) and,
//! in ../../tests/brick_lowering.rs, against the Form IR evaluator on the
//! terms of tests/brick/catalog.json, which this test keeps equal to
//! `rce index --vectors 8` of the fixtures (set RCE_BLESS=1 to rewrite it);
//! the engine also runs every term on the generated vectors there.
use rce_spec_check::{brick, check_one};
use serde_json::{Value, json};

const FIX: &str = include_str!("brick/bricks.rce");

fn vectors() -> Vec<Value> {
    serde_json::from_str(include_str!("brick/vectors.json")).unwrap()
}

fn catalog() -> Value {
    brick::index(&[("bricks.rce".into(), FIX.into())])
}

fn entry<'a>(c: &'a Value, n: &str) -> &'a Value {
    c["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["name"] == n)
        .unwrap()
}

fn lit(v: &Value) -> String {
    match v {
        Value::String(s) => format!("{s:?}"),
        _ => v.to_string(),
    }
}

#[test]
fn vectors_hold_in_the_reference_interpreter() {
    let c = catalog();
    let mut src = FIX.to_string();
    for (k, v) in vectors().iter().enumerate() {
        let f = v[0].as_str().unwrap();
        let e = entry(&c, f);
        // a borrowed parameter needs a place: bind the literal first
        let mut args = vec![];
        for (i, (a, p)) in v[1]
            .as_array()
            .unwrap()
            .iter()
            .zip(e["params"].as_array().unwrap())
            .enumerate()
        {
            let ty = p[1].as_str().unwrap();
            if let Some(inner) = ty.strip_prefix('&') {
                let n = format!("ARG{k}_{i}");
                src.push_str(&format!("\nconst {n}: {inner} = {}\n", lit(a)));
                args.push(format!("&{n}"));
            } else {
                args.push(lit(a));
            }
        }
        let call = format!("{f}({})", args.join(", "));
        let fails = e["ret"].as_str().unwrap().contains(" ! ");
        match v[2].get("$err").and_then(|e| e.as_object()) {
            Some(err) => {
                for (k, x) in err {
                    src.push_str(&format!("\ncheck ({call} catch |e| e.{k}) == {}\n", lit(x)));
                }
            }
            None if fails => src.push_str(&format!(
                "\ncheck ({call} catch |_e| {} + 1) == {}\n",
                lit(&v[2]),
                lit(&v[2])
            )),
            None => src.push_str(&format!("\ncheck {call} == {}\n", lit(&v[2]))),
        }
    }
    check_one(&src).unwrap();
}

#[test]
fn index_is_canonical_and_layout_free() {
    let a = brick::canon(&catalog());
    assert_eq!(a, brick::canon(&catalog()));
    // moving functions down and adding comments changes lines, not hashes
    let moved = FIX.replace("const LIMIT", "# a comment\n\n\nconst LIMIT");
    let b = brick::index(&[("bricks.rce".into(), moved)]);
    let c = catalog();
    for (x, y) in b["entries"]
        .as_array()
        .unwrap()
        .iter()
        .zip(c["entries"].as_array().unwrap())
    {
        assert_eq!(x["hash"], y["hash"]);
        assert_ne!(x["line"], y["line"]);
    }
    // a body change changes that hash and the catalog hash
    let edited = FIX.replace("(a + b) / 2", "(a + b) / 3");
    let d = brick::index(&[("bricks.rce".into(), edited)]);
    assert_ne!(entry(&d, "mean2")["hash"], entry(&c, "mean2")["hash"]);
    assert_eq!(entry(&d, "clamp")["hash"], entry(&c, "clamp")["hash"]);
    assert_ne!(d["hash"], c["hash"]);
}

#[test]
fn index_records() {
    let c = catalog();
    assert_eq!(c["format"], "rce-catalog/1");
    // `helper` is not pub, and the module has pub functions
    let names: Vec<&str> = c["entries"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "area",
            "bag_add",
            "between",
            "capped_sum",
            "clamp",
            "count_down",
            "digit",
            "digit_pt",
            "digits_or",
            "doubled",
            "fact",
            "find_first",
            "first",
            "first_or",
            "greet",
            "is_even",
            "is_odd",
            "is_short",
            "last_index",
            "le16",
            "mean2",
            "mid_len",
            "mix",
            "popcount",
            "put2",
            "put4",
            "rev",
            "safe_div",
            "shift",
            "sign",
            "sum_list",
            "swap_at",
            "tail2",
            "take2",
            "take_last",
            "trimmed",
            "two_digits",
            "u64_mix",
            "u8_mix"
        ]
    );
    let f = entry(&c, "first");
    assert_eq!(f["key"], "(List[t0], t0) -> t0");
    assert_eq!(f["checked"], true);
    assert_eq!(entry(&c, "clamp")["key"], "(i64, i64, i64) -> i64");
    // recursion lowers to a fix (with `bound 64` counted in the term)
    assert_eq!(entry(&c, "fact")["lowering"]["term"]["t"], "fix");
    // (a term with shared subterms is format 2: `main` and its `defs`)
    let ie = &entry(&c, "is_even")["lowering"]["term"];
    assert_eq!(if ie["main"].is_object() { &ie["main"]["t"] } else { &ie["t"] }, "fix");
    // loops, lists, structs, text `+`, bit operations, generics and usize lower
    assert_eq!(c["lowered"], 39);
}

#[test]
fn search_by_type() {
    let c = catalog();
    let s = |q: &str| -> Vec<String> {
        brick::search(&c, q)
            .unwrap()
            .into_iter()
            .map(|(r, _, n, _)| format!("{r} {n}"))
            .collect()
    };
    assert_eq!(
        s("(i64, i64) -> i64"),
        ["0 area", "0 capped_sum", "0 mean2", "0 mix", "0 safe_div"]
    );
    assert_eq!(s("(i64, i64, i64) -> bool"), ["0 between"]);
    assert_eq!(s("(Pt, i64) -> Pt"), ["0 shift"]);
    // a variable query is an instance of nothing here, a concrete query is an instance of `first`
    assert_eq!(s("(List[i64], i64) -> i64"), ["0 find_first", "0 first_or", "1 first"]);
    assert_eq!(s("(List[a], a) -> a"), ["0 first", "1 find_first", "1 first_or"]);
    // argument order is found, at a cost
    assert_eq!(s("(i64, List[i64]) -> i64"), ["2 find_first", "2 first_or", "3 first"]);
    assert_eq!(s("text -> bool"), ["0 is_short"]);
    assert!(s("text -> i64").is_empty());
    assert!(brick::search(&c, "i64").is_err());
}

#[test]
fn catalog_with_vectors_matches_golden() {
    let mut files = vec![(dir("bricks.rce"), FIX.to_string())];
    files.extend(dep_files());
    let c = brick::index_with(&files, &brick::Opts { vectors: 8 });
    // paths in the golden file are relative to the fixture directory
    let now = brick::canon(&c).replace(&dir(""), "") + "\n";
    let p = dir("catalog.json");
    if std::env::var("RCE_BLESS").is_ok() {
        std::fs::write(&p, &now).unwrap();
    }
    assert_eq!(std::fs::read_to_string(&p).unwrap(), now);
    // every lowered brick has vectors, and some vectors trap
    let vs = c["vectors"].as_object().unwrap();
    for e in c["entries"].as_array().unwrap() {
        if e["lowering"].get("term").is_some() {
            assert!(
                vs.contains_key(e["hash"].as_str().unwrap()),
                "{} has no vectors",
                e["name"]
            );
        }
    }
    assert!(
        vs.values()
            .flat_map(|v| v.as_array().unwrap())
            .any(|v| v.get("trap").is_some())
    );
}

#[test]
fn cli_index_and_search() {
    use rce_spec_check::cli;
    let run = |a: &[&str]| {
        let a: Vec<String> = a.iter().map(|s| s.to_string()).collect();
        let (mut o, mut e) = (vec![], vec![]);
        let code = cli::main(&a, &mut &b""[..], &mut o, &mut e);
        (code, String::from_utf8(o).unwrap())
    };
    let fix = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/brick/bricks.rce");
    let (code, out) = run(&["index", fix]);
    assert_eq!(code, 0);
    let p = std::env::temp_dir().join(format!("rce-brick-{}.json", std::process::id()));
    std::fs::write(&p, &out).unwrap();
    let (code, out) = run(&["search", p.to_str().unwrap(), "(i64, i64, i64) -> bool"]);
    assert_eq!(
        (code, out.as_str()),
        (0, "0 demo.bricks.between : (i64, i64, i64) -> bool\n")
    );
}

fn dir(p: &str) -> String {
    format!("{}/tests/brick/{p}", env!("CARGO_MANIFEST_DIR"))
}

fn dep_files() -> Vec<(String, String)> {
    ["dep/ddir.rce", "dep/dlib.rce", "dep/duser.rce"]
        .iter()
        .map(|f| (dir(f), std::fs::read_to_string(dir(f)).unwrap()))
        .collect()
}

#[test]
fn library_dependencies_resolve_by_use() {
    let c = brick::index(&dep_files());
    assert_eq!(c["pending_files"], json!({}));
    assert!(
        c["entries"]
            .as_array()
            .unwrap()
            .iter()
            .all(|e| e["checked"] == true && e["lowering"].get("term").is_some())
    );
    // without dlib, the module duser imports is not there and duser is refused
    let alone = brick::index(&[(
        dir("dep/duser.rce"),
        std::fs::read_to_string(dir("dep/duser.rce")).unwrap(),
    )]);
    assert!(
        alone["pending_files"][dir("dep/duser.rce")]
            .as_str()
            .unwrap()
            .contains("unknown module `dlib`")
    );
}

#[test]
fn a_lib_line_is_a_comment_to_the_index() {
    // `#!lib` was retired in favour of `use`: the line places nothing in
    // front of the module, so a bare name of dlib stays unknown
    let src = "module dold\n#!lib dlib\n\nfn six(x: i64) -> i64 = triple(x) * 2\n";
    let mut files = dep_files();
    files.push((dir("dep/dold.rce"), src.to_string()));
    let c = brick::index(&files);
    assert!(
        c["pending_files"][dir("dep/dold.rce")]
            .as_str()
            .unwrap()
            .contains("`triple`")
    );
    assert_eq!(c["pending_files"].as_object().unwrap().len(), 1);
}
