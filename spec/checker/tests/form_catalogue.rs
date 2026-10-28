//! The catalogue of `rce-form/1` as data (`spec/proposals/rce-form.md` §4):
//! `spec/data/form-catalogue.json`, format `rce-catalogue/1`. Until a chapter
//! of the book defines `rce-form/1` the file is written by hand, and this test
//! holds it to the proposal: the bytes are canonical JSON (6.1.1); the ids are
//! unique, in ascending order, and exactly the list of §4; the partial
//! entries are exactly the partial primitives of W4, and each names the
//! domain fact that §7 gives it and tests from the file's closed list; every
//! field draws on the closed vocabulary the file's `terms` define, pattern
//! heads included; and every clause names a section of the book. The entries' vectors come
//! from the Form evaluator, which does not exist yet, so none is checked here.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

fn spec() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn bytes() -> String {
    std::fs::read_to_string(spec().join("data/form-catalogue.json")).expect("spec/data/form-catalogue.json")
}

fn catalogue() -> Value {
    serde_json::from_str(&bytes()).expect("spec/data/form-catalogue.json is JSON")
}

fn entries(c: &Value) -> &Vec<Value> {
    c["entries"].as_array().expect("entries is a list")
}

fn ids(c: &Value) -> Vec<String> {
    entries(c).iter().map(|e| e["id"].as_str().expect("id is text").to_string()).collect()
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().map(|m| m.keys().cloned().collect()).unwrap_or_default()
}

fn set(xs: &[&str]) -> BTreeSet<String> {
    xs.iter().map(|x| x.to_string()).collect()
}

/// Canonical JSON (6.1.1) of the values the catalogue holds, written from
/// the text of 6.1.1: no whitespace, record fields in ascending order of key
/// by code point, integers in decimal, text with the escapes of rule 4. The
/// catalogue holds no float; one is an error.
fn canon(v: &Value, out: &mut String) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            assert!(n.is_i64() || n.is_u64(), "the catalogue holds a number that is not an integer: {n}");
            out.push_str(&n.to_string());
        }
        Value::String(s) => text(s, out),
        Value::Array(xs) => {
            out.push('[');
            for (i, x) in xs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                canon(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            // sorted here, whatever order the parser keeps (UTF-8 byte order is code-point order)
            let sorted: BTreeMap<&String, &Value> = m.iter().collect();
            out.push('{');
            for (i, (k, x)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                text(k, out);
                out.push(':');
                canon(x, out);
            }
            out.push('}');
        }
    }
}

fn text(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[test]
fn the_file_is_canonical_json() {
    let have = bytes();
    let mut want = String::new();
    canon(&catalogue(), &mut want);
    assert!(
        have == want,
        "spec/data/form-catalogue.json is not canonical JSON (6.1.1): no whitespace, no line end, fields in order of key, \
         and no key twice; first difference at byte {}",
        have.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count()
    );
}

#[test]
fn the_header_is_closed() {
    let c = catalogue();
    assert_eq!(c["format"], "rce-catalogue/1");
    assert_eq!(keys(&c), set(&["derived", "entries", "excluded", "format", "proposal", "status", "terms", "vectors"]));
    // the four primitives of rce-core/1 that no Form names (§4)
    let ex = keys(&c["excluded"]);
    assert_eq!(ex, set(&["drop", "io.byte", "mem.get", "mem.set"]));
    let ids: BTreeSet<String> = ids(&c).into_iter().collect();
    for x in &ex {
        assert!(!ids.contains(x), "{x} is excluded and also an entry");
    }
}

#[test]
fn ids_are_unique_and_sorted() {
    let ids = ids(&catalogue());
    for w in ids.windows(2) {
        assert!(w[0] < w[1], "ids out of order or repeated: {} then {}", w[0], w[1]);
    }
}

/// The ids that §4 of the proposal lists, from the block after the line that
/// names the entries of `rce-catalogue/1`: one line per family, `family (n): ids`.
fn proposal_ids() -> Vec<String> {
    let p = std::fs::read_to_string(spec().join("proposals/rce-form.md")).unwrap();
    let mut lines = p.lines().skip_while(|l| !l.contains("The entries of `rce-catalogue/1`"));
    assert!(lines.next().is_some(), "rce-form.md §4 does not list the entries of `rce-catalogue/1`");
    let mut lines = lines.skip_while(|l| !l.starts_with("```"));
    assert!(lines.next().is_some(), "rce-form.md: the list of entries is not in a fenced block");
    let mut out = vec![];
    for l in lines.take_while(|l| !l.starts_with("```")) {
        let (fam, list) = l.split_once(':').unwrap_or_else(|| panic!("rce-form.md: a line of the entry list is not `family (n): ids`: {l}"));
        let n: usize = fam
            .split_once('(')
            .and_then(|(_, r)| r.trim_end_matches(')').parse().ok())
            .unwrap_or_else(|| panic!("rce-form.md: the family `{fam}` has no count"));
        let these: Vec<String> = list.split_whitespace().map(String::from).collect();
        assert_eq!(these.len(), n, "rce-form.md: the family `{fam}` lists {} entries", these.len());
        out.extend(these);
    }
    out
}

#[test]
fn ids_are_the_proposals_list() {
    let p = proposal_ids();
    let ps: BTreeSet<String> = p.iter().cloned().collect();
    assert_eq!(ps.len(), p.len(), "rce-form.md lists an entry twice");
    let fs: BTreeSet<String> = ids(&catalogue()).into_iter().collect();
    let missing: Vec<_> = ps.difference(&fs).collect();
    let extra: Vec<_> = fs.difference(&ps).collect();
    assert!(missing.is_empty() && extra.is_empty(), "listed in rce-form.md but not in the file: {missing:?}; in the file but not listed: {extra:?}");
}

/// The partial primitives of W4 (`one/wf.rs`, `domain`, on the branch
/// `claude/one-emitter` at 123f5eee) other than `mem.get` and `mem.set`, which
/// the catalogue leaves out: the entries that §7 of the proposal covers by a
/// domain fact. Written out here rather than read from §7, whose lines also
/// name total entries (`bytes.len`, `list.len` and `arr.len` under DF5).
const PARTIAL: [&str; 30] = [
    "int.add", "int.sub", "int.mul", "int.neg", "int.abs", "int.cast", "int.shl", "int.shr", "int.quo", "int.rem",
    "f64.add", "f64.sub", "f64.mul", "f64.div", "f64.sqrt", "f64.trunc", "f64.to_f32",
    "f32.add", "f32.sub", "f32.mul", "f32.div",
    "bytes.get", "bytes.slice", "list.get", "arr.get", "list.set", "arr.set", "list.pop", "sum.get", "arr.new",
];

/// The line of §7 of the proposal that states each domain fact.
fn fact_lines() -> BTreeMap<String, String> {
    let p = std::fs::read_to_string(spec().join("proposals/rce-form.md")).unwrap();
    let mut out = BTreeMap::new();
    for l in p.lines() {
        if let Some(r) = l.strip_prefix("- **DF")
            && let Some((n, _)) = r.split_once(' ')
        {
            out.insert(format!("DF{n}"), l.to_string());
        }
    }
    out
}

#[test]
fn every_partial_entry_names_a_domain_fact() {
    let c = catalogue();
    let facts = keys(&c["terms"]["facts"]);
    let named = keys(&c["terms"]["tests"]);
    let lines = fact_lines();
    assert_eq!(lines.keys().cloned().collect::<BTreeSet<_>>(), facts, "the facts of the file are not DF1 to DF8 of rce-form.md §7");
    let by_id: BTreeMap<&str, &Value> = entries(&c).iter().map(|e| (e["id"].as_str().unwrap(), e)).collect();
    // exactly the partial primitives of W4 have a domain
    let partial = set(&PARTIAL);
    assert_eq!(partial.len(), PARTIAL.len(), "PARTIAL names an entry twice");
    let with_domain: BTreeSet<String> = by_id.iter().filter(|(_, e)| !e["domain"].is_null()).map(|(id, _)| id.to_string()).collect();
    assert_eq!(with_domain, partial, "the entries with a domain are not the partial primitives of W4");
    let mut tested = BTreeSet::new();
    for e in entries(&c) {
        let id = e["id"].as_str().unwrap();
        let meaning = e["meaning"].as_str().unwrap();
        let d = &e["domain"];
        if d.is_null() {
            assert!(!meaning.contains("partial:"), "{id}: its meaning states a domain, and it has none");
            continue;
        }
        assert!(meaning.contains("partial:"), "{id}: a partial entry whose meaning does not state its domain");
        assert!(!id.contains(".bad."), "{id}: a domain test is total");
        assert!(e["traps"].as_array().unwrap().is_empty(), "{id}: a partial entry never halts, since FG8 places it inside its domain");
        assert_eq!(keys(d), set(&["fact", "tests"]), "{id}");
        let f = d["fact"].as_str().unwrap_or_else(|| panic!("{id}: a partial entry with no domain fact"));
        assert!(facts.contains(f), "{id}: {f} is not a domain fact");
        // §7 names the entry, or its family for the float operations, under that fact
        let family = id.split('.').next().unwrap();
        let line = &lines[f];
        let on_line = line.contains(&format!("`{id}`")) || line.contains(&format!("`{id} ")) || ((family == "f64" || family == "f32") && line.contains(&format!("`{family}`")));
        assert!(on_line, "{id}: rce-form.md §7 does not name it under {f}");
        let ts = d["tests"].as_array().unwrap();
        assert!(!ts.is_empty(), "{id}: a domain fact with no test");
        for t in ts {
            let t = t.as_str().unwrap();
            if named.contains(t) {
                continue;
            }
            let g = by_id.get(t).unwrap_or_else(|| panic!("{id}: the test {t} is neither a named test nor an entry"));
            assert!(t.contains(".bad."), "{id}: the test {t} is not a domain test");
            assert_eq!(g["results"].as_array().unwrap().len(), 1, "{t}");
            assert_eq!(g["results"][0]["type"], serde_json::json!(["bool"]), "{t}: a domain test gives a truth value");
            assert_eq!(g["operands"].as_array().unwrap().len(), e["operands"].as_array().unwrap().len(), "{t} does not take the operands of {id}");
            tested.insert(t.to_string());
        }
    }
    // each partial entry that §7 names under a fact has that fact in the file
    for (f, line) in &lines {
        for tok in line.split('`').skip(1).step_by(2) {
            let tok = tok.split(' ').next().unwrap();
            if partial.contains(tok) {
                assert_eq!(by_id[tok]["domain"]["fact"], *f, "{tok}: rce-form.md §7 names it under {f}");
            }
        }
    }
    // and every domain test guards some entry
    for id in by_id.keys() {
        if id.contains(".bad.") {
            assert!(tested.contains(*id), "{id} is the domain test of no entry");
        }
    }
}

/// The pattern heads that `pattern` below knows how to check.
const HEADS: [&str; 17] = ["arr", "bool", "bytes", "f32", "f64", "index", "int", "list", "order", "param", "part", "parts", "repr", "sum", "tag", "tup", "within"];

/// The pattern heads the file's `terms.patterns` defines: its keys other than
/// `other`, and the heads that `other` lists, each written `["head"` there.
fn heads(c: &Value) -> BTreeSet<String> {
    let ps = &c["terms"]["patterns"];
    let mut hs = keys(ps);
    assert!(hs.remove("other"), "terms.patterns has no `other`");
    let other = ps["other"].as_str().expect("terms.patterns.other is text");
    for r in other.split("[\"").skip(1) {
        hs.insert(r.split('"').next().unwrap().to_string());
    }
    hs
}

#[test]
fn the_pattern_heads_are_those_of_the_header() {
    assert_eq!(heads(&catalogue()), set(&HEADS), "the pattern heads of terms.patterns are not those this test checks");
}

/// Whether `p` is a pattern of the file's `terms.patterns`, whose heads are
/// `hs`, over `np` type arguments and `no` operands.
fn pattern(p: &Value, hs: &BTreeSet<String>, np: usize, no: usize) -> bool {
    let Some(a) = p.as_array() else { return false };
    let Some(h) = a.first().and_then(|h| h.as_str()) else { return false };
    if !hs.contains(h) {
        return false;
    }
    let ix = |k: usize, n: usize| a.get(k).and_then(|x| x.as_u64()).is_some_and(|x| (x as usize) < n);
    match h {
        "bool" | "f64" | "f32" | "bytes" | "order" => a.len() == 1,
        "int" => {
            let b = |k: usize| a.get(k).and_then(|x| x.as_str()).and_then(|s| s.parse::<i128>().ok());
            match (a.len(), b(1), b(2)) {
                (3, Some(lo), Some(hi)) => -(1i128 << 63) <= lo && lo <= hi && hi < (1i128 << 64),
                _ => false,
            }
        }
        "arr" | "list" | "within" => a.len() == 2 && pattern(&a[1], hs, np, no),
        "tup" | "sum" => a[1..].iter().all(|x| pattern(x, hs, np, no)),
        "param" | "parts" | "index" | "tag" | "repr" => a.len() == 2 && ix(1, np),
        "part" => a.len() == 3 && ix(1, np) && ix(2, no),
        _ => false,
    }
}

#[test]
fn every_entry_is_well_formed() {
    let c = catalogue();
    let t = &c["terms"];
    let (modes, params, exact, hs) = (keys(&t["modes"]), keys(&t["params"]), keys(&t["exactness"]), heads(&c));
    let fields = set(&["cell", "clause", "domain", "effect", "exactness", "id", "meaning", "operands", "params", "results", "symmetric", "traps"]);
    for e in entries(&c) {
        let id = e["id"].as_str().unwrap();
        assert_eq!(keys(e), fields, "{id}: its fields");
        let ps = e["params"].as_array().unwrap();
        for p in ps {
            assert!(params.contains(p.as_str().unwrap_or("")), "{id}: the type argument kind {p}");
        }
        let ops = e["operands"].as_array().unwrap();
        let rs = e["results"].as_array().unwrap();
        let (np, no) = (ps.len(), ops.len());
        let mut returned = vec![];
        for (k, o) in ops.iter().enumerate() {
            assert_eq!(keys(o), set(&["mode", "type"]), "{id}: operand {k}");
            let m = o["mode"].as_str().unwrap_or("");
            assert!(modes.contains(m), "{id}: operand {k} has the mode {m}");
            assert!(pattern(&o["type"], &hs, np, no), "{id}: operand {k} has the type {}", o["type"]);
            assert_eq!(m == "order", o["type"] == serde_json::json!(["order"]), "{id}: ω and only ω has the mode order");
            if m == "order" {
                assert_eq!(k, no - 1, "{id}: ω is the last operand");
            }
            if m != "consume" {
                returned.push(k);
            }
        }
        assert!(!rs.is_empty(), "{id}: no result");
        let mut of = vec![];
        for (j, r) in rs.iter().enumerate() {
            assert_eq!(keys(r), set(&["of", "type"]), "{id}: result {j}");
            assert!(pattern(&r["type"], &hs, np, no), "{id}: result {j} has the type {}", r["type"]);
            if let Some(k) = r["of"].as_u64() {
                let k = k as usize;
                assert!(returned.contains(&k), "{id}: result {j} returns operand {k}, whose mode returns nothing");
                assert_eq!(r["type"], ops[k]["type"], "{id}: result {j} is operand {k} come back, of its type");
                if ops[k]["mode"] == "order" {
                    assert_eq!(j, rs.len() - 1, "{id}: ω is the last result");
                }
                of.push(k);
            } else {
                assert!(r["of"].is_null(), "{id}: result {j}: of");
            }
        }
        of.sort();
        assert_eq!(of, returned, "{id}: every operand that comes back is one result");
        let takes_order = ops.iter().any(|o| o["mode"] == "order");
        let eff = e["effect"].as_u64().unwrap();
        assert!(eff <= 2, "{id}: no entry is of class 3; capabilities are cap nodes");
        assert_eq!(eff >= 1, takes_order, "{id}: an entry is of class 1 or above exactly when it takes ω");
        let traps = e["traps"].as_array().unwrap();
        assert!(traps.iter().all(|x| x.as_str().is_some_and(|s| !s.is_empty())), "{id}: traps");
        assert!(traps.is_empty() || takes_order, "{id}: an entry that halts takes ω");
        let x = e["exactness"].as_str().unwrap_or("");
        assert!(exact.contains(x), "{id}: the exactness class {x}");
        let sy = e["symmetric"].as_array().unwrap();
        if !sy.is_empty() {
            assert_eq!(sy.len(), 2, "{id}: symmetric names two positions");
            let (i, j) = (sy[0].as_u64().unwrap() as usize, sy[1].as_u64().unwrap() as usize);
            assert!(i < j && j < no, "{id}: symmetric");
            assert_eq!(ops[i], ops[j], "{id}: exchangeable operands have one type and one mode");
        }
        assert_eq!(e["cell"].as_bool().unwrap(), id.starts_with("memo."), "{id}: only the memo entries name a cell");
        assert!(e["meaning"].as_str().is_some_and(|s| !s.is_empty()), "{id}: no meaning");
        assert!(!e["clause"].as_array().unwrap().is_empty(), "{id}: no clause");
    }
}

/// The text of the section of the book that `sec` numbers (a heading such as
/// `### 6.3` or `#### 6.1.1`), up to the next heading of its level or above;
/// for a paragraph label such as C.6.2, the paragraph's section when that
/// paragraph opens a line of it.
fn section(book: &[String], sec: &str) -> Option<String> {
    for ch in book {
        let lines: Vec<&str> = ch.lines().collect();
        for (i, l) in lines.iter().enumerate() {
            let level = l.chars().take_while(|c| *c == '#').count();
            if level >= 2 && l[level..].strip_prefix(' ').and_then(|r| r.strip_prefix(sec)).is_some_and(|r| r.starts_with(' ')) {
                let body: Vec<&str> = lines[i + 1..]
                    .iter()
                    .take_while(|m| {
                        let k = m.chars().take_while(|c| *c == '#').count();
                        k == 0 || k > level
                    })
                    .copied()
                    .collect();
                return Some(body.join("\n"));
            }
        }
    }
    let (parent, _) = sec.rsplit_once('.')?;
    let body = section(book, parent)?;
    body.lines().any(|l| l.starts_with(&format!("{sec} "))).then_some(body)
}

#[test]
fn every_clause_names_a_section_of_the_book() {
    let mut book = vec![];
    for f in std::fs::read_dir(spec().join("book")).unwrap() {
        let p = f.unwrap().path();
        if p.extension().is_some_and(|x| x == "md") {
            book.push(std::fs::read_to_string(p).unwrap());
        }
    }
    for e in entries(&catalogue()) {
        let id = e["id"].as_str().unwrap();
        for cl in e["clause"].as_array().unwrap() {
            let cl = cl.as_str().unwrap();
            let mut parts = cl.split(' ');
            let sec = parts.next().unwrap();
            let body = section(&book, sec).unwrap_or_else(|| panic!("{id}: the clause {cl} names no section of the book"));
            for rule in parts {
                assert!(body.contains(&format!("**{rule} ")), "{id}: section {sec} has no rule {rule}");
            }
        }
    }
}
