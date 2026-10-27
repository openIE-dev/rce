//! W2: the spec as machine-readable data. Every table here is read from the
//! normative chapter files of `spec/book`, never typed in by hand, and is
//! stamped with the body SHA-256 of each chapter it came from (the hash in
//! the chapter footer, see `book.rs`). Regenerating from the same spec
//! commit gives the same bytes (`tests/spec_data.rs`).
//!
//! Contents:
//! * `chapters`: file name to body SHA-256, for every chapter of INDEX.md;
//! * `grammar` / `grammar_06`: the LALR(1) terminals, nonterminals,
//!   productions, ACTION and GOTO tables built from section 3 (and with C.2
//!   applied);
//! * `w_rules`: every W-rule of section 7 with its text;
//! * `edges`: the 6.3 edge semantics N1 to N11 with their text;
//! * `methods`: the 5.15 standard library table (receiver, RCE type, methods);
//! * `method_work`: the C.6.4 complexity classes (class, methods, work).
use crate::book::split;
use crate::lalr::{Lalr, Sym, book_grammar, book_grammar_06};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

fn read(spec: &Path, ch: &str) -> String {
    std::fs::read_to_string(spec.join("book").join(ch)).unwrap_or_else(|e| panic!("{ch}: {e}"))
}

fn grammar(l: &Lalr) -> Value {
    let sym = |s: &Sym| match s {
        Sym::T(t) => json!(["t", t]),
        Sym::N(n) => json!(["n", n]),
    };
    let (a, g) = l.tables();
    json!({
        "terminals": l.g.terms,
        "nonterminals": l.g.nts,
        "productions": l.g.prods.iter().map(|(lhs, rhs)| json!([lhs, rhs.iter().map(sym).collect::<Vec<_>>()])).collect::<Vec<_>>(),
        "states": l.states,
        "conflicts": l.conflicts,
        "action": a.iter().map(|(s, t, k, x)| json!([s, t, k, x])).collect::<Vec<_>>(),
        "goto": g.iter().map(|(s, n, x)| json!([s, n, x])).collect::<Vec<_>>(),
    })
}

/// Rules defined in bulleted items of a chapter. A definition is a rule
/// name at the start of an item, or after `; ` or `. ` inside one, followed
/// by a space (section 7 style, `W-OWN-1 Use after move`) or by `: `
/// (Appendix C style, `W-DL-1: outside ...`); its text runs to the next
/// definition in the item. Mentions elsewhere (`... as W-VIS-1`) are not
/// definitions there; a rule that is only named inside an item (Appendix
/// C writes `... (W-INTR-1)` and `a failure is W-CRDT-1`) takes the whole
/// item as its text, after every definition is placed.
fn w_rules(md: &str) -> BTreeMap<String, String> {
    let mut items: Vec<String> = vec![];
    let mut open = false;
    for line in md.lines() {
        if let Some(r) = line.strip_prefix("- ") {
            items.push(r.to_string());
            open = true;
        } else if open && line.starts_with("  ") {
            let last = items.last_mut().unwrap();
            last.push(' ');
            last.push_str(line.trim());
        } else {
            open = false;
        }
    }
    let re = regex::Regex::new(r"(?:^|; |\. )(W-[A-Z]+(?:-\d+)?):? ").unwrap();
    let mut out = BTreeMap::new();
    for it in &items {
        let starts: Vec<(usize, usize, String)> = re
            .captures_iter(it)
            .map(|c| (c.get(0).unwrap().start(), c.get(0).unwrap().end(), c[1].to_string()))
            .collect();
        for (i, (_, body, name)) in starts.iter().enumerate() {
            let end = starts.get(i + 1).map(|x| x.0 + 1).unwrap_or(it.len());
            out.entry(name.clone()).or_insert_with(|| it[*body..end].trim().trim_end_matches(';').to_string());
        }
    }
    let any = regex::Regex::new(r"W-[A-Z]+-\d+").unwrap();
    for it in &items {
        for m in any.find_iter(it) {
            out.entry(m.as_str().to_string()).or_insert_with(|| it.trim().to_string());
        }
    }
    out
}

fn mentions(md: &str) -> Vec<String> {
    let re = regex::Regex::new(r"W-[A-Z]+-\d+").unwrap();
    let mut v: Vec<String> = re.find_iter(md).map(|m| m.as_str().to_string()).collect();
    v.sort();
    v.dedup();
    v
}

fn edges(md: &str) -> BTreeMap<String, String> {
    let re = regex::Regex::new(r"^- \*\*(N\d+) ([^*]+)\*\*(.*)$").unwrap();
    let mut out = BTreeMap::new();
    let mut cur: Option<(String, String)> = None;
    for line in md.lines() {
        if let Some(c) = re.captures(line) {
            if let Some((k, v)) = cur.take() {
                out.insert(k, v);
            }
            cur = Some((c[1].to_string(), format!("{}{}", c[2].trim_end_matches('.'), c[3].trim_end()).trim().to_string()));
        } else if let Some((_, v)) = cur.as_mut() {
            if line.starts_with("  ") {
                v.push(' ');
                v.push_str(line.trim());
            } else {
                let (k, v) = cur.take().unwrap();
                out.insert(k, v);
            }
        }
    }
    if let Some((k, v)) = cur {
        out.insert(k, v);
    }
    out
}

fn table_rows(md: &str, after: &str) -> Vec<Vec<String>> {
    let i = md.find(after).unwrap_or_else(|| panic!("table marker {after:?}"));
    let mut rows = vec![];
    let mut started = false;
    for line in md[i..].lines() {
        if line.starts_with('|') {
            started = true;
            if line.chars().all(|x| matches!(x, '|' | '-' | ':' | ' ')) {
                continue;
            }
            let cells: Vec<String> = line.trim().trim_matches('|').split(" | ").map(|c| c.trim().to_string()).collect();
            rows.push(cells);
        } else if started {
            break;
        }
    }
    rows.remove(0); // header
    rows
}

fn names(cell: &str) -> Vec<String> {
    cell.split(", ").map(|x| x.trim().trim_matches('`').to_string()).collect()
}

pub fn export(spec: &Path) -> String {
    let mut chapters = BTreeMap::new();
    let book = crate::book::Book { here: spec.join("book"), spec: spec.to_path_buf() };
    for ch in book.chapters() {
        let (_, h) = split(&read(spec, &ch));
        chapters.insert(ch.clone(), h.unwrap_or_default());
    }
    let syn = read(spec, "03-syntax.md");
    let app = read(spec, "14-draft06-concurrency.md");
    let g05 = Lalr::build(book_grammar(&syn).expect("section 3 grammar"));
    let mut g06 = Lalr::build(book_grammar_06(&syn, &app).expect("C.2 grammar"));
    g06.draft06 = true;
    let ch6 = read(spec, "06-regions-effects-extensions.md");
    let methods: Vec<Value> = table_rows(&ch6, "| Receiver | RCE type | Methods |")
        .iter()
        .map(|r| json!({"receiver": r[0].trim_matches('`'), "type": r[1], "methods": names(&r[2])}))
        .collect();
    let work: Vec<Value> = table_rows(&app, "| Class | Methods | work |")
        .iter()
        .map(|r| json!({"class": r[0], "methods": if r[1].starts_with('`') { json!(names(&r[1])) } else { json!(r[1]) }, "work": r[2]}))
        .collect();
    let v = json!({
        "format": "rce-spec-data/1",
        "source": "spec/book (normative chapters); see crates/rce/spec/checker/src/specdata.rs",
        "chapters": chapters,
        "grammar": {"from": ["03-syntax.md"], "lalr1": grammar(&g05)},
        "grammar_06": {"from": ["03-syntax.md", "14-draft06-concurrency.md"], "lalr1": grammar(&g06)},
        "w_rules": {"from": "09-wellformedness.md", "rules": w_rules(&read(spec, "09-wellformedness.md")), "mentioned": mentions(&read(spec, "09-wellformedness.md"))},
        "w_rules_06": {"from": "14-draft06-concurrency.md", "rules": w_rules(&app), "mentioned": mentions(&app)},
        "edges": {"from": "07-form-evaluation.md", "rules": edges(&read(spec, "07-form-evaluation.md"))},
        "methods": {"from": "06-regions-effects-extensions.md", "table": methods},
        "method_work": {"from": "14-draft06-concurrency.md", "table": work},
    });
    serde_json::to_string(&v).unwrap() + "\n"
}
