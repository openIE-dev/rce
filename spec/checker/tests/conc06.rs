//! Grammar-level conformance tests for the draft 0.6 concurrency chapter
//! (Appendix C, book/14-draft06-concurrency.md, section C.2).
//!
//! The chapter's ```ebnf block lists new and replaced rules. This test
//! takes the section-3 grammar from book/03-syntax.md, renames each rule
//! the chapter replaces to `Name__base` (so the old rule is unreachable),
//! appends the chapter's rules, and then:
//! 1. builds the LALR(1) tables and requires 0 conflicts;
//! 2. requires every program in spec/examples/ to stay accepted when the
//!    six new keywords are reserved (backward compatibility);
//! 3. requires every program in spec/conc06/accept/ to be accepted and
//!    every program in spec/conc06/change/ to require change;
//! 4. requires every program in spec/conc06/accept/ to require change under
//!    the unpatched draft 0.5 grammar (so the tests exercise the new rules).
//!
//! Only the grammar is tested here; the typing rules W-PAR-*, W-ND-1,
//! W-BUD-1 and W-DL-1 are not implemented in rce-check yet.
use rce_spec_check::lalr::{Lalr, book_grammar};
use rce_spec_check::parse::{KEYWORDS, Tok, lex};
use std::path::{Path, PathBuf};

const NEW_KEYWORDS: &[&str] = &["par", "branch", "nondet", "budget", "deadline", "priority"];
/// Draft 0.6b keywords (C.11); grammar level only, not in the Rust parser.
const EXT_KEYWORDS: &[&str] = &[
    "replicated", "interrupt", "session", "lvar", "group", "lanes", "barrier", "scope", "atomic",
    "protocol", "offer", "choose", "node", "supervisor", "strategy", "child", "restart",
];

fn spec_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn chapter_rules() -> String {
    chapter_block("```ebnf\n")
}

/// The draft 0.6b blocks (```ebnf-ext), which may replace core 0.6 rules.
fn ext_rules() -> String {
    chapter_block("```ebnf-ext\n")
}

fn chapter_block(tag: &str) -> String {
    let md = std::fs::read_to_string(spec_dir().join("CONCURRENCY-0.6.md")).unwrap();
    let tag = tag.trim_end();
    let mut out = String::new();
    let mut inside = false;
    for line in md.lines() {
        let t = line.trim();
        if !inside && t == tag {
            inside = true;
            continue;
        }
        if inside && t.starts_with("```") {
            inside = false;
            continue;
        }
        if inside {
            out.push_str(line);
            out.push('\n');
        }
    }
    assert!(!out.is_empty(), "no {tag} block in CONCURRENCY-0.6.md");
    out
}

/// Names defined by the chapter's rules.
fn defined_names(rules: &str) -> Vec<String> {
    let re = regex::Regex::new(r"(?m)(?:^|\s)([A-Z][A-Za-z]*)\s+=\s").unwrap();
    re.captures_iter(rules).map(|c| c[1].to_string()).collect()
}

fn rename_defs(code: &str, rules: &str, suffix: &str) -> String {
    let mut code = code.to_string();
    for n in defined_names(rules) {
        let re = regex::Regex::new(&format!(r"(?m)(^|\s){n}(\s+=\s)")).unwrap();
        code = re.replace_all(&code, format!("${{1}}{n}{suffix}${{2}}")).into_owned();
    }
    code
}

fn base_code() -> String {
    let base = std::fs::read_to_string(spec_dir().join("book/03-syntax.md")).unwrap();
    let mut code = String::new();
    let mut inside = false;
    for line in base.lines() {
        if line.trim_start().starts_with("```") {
            inside = !inside;
            continue;
        }
        if inside {
            code.push_str(line);
            code.push('\n');
        }
    }
    code
}

/// Draft 0.5 grammar with the core 0.6 rules (C.2) applied.
fn patched_grammar_md() -> String {
    let rules = chapter_rules();
    format!("```\n{}{rules}```\n", rename_defs(&base_code(), &rules, "__base"))
}

/// Core 0.6 plus the 0.6b extension rules (C.11), which override by name.
fn ext_grammar_md() -> String {
    let core = chapter_rules();
    let ext = ext_rules();
    let l1 = format!("{}{core}", rename_defs(&base_code(), &core, "__base"));
    format!("```\n{}{ext}```\n", rename_defs(&l1, &ext, "__core"))
}

fn tokens(l: &Lalr, src: &str, extra_kw: bool) -> Option<Vec<usize>> {
    tokens_kw(l, src, if extra_kw { NEW_KEYWORDS } else { &[] })
}

fn tokens_kw(l: &Lalr, src: &str, extra: &[&str]) -> Option<Vec<usize>> {
    let toks = lex(src).ok()?;
    let mut out = Vec::new();
    let mut after_dot = false;
    for (t, _) in toks {
        let name: String = match &t {
            Tok::Id(s) if s == "_" => "_".into(),
            Tok::Id(s)
                if !after_dot
                    && (KEYWORDS.contains(&s.as_str())
                        || extra.contains(&s.as_str())) =>
            {
                s.clone()
            }
            Tok::Id(_) => "NAME".into(),
            Tok::Int(_) => "INT".into(),
            Tok::Float(_) => "FLOAT".into(),
            Tok::Str(_) => "STRING".into(),
            Tok::Tpl(_) => "TEMPLATE".into(),
            Tok::P(p) => p.to_string(),
            Tok::Nl => "NEWLINE".into(),
            Tok::Indent => "INDENT".into(),
            Tok::Dedent => "DEDENT".into(),
            Tok::Eof => break,
        };
        after_dot = matches!(&t, Tok::P("."));
        out.push(l.term(&name)?);
    }
    Some(out)
}

fn accepts(l: &Lalr, src: &str, extra_kw: bool) -> bool {
    tokens(l, src, extra_kw).is_some_and(|t| l.accepts_ids(&t))
}

fn files(d: &Path) -> Vec<(String, String)> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v.into_iter()
        .map(|p| {
            let s = std::fs::read_to_string(&p).unwrap();
            (p.file_name().unwrap().to_string_lossy().into_owned(), s)
        })
        .collect()
}

#[test]
fn conc06_grammar_is_lalr1() {
    if !spec_dir().join("CONCURRENCY-0.6.md").exists() {
        eprintln!("skipped: the draft 0.6 chapter is not published in this repository");
        return;
    }
    let g = book_grammar(&patched_grammar_md()).expect("patched grammar reads");
    let rules = g.source_rules;
    let l = Lalr::build(g);
    println!(
        "draft 0.6 grammar: {rules} rules, {} productions, {} states, {} conflicts",
        l.g.prods.len(),
        l.states,
        l.conflicts.len()
    );
    assert!(l.conflicts.is_empty(), "{:#?}", l.conflicts);
}

#[test]
fn conc06_examples_still_parse() {
    if !spec_dir().join("CONCURRENCY-0.6.md").exists() {
        eprintln!("skipped: the draft 0.6 chapter is not published in this repository");
        return;
    }
    let l = Lalr::build(book_grammar(&patched_grammar_md()).unwrap());
    let ex = files(&spec_dir().join("examples"));
    assert!(ex.len() >= 51);
    for (n, s) in &ex {
        assert!(accepts(&l, s, true), "draft 0.5 example {n} does not parse with the 0.6 grammar");
    }
    for (n, s) in files(&spec_dir().join("examples/change/syntax")) {
        assert!(!accepts(&l, &s, true), "reject {n} accepted by 0.6 grammar");
    }
}

#[test]
fn conc06_accept_and_reject() {
    if !spec_dir().join("CONCURRENCY-0.6.md").exists() {
        eprintln!("skipped: the draft 0.6 chapter is not published in this repository");
        return;
    }
    let l = Lalr::build(book_grammar(&patched_grammar_md()).unwrap());
    let acc = files(&spec_dir().join("conc06/accept"));
    let rej = files(&spec_dir().join("conc06/change"));
    assert!(!acc.is_empty() && !rej.is_empty());
    for (n, s) in &acc {
        assert!(accepts(&l, s, true), "conc06/accept/{n} does not parse");
    }
    for (n, s) in &rej {
        assert!(!accepts(&l, s, true), "conc06/change/{n} accepted");
    }
    println!("conc06: {} accepted, {} require change", acc.len(), rej.len());
}

#[test]
fn conc06_programs_need_the_new_rules() {
    let base = std::fs::read_to_string(spec_dir().join("book/03-syntax.md")).unwrap();
    let l = Lalr::build(book_grammar(&base).unwrap());
    for (n, s) in files(&spec_dir().join("conc06/accept")) {
        assert!(!accepts(&l, &s, false), "conc06/accept/{n} already parses under draft 0.5");
    }
}

fn ext_kw() -> Vec<&'static str> {
    NEW_KEYWORDS.iter().chain(EXT_KEYWORDS).copied().collect()
}

#[test]
fn conc06b_grammar_is_lalr1_and_conforms() {
    if !spec_dir().join("CONCURRENCY-0.6.md").exists() {
        eprintln!("skipped: the draft 0.6 chapter is not published in this repository");
        return;
    }
    let g = book_grammar(&ext_grammar_md()).expect("0.6b grammar reads");
    let rules = g.source_rules;
    let l = Lalr::build(g);
    println!(
        "draft 0.6b grammar: {rules} rules, {} productions, {} states, {} conflicts",
        l.g.prods.len(),
        l.states,
        l.conflicts.len()
    );
    assert!(l.conflicts.is_empty(), "{:#?}", l.conflicts);
    let kw = ext_kw();
    let acc = |s: &str| tokens_kw(&l, s, &kw).is_some_and(|t| l.accepts_ids(&t));
    for (n, s) in files(&spec_dir().join("examples")) {
        assert!(acc(&s), "draft 0.5 example {n} does not parse with the 0.6b grammar");
    }
    for (n, s) in files(&spec_dir().join("examples/change/syntax")) {
        assert!(!acc(&s), "reject {n} accepted by 0.6b grammar");
    }
    for d in ["conc06/accept", "conc06b/accept"] {
        for (n, s) in files(&spec_dir().join(d)) {
            assert!(acc(&s), "{d}/{n} does not parse with the 0.6b grammar");
        }
    }
    let rej = files(&spec_dir().join("conc06b/change"));
    for (n, s) in &rej {
        assert!(!acc(s), "conc06b/change/{n} accepted");
    }
    // the 0.6b programs need the 0.6b rules
    let core = Lalr::build(book_grammar(&patched_grammar_md()).unwrap());
    for (n, s) in files(&spec_dir().join("conc06b/accept")) {
        assert!(
            !tokens_kw(&core, &s, NEW_KEYWORDS).is_some_and(|t| core.accepts_ids(&t)),
            "conc06b/accept/{n} already parses under core 0.6"
        );
    }
    println!("conc06b: {} programs in reject/ require change", rej.len());
}
