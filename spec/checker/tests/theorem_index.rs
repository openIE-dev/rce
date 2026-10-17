//! Spec 8, numbering and assurance: every theorem cited in the book has a
//! row, numbers are unique, and every Lean name in a status table exists
//! as a declaration (or module) in spec/lean.
use regex::Regex;
use std::collections::BTreeSet;

#[test]
fn theorem_rows_and_lean_names_exist() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let read = |p: &str| std::fs::read_to_string(dir.join(p)).unwrap();
    let tables = [read("book/10-theorems.md"), read("book/14-draft06-concurrency.md")];
    let row = Regex::new(r"(?m)^\| (T\d+[ab]?) \|(.*)\|\s*$").unwrap();
    let mut ids = Vec::new();
    let mut names = BTreeSet::new();
    let tick = Regex::new(r"`([A-Za-z_][A-Za-z0-9_.']*)`").unwrap();
    for t in &tables {
        for c in row.captures_iter(t) {
            ids.push(c[1].to_string());
            let last = c[2].rsplit('|').next().unwrap();
            for n in tick.captures_iter(last) {
                names.insert(n[1].to_string());
            }
        }
    }
    let uniq: BTreeSet<_> = ids.iter().cloned().collect();
    assert_eq!(uniq.len(), ids.len(), "a theorem number has two rows");
    // Every theorem cited in the book has a row.
    let cite = Regex::new(r"\bT(\d+)([ab]?)\b").unwrap();
    for e in std::fs::read_dir(dir.join("book")).unwrap() {
        let p = e.unwrap().path();
        if p.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let s = std::fs::read_to_string(&p).unwrap();
        for c in cite.captures_iter(&s) {
            let id = format!("T{}{}", &c[1], &c[2]);
            let base = format!("T{}", &c[1]);
            assert!(
                uniq.contains(&id) || uniq.iter().any(|u| u.starts_with(&base) && u.len() == base.len() + 1),
                "{} cites {id}, which has no row",
                p.display()
            );
        }
    }
    // Every Lean name exists.
    let mut lean = String::new();
    // Walk lean/RCE recursively (it has subdirectories such as Core/);
    // a module is named by its file stem, and only .lean files are read.
    let mut stack = vec![dir.join("lean/RCE")];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if p.extension().and_then(|x| x.to_str()) != Some("lean") {
                continue;
            }
            lean.push_str(&p.file_stem().unwrap().to_string_lossy());
            lean.push('\n');
            lean.push_str(&std::fs::read_to_string(&p).unwrap());
        }
    }
    let mut missing = vec![];
    for n in &names {
        // A cited file name (`SessionVectors.lean`) names its module.
        if let Some(stem) = n.strip_suffix(".lean") {
            let stem = stem.rsplit('/').next().unwrap();
            if !lean.lines().any(|l| l == stem) {
                missing.push(n.clone());
            }
            continue;
        }
        let short = n.rsplit('.').next().unwrap();
        let decl = Regex::new(&format!(
            r"(?m)(^|\s)(theorem|lemma|def|abbrev|structure|inductive)\s+(\S*\.)?{}(\s|$)",
            regex::escape(short)
        ))
        .unwrap();
        let module = lean.lines().any(|l| l == short);
        if !decl.is_match(&lean) && !module {
            missing.push(n.clone());
        }
    }
    assert!(missing.is_empty(), "Lean names in the status tables with no declaration: {missing:?}");
    assert!(names.len() > 60, "only {} Lean names found", names.len());
}
