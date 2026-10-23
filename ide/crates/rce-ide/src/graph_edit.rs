//! Graph edits that become text edits (Enso lesson: graph and text are two
//! equal views of one program). Each operation rewrites the source files,
//! then the whole program must pass `rce check`; otherwise nothing changes
//! and the diagnostics are returned. The graph is rebuilt from the new
//! text, so there is one source of truth.
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    /// Rename a top-level definition and every whole-word use.
    Rename { from: String, to: String },
    /// Delete the top-level item that defines `name`.
    Delete { name: String },
    /// Add a function stub to `file` (a hole to fill in the text view).
    AddFn { file: String, name: String },
    /// Move a definition to another file.
    Move { name: String, to_file: String },
}

pub type Files = BTreeMap<String, String>;

fn ident(c: char) -> bool { c.is_alphanumeric() || c == '_' }

fn rename_word(src: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while let Some(o) = src[i..].find(from) {
        let s = i + o;
        let e = s + from.len();
        let before = src[..s].chars().next_back().is_some_and(ident);
        let after = src[e..].chars().next().is_some_and(ident);
        out.push_str(&src[i..s]);
        out.push_str(if before || after { from } else { to });
        i = e;
    }
    out.push_str(&src[i..]);
    out
}

/// Line range [start, end) of the top-level item defining `name`.
fn item_span(src: &str, name: &str) -> Option<(usize, usize)> {
    let lines: Vec<&str> = src.lines().collect();
    let kw = ["fn", "state", "event", "step", "struct", "kind", "type", "error", "trait", "actor", "const", "linear struct"];
    let start = lines.iter().position(|l| {
        kw.iter().any(|k| l.strip_prefix(k).and_then(|r| r.strip_prefix(' ')).is_some_and(|r| {
            r.starts_with(name) && !r[name.len()..].chars().next().is_some_and(ident)
        }))
    })?;
    let mut end = start + 1;
    while end < lines.len() && (lines[end].is_empty() || lines[end].starts_with([' ', '\t'])) { end += 1 }
    while end > start + 1 && lines[end - 1].trim().is_empty() { end -= 1 }
    Some((start, end))
}

/// The text of the top-level item defining `name` (what a review pins).
pub fn item_text(src: &str, name: &str) -> Option<String> {
    let (a, b) = item_span(src, name)?;
    Some(src.lines().skip(a).take(b - a).collect::<Vec<_>>().join("\n") + "\n")
}

fn cut(src: &str, (a, b): (usize, usize)) -> (String, String) {
    let lines: Vec<&str> = src.lines().collect();
    let item = lines[a..b].join("\n") + "\n";
    let mut rest: Vec<&str> = lines[..a].to_vec();
    let mut tail = &lines[b..];
    while tail.first().is_some_and(|l| l.trim().is_empty()) && rest.last().is_none_or(|l| l.trim().is_empty()) { tail = &tail[1..] }
    rest.extend_from_slice(tail);
    let mut r = rest.join("\n");
    if !r.is_empty() { r.push('\n') }
    (item, r)
}

fn apply_raw(files: &Files, op: &GraphOp) -> Result<Files, String> {
    let mut f = files.clone();
    match op {
        GraphOp::Rename { from, to } => {
            if !to.chars().all(ident) || to.is_empty() { return Err(format!("`{to}` is not a name")) }
            for v in f.values_mut() { *v = rename_word(v, from, to) }
        }
        GraphOp::Delete { name } => {
            let (k, sp) = f.iter().find_map(|(k, v)| item_span(v, name).map(|s| (k.clone(), s))).ok_or(format!("no definition {name}"))?;
            let (_, rest) = cut(&f[&k], sp);
            f.insert(k, rest);
        }
        GraphOp::AddFn { file, name } => {
            let v = f.entry(file.clone()).or_default();
            if !v.is_empty() && !v.ends_with("\n\n") { if !v.ends_with('\n') { v.push('\n') } v.push('\n') }
            v.push_str(&format!("fn {name}() -> i64\n    return 0\n"));
        }
        GraphOp::Move { name, to_file } => {
            let (k, sp) = f.iter().find_map(|(k, v)| item_span(v, name).map(|s| (k.clone(), s))).ok_or(format!("no definition {name}"))?;
            if &k == to_file { return Ok(f) }
            let (item, rest) = cut(&f[&k], sp);
            f.insert(k, rest);
            let v = f.entry(to_file.clone()).or_default();
            if !v.is_empty() { if !v.ends_with('\n') { v.push('\n') } v.push('\n') }
            v.push_str(&item);
        }
    }
    Ok(f)
}

/// Apply `op`; return the new files only if `rce check` admits the program.
pub fn apply(files: &Files, op: &GraphOp) -> Result<Files, Vec<String>> {
    let f = apply_raw(files, op).map_err(|e| vec![e])?;
    let prog: Vec<(String, String)> = f.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    let errs: Vec<String> = rce_spec_check::check_program(&prog, false).into_iter()
        .filter_map(|(p, r)| r.err().map(|e| format!("{p}: {e}"))).collect();
    if errs.is_empty() { Ok(f) } else { Err(errs) }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn files() -> Files {
        [("a.rce".to_string(), "fn leaf(x: i64) -> i64\n    return x + 1\n\nfn top() -> i64\n    return leaf(2)\n".to_string())].into()
    }
    #[test]
    fn rename_add_move_delete() {
        let f = apply(&files(), &GraphOp::Rename { from: "leaf".into(), to: "bump".into() }).unwrap();
        assert!(f["a.rce"].contains("fn bump(") && f["a.rce"].contains("return bump(2)"));
        let f = apply(&f, &GraphOp::AddFn { file: "a.rce".into(), name: "fresh".into() }).unwrap();
        assert!(f["a.rce"].contains("fn fresh() -> i64"));
        let f = apply(&f, &GraphOp::Move { name: "fresh".into(), to_file: "b.rce".into() }).unwrap();
        assert!(f["b.rce"].starts_with("fn fresh") && !f["a.rce"].contains("fresh"));
        let f = apply(&f, &GraphOp::Delete { name: "fresh".into() }).unwrap();
        assert!(!f["b.rce"].contains("fresh"));
    }
    #[test]
    fn edits_the_checker_refuses_change_nothing() {
        // Deleting a function that is still called must be refused.
        assert!(apply(&files(), &GraphOp::Delete { name: "leaf".into() }).is_err());
        assert!(apply(&files(), &GraphOp::Rename { from: "leaf".into(), to: "a b".into() }).is_err());
    }
}
