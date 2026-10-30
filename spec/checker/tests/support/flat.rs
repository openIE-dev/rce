//! Standard library modules as one flat program, for the tests that run
//! their functions as host steps of one module (as stdlib/scripts/flatten.py
//! does for `rce run`): each module loses its `module` and `use` lines (and
//! its `check` lines unless `checks`), and the `alias.` qualifiers of the
//! listed modules, outside strings and comments. The flat text means what
//! the modules mean linked (4.1).
use std::path::Path;

pub fn flat(mods: &[&str], checks: bool) -> String {
    let mut out = String::new();
    for m in mods {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../stdlib/lib").join(format!("{m}.rce"));
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        for l in text.lines() {
            if l.starts_with("module ") || l.starts_with("use ") || (!checks && l.starts_with("check ")) {
                continue;
            }
            out.push_str(&unqualify(l, mods));
            out.push('\n');
        }
    }
    out
}

/// `line` with `m.` removed before an identifier for each listed module m,
/// in code only (not in a string or after `#`).
pub fn unqualify(line: &str, mods: &[&str]) -> String {
    let b = line.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'#' {
            out.push_str(&line[i..]);
            break;
        }
        if c == b'"' {
            let mut j = i + 1;
            while j < b.len() && b[j] != b'"' {
                j += if b[j] == b'\\' { 2 } else { 1 };
            }
            let j = (j + 1).min(b.len());
            out.push_str(&line[i..j]);
            i = j;
            continue;
        }
        if (c.is_ascii_alphabetic() || c == b'_') && (i == 0 || !(ident(b[i - 1]) || b[i - 1] == b'.')) {
            let mut j = i;
            while j < b.len() && ident(b[j]) {
                j += 1;
            }
            let w = &line[i..j];
            if mods.contains(&w) && b.get(j) == Some(&b'.') && b.get(j + 1).is_some_and(|x| x.is_ascii_alphabetic() || *x == b'_') {
                i = j + 1;
            } else {
                out.push_str(w);
                i = j;
            }
            continue;
        }
        let ch = line[i..].chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}
