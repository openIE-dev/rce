//! `rce fmt --nameless` (10.6): the name-free projection of a module's text.
//!
//! Every identifier the module declares is replaced by a positional name in
//! order of first declaration, by kind: regions `r0, r1, …`, events `e0, …`,
//! errors `x0, …`, functions `f0, …`, types (struct, kind, trait, alias,
//! actor, protocol) `t0, …`, kind variants `k0, …`, and fields, parameters
//! and locals `v0, …`. Keywords, built-in types, standard library names and
//! string contents are untouched, so the projection is accepted by the
//! checker whenever the source is (T7: names carry no meaning) and has the
//! same Form up to the name-free keys of 6.6. Intent strings are kept: they
//! are the concept record, not names.
//!
//! Declared names are found from the token stream by their declaring
//! keyword or position (`NAME :` after a line start, `(` or `,`; a bare
//! name on its own line inside a `kind` block; the binding side of a
//! payload pattern `field: NAME`). A name the scan does not recognise as
//! declared is left as written, which keeps the output a valid program.
//! The projection is deterministic and idempotent.
//!
//! Status: written without the toolchain at hand; not yet built or run.

use crate::parse::{lex, Tok, DRAFT06_KEYWORDS, KEYWORDS};
use std::collections::{BTreeMap, HashSet};

const BUILTIN_TYPES: &[&str] = &[
    "bool", "text", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize", "f32", "f64",
    "List", "Map", "Set", "Box", "Option", "Secret", "Cap", "Guarded", "Text", "World", "Template",
    "Shell", "Prompt", "Header", "mathint", "Tick", "self", "some", "none", "true", "false",
];

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Kind {
    Region,
    Event,
    Error,
    Fn,
    Type,
    Variant,
    Value,
}

fn prefix(k: Kind) -> &'static str {
    match k {
        Kind::Region => "r",
        Kind::Event => "e",
        Kind::Error => "x",
        Kind::Fn => "f",
        Kind::Type => "t",
        Kind::Variant => "k",
        Kind::Value => "v",
    }
}

/// The rename map: declared name → positional name.
fn renames(src: &str) -> Result<BTreeMap<String, String>, String> {
    let toks: Vec<Tok> = lex(src)?.into_iter().map(|(t, _)| t).collect();
    let reserved: HashSet<&str> = KEYWORDS.iter().chain(DRAFT06_KEYWORDS.iter()).chain(BUILTIN_TYPES.iter()).copied().collect();
    let mut declared: Vec<(String, Kind)> = vec![];
    let mut seen: HashSet<String> = HashSet::new();
    let mut declare = |n: &str, k: Kind| {
        if !reserved.contains(n) && !seen.contains(n) {
            seen.insert(n.to_string());
            declared.push((n.to_string(), k));
        }
    };
    let mut in_kind_block = false;
    let mut kind_depth = 0usize;
    let mut depth = 0usize;
    for i in 0..toks.len() {
        match &toks[i] {
            Tok::Indent => {
                depth += 1;
            }
            Tok::Dedent => {
                depth = depth.saturating_sub(1);
                if in_kind_block && depth < kind_depth {
                    in_kind_block = false;
                }
            }
            Tok::Id(n) => {
                let prev = if i > 0 { Some(&toks[i - 1]) } else { None };
                let next = toks.get(i + 1);
                let prev_kw = |k: &str| matches!(prev, Some(Tok::Id(p)) if p == k);
                if prev_kw("state") || prev_kw("actor") {
                    declare(n, Kind::Region);
                } else if prev_kw("event") {
                    declare(n, Kind::Event);
                } else if prev_kw("error") {
                    declare(n, Kind::Error);
                } else if prev_kw("fn") {
                    declare(n, Kind::Fn);
                } else if prev_kw("struct") || prev_kw("trait") || prev_kw("type") || prev_kw("protocol") {
                    declare(n, Kind::Type);
                } else if prev_kw("kind") {
                    declare(n, Kind::Type);
                    in_kind_block = true;
                    kind_depth = depth + 1;
                } else if prev_kw("const") || prev_kw("let") || prev_kw("var") || prev_kw("for") {
                    declare(n, Kind::Value);
                } else if in_kind_block && depth == kind_depth && matches!(prev, Some(Tok::Nl) | Some(Tok::Indent)) {
                    declare(n, Kind::Variant);
                } else if matches!(next, Some(Tok::P(":")))
                    && matches!(prev, Some(Tok::Nl) | Some(Tok::Indent) | Some(Tok::P("(")) | Some(Tok::P(",")) | Some(Tok::P("|")))
                {
                    // a field, a parameter, a payload field or a struct-literal key
                    declare(n, Kind::Value);
                } else if matches!(prev, Some(Tok::P(":"))) && matches!(next, Some(Tok::P(")")) | Some(Tok::P(","))) && !reserved.contains(n.as_str()) {
                    // the binding side of a payload pattern: `Deposit(amount: a)`
                    // (a type in this position would be a declared type or built-in)
                    if !seen.contains(n) && !declared.iter().any(|(d, _)| d == n) && n.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
                        declare(n, Kind::Value);
                    }
                }
            }
            _ => {}
        }
    }
    let mut counters: BTreeMap<Kind, usize> = BTreeMap::new();
    let mut map = BTreeMap::new();
    let all_ids: HashSet<String> = toks.iter().filter_map(|t| if let Tok::Id(s) = t { Some(s.clone()) } else { None }).collect();
    for (n, k) in declared {
        loop {
            let c = counters.entry(k).or_insert(0);
            let cand = format!("{}{}", prefix(k), *c);
            *c += 1;
            // never collide with a name the source already uses and does not rename
            if !all_ids.contains(&cand) || map.contains_key(&cand) {
                map.insert(n.clone(), cand);
                break;
            }
        }
    }
    Ok(map)
}

/// Rewrite identifiers outside strings and comments by the map.
pub fn nameless(src: &str) -> Result<String, String> {
    let map = renames(src)?;
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i] as char;
        if c == '"' {
            // string or template literal body: copy verbatim with escapes
            let start = i;
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' {
                    i += 1;
                }
                i += 1;
            }
            i = (i + 1).min(b.len());
            out.push_str(&src[start..i]);
        } else if c == '#' {
            let start = i;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            out.push_str(&src[start..i]);
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let word = &src[start..i];
            match map.get(word) {
                Some(r) => out.push_str(r),
                None => out.push_str(word),
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}
