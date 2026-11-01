//! `rce fmt nameless` (10.6): the name-free projection of a module's text.
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
//! A use is placed by its position as well as its name: after `.` a call
//! is renamed only when it names a method of the module's own `impl`
//! blocks, a field only when it is a field of the module's own types, and a
//! label in parentheses only when the callee is the module's own name, so a
//! standard library method or another module's field keeps its name even
//! when the module declares the same name for something else.
//!
//! A field name that a standard library type also declares keeps its text
//! where it names a field of a struct, event, error or variant (after `.`,
//! as a label, in a `struct` block), and is renamed where it names a local,
//! a parameter or a field of a `state` or `actor`, which the steps name by
//! its bare name. Such a name is placed by its first declaration that is
//! renamed, so the two are projected apart and the projection stays
//! idempotent.

use crate::parse::{lex, Tok, DRAFT06_KEYWORDS, KEYWORDS};
use std::collections::{BTreeMap, HashSet};

const BUILTIN_TYPES: &[&str] = &[
    "bool", "text", "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64", "usize", "f32", "f64",
    "List", "Map", "Set", "Box", "Option", "Secret", "Cap", "Guarded", "Text", "World", "Template",
    "Shell", "Prompt", "Header", "mathint", "Tick", "self", "some", "none", "true", "false",
    // semantic names: the entry point, the built-in traits and their methods
    "main", "Eq", "Ord", "Hash", "Show", "Copy", "Data", "Merge", "eq", "cmp", "hash", "show", "merge", "bottom",
    "Start", "Input", "Display", "Exit", "Synced", "Backpressure", "Deadlock",
    // the authority structs of foreign file access (F.5.1, W-CAP-3)
    "DirRead", "DirWrite",
    // per-program constants the checker reads by name (6.7, C.11.11)
    "LOOP_LIMIT", "RESTART_WINDOW", "TICK_NS",
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

/// What the scan finds about the module's own names.
struct Names {
    /// declared name → positional name
    map: BTreeMap<String, String>,
    /// names declared as fields of the module's own types (struct, state,
    /// event and error fields, kind variant payloads)
    fields: HashSet<String>,
    /// names declared as methods in the module's own `impl` blocks
    methods: HashSet<String>,
    /// every type, event, error, region, variant and function the module
    /// declares, including those whose names are reserved
    own: HashSet<String>,
    /// the module's own functions (their labels are parameters)
    fns: HashSet<String>,
    /// the lines that declare a field of one of the module's `struct`s: such
    /// a field is reached only after `.` or as a label, where a field name of
    /// an imported type keeps its name, so its declaration keeps it too. A
    /// field of a `state` or `actor` is used by its bare name in the steps
    /// and handlers, so it is renamed with its uses.
    struct_field_lines: HashSet<usize>,
    /// field names declared by the standard library modules the text uses
    /// (and `root`, read by name, beside DirRead or DirWrite)
    foreign: HashSet<String>,
}

/// What may keep a field declaration's name as written
enum Hint {
    None,
    /// `NAME:` at a line start in a `struct` block
    StructField,
    /// `NAME:` inside the parentheses after the name given
    Label(String),
}

/// The rename map and the facts that place each use.
fn renames(src: &str) -> Result<Names, String> {
    let lexed = lex(src)?;
    let lines: Vec<usize> = lexed.iter().map(|(_, l)| *l).collect();
    let toks: Vec<Tok> = lexed.into_iter().map(|(t, _)| t).collect();
    let reserved: HashSet<&str> = KEYWORDS.iter().chain(DRAFT06_KEYWORDS.iter()).chain(BUILTIN_TYPES.iter()).copied().collect();
    // every declaration in order, with what may keep it as written: a
    // field of a `struct` declared at a line start, or a label after a name
    let mut raw: Vec<(String, Kind, Hint)> = vec![];
    // every name the module declares, reserved ones too (`event Input(…)`)
    let mut own: HashSet<String> = HashSet::new();
    let mut declare = |n: &str, k: Kind, h: Hint| {
        if k != Kind::Value {
            own.insert(n.to_string());
        }
        // `_` is the wildcard and a `_` prefix means "deliberately unused"
        // (F.3.2, W-UNUSED-1): both carry meaning and are kept as written.
        if n == "_" {
            return;
        }
        if !reserved.contains(n) {
            raw.push((n.to_string(), k, h));
        }
    };
    let mut in_kind_block = false;
    let mut kind_depth = 0usize;
    let mut depth = 0usize;
    // `impl` blocks: their `fn` members are methods, reached after `.`
    let mut in_impl = false;
    let mut impl_depth = 0usize;
    let mut methods: HashSet<String> = HashSet::new();
    // fields: `NAME :` at a line start in a type block, and the labels in
    // parentheses after a name (resolved below, once every name is known)
    let mut line_fields: HashSet<String> = HashSet::new();
    let mut labels: Vec<(String, String)> = vec![];
    let mut parens: Vec<Option<String>> = vec![];
    // the declaring word of each open indented block (`struct`, `state`, …)
    let mut blocks: Vec<Option<String>> = vec![];
    let mut header: Option<String> = None;
    let mut struct_field_lines: HashSet<usize> = HashSet::new();
    for i in 0..toks.len() {
        match &toks[i] {
            Tok::Nl | Tok::Indent | Tok::Dedent => {}
            _ if i == 0 || matches!(toks[i - 1], Tok::Nl | Tok::Indent | Tok::Dedent) => {
                header = if let Tok::Id(w) = &toks[i] { Some(w.clone()) } else { None };
            }
            _ => {}
        }
        match &toks[i] {
            Tok::P("{") => {
                // a map or set literal: `NAME:` in it is a key, not a label
                parens.push(None);
            }
            Tok::P("}") => {
                parens.pop();
            }
            Tok::P("(") => {
                let callee = match (i.checked_sub(2).map(|k| &toks[k]), i.checked_sub(1).map(|k| &toks[k])) {
                    (Some(Tok::P(".")), _) => None,
                    (_, Some(Tok::Id(c))) => Some(c.clone()),
                    _ => None,
                };
                parens.push(callee);
            }
            Tok::P(")") => {
                parens.pop();
            }
            Tok::Indent => {
                depth += 1;
                blocks.push(header.take());
            }
            Tok::Dedent => {
                depth = depth.saturating_sub(1);
                blocks.pop();
                if in_kind_block && depth < kind_depth {
                    in_kind_block = false;
                }
                if in_impl && depth < impl_depth {
                    in_impl = false;
                }
            }
            Tok::Id(n) => {
                let prev = if i > 0 { Some(&toks[i - 1]) } else { None };
                let next = toks.get(i + 1);
                let prev_kw = |k: &str| matches!(prev, Some(Tok::Id(p)) if p == k);
                if prev_kw("state") || prev_kw("actor") {
                    declare(n, Kind::Region, Hint::None);
                } else if prev_kw("event") {
                    declare(n, Kind::Event, Hint::None);
                } else if prev_kw("error") {
                    declare(n, Kind::Error, Hint::None);
                } else if prev_kw("impl") {
                    in_impl = true;
                    impl_depth = depth + 1;
                } else if prev_kw("fn") {
                    if in_impl && depth == impl_depth {
                        methods.insert(n.clone());
                    }
                    declare(n, Kind::Fn, Hint::None);
                } else if prev_kw("struct") || prev_kw("trait") || prev_kw("type") || prev_kw("protocol") {
                    declare(n, Kind::Type, Hint::None);
                } else if prev_kw("kind") {
                    declare(n, Kind::Type, Hint::None);
                    in_kind_block = true;
                    kind_depth = depth + 1;
                } else if prev_kw("const") || prev_kw("let") || prev_kw("var") || prev_kw("for") {
                    declare(n, Kind::Value, Hint::None);
                } else if in_kind_block && depth == kind_depth && matches!(prev, Some(Tok::Nl) | Some(Tok::Indent)) {
                    declare(n, Kind::Variant, Hint::None);
                } else if matches!(next, Some(Tok::P(":")))
                    && matches!(prev, Some(Tok::Nl) | Some(Tok::Indent) | Some(Tok::P("(")) | Some(Tok::P(",")) | Some(Tok::P("|")))
                {
                    // a field, a parameter, a payload field or a struct-literal key
                    let mut hint = Hint::None;
                    if matches!(prev, Some(Tok::Nl) | Some(Tok::Indent)) {
                        line_fields.insert(n.clone());
                        if blocks.last().is_some_and(|b| b.as_deref() == Some("struct")) {
                            struct_field_lines.insert(lines[i]);
                            hint = Hint::StructField;
                        }
                    } else if let Some(Some(c)) = parens.last() {
                        labels.push((c.clone(), n.clone()));
                        hint = Hint::Label(c.clone());
                    }
                    declare(n, Kind::Value, hint);
                } else if matches!(prev, Some(Tok::P(":"))) && matches!(next, Some(Tok::P(")")) | Some(Tok::P(","))) && !reserved.contains(n.as_str()) {
                    // the binding side of a payload pattern: `Deposit(amount: a)`
                    // (a type in this position would be a declared type or built-in)
                    if n.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
                        declare(n, Kind::Value, Hint::None); // idempotent: only a name's first declaration places it
                    }
                }
            }
            _ => {}
        }
    }
    // the field names that reach the module from the standard library
    let mut foreign = foreign_fields(src);
    // F.5.1: the checker reads the field `root` of DirRead and DirWrite by name
    if own.contains("DirRead") || own.contains("DirWrite") {
        foreign.insert("root".to_string());
    }
    // A field declaration that keeps its name (a struct field or a label of
    // one of the module's own types, events, errors or variants, named like
    // a field of an imported type) does not place the name: the name is
    // placed by its first declaration that is renamed, or, when every one is
    // kept, is not renamed at all. Two declarations of one name, one kept
    // and one renamed, are then two names, as they are in the source; and
    // the projection of a projection places every name where it did.
    let first_kind: BTreeMap<&str, Kind> = raw.iter().rev().map(|(n, k, _)| (n.as_str(), *k)).collect();
    let kept = |n: &str, h: &Hint| {
        foreign.contains(n)
            && match h {
                Hint::None => false,
                Hint::StructField => true,
                Hint::Label(c) => own.contains(c) && first_kind.get(c.as_str()) != Some(&Kind::Fn),
            }
    };
    let mut seen: HashSet<&str> = HashSet::new();
    let mut declared: Vec<(String, Kind)> = vec![];
    for (n, k, h) in &raw {
        if !kept(n, h) && seen.insert(n.as_str()) {
            declared.push((n.clone(), *k));
        }
    }
    let mut counters: BTreeMap<Kind, usize> = BTreeMap::new();
    let mut map = BTreeMap::new();
    // Identifiers the source uses and this projection does not rename (keywords,
    // built-ins, library names): a positional name must not collide with one.
    // A declared name is never a collision, which is what makes the projection
    // idempotent: `e0` in the input is renamed to `e0` again.
    let declared_set: HashSet<&str> = declared.iter().map(|(n, _)| n.as_str()).collect();
    let fixed_ids: HashSet<String> = toks
        .iter()
        .filter_map(|t| if let Tok::Id(s) = t { Some(s.clone()) } else { None })
        .filter(|s| !declared_set.contains(s.as_str()))
        .collect();
    // a label is a field when its callee is one of the module's own types,
    // variants, events or errors (a function's labels are its parameters)
    let kinds: BTreeMap<&str, Kind> = declared.iter().map(|(n, k)| (n.as_str(), *k)).collect();
    let mut fields = line_fields;
    for (c, n) in labels {
        if own.contains(&c) && kinds.get(c.as_str()) != Some(&Kind::Fn) {
            fields.insert(n);
        }
    }
    for (n, k) in &declared {
        loop {
            let c = counters.entry(*k).or_insert(0);
            let cand = if n.starts_with('_') { format!("_{}{}", prefix(*k), *c) } else { format!("{}{}", prefix(*k), *c) };
            *c += 1;
            if !fixed_ids.contains(&cand) {
                map.insert(n.clone(), cand);
                break;
            }
        }
    }
    let fns = declared.iter().filter(|(_, k)| *k == Kind::Fn).map(|(n, _)| n.clone()).collect();
    Ok(Names { map, fields, methods, own, fns, struct_field_lines, foreign })
}

/// The module paths of a text's `use` lines.
fn use_paths(src: &str) -> Vec<String> {
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("use "))
        .filter_map(|r| r.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '.')).next())
        .map(|p| p.trim_end_matches('.').to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// The field names of the types declared by the standard library modules a
/// text uses, and by the modules those use: a value of such a type can reach
/// the module, and its fields are named by that module, not by this one.
fn foreign_fields(src: &str) -> HashSet<String> {
    use crate::ast::Item;
    let mut out = HashSet::new();
    let mut todo = use_paths(src);
    let mut done: HashSet<String> = HashSet::new();
    while let Some(p) = todo.pop() {
        if !done.insert(p.clone()) {
            continue;
        }
        let (Some(text), Some(m)) = (crate::stdlib::source(&p), crate::stdlib::module(&p)) else { continue };
        todo.extend(use_paths(text));
        for it in &m.items {
            match it {
                Item::Struct { fields, .. } | Item::State { fields, .. } => out.extend(fields.iter().map(|f| f.name.clone())),
                Item::Event { payload, .. } | Item::Error { payload, .. } => out.extend(payload.iter().map(|(n, _)| n.clone())),
                Item::Kind { variants, .. } => {
                    out.extend(variants.iter().flat_map(|(_, ps)| ps.iter().map(|(n, _)| n.clone())))
                }
                _ => {}
            }
        }
    }
    out
}

/// An expression inside a template's `{…}`: the same rules as in code for
/// a member after `.` (5.15) and for every other declared name.
fn project_expr(
    e: &str,
    map: &BTreeMap<String, String>,
    fields: &HashSet<String>,
    methods: &HashSet<String>,
    foreign: &HashSet<String>,
) -> String {
    let b = e.as_bytes();
    let mut out = String::with_capacity(e.len());
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let word = &e[start..i];
            let after_dot = start > 0 && b[start - 1] == b'.' && !(start > 1 && b[start - 2] == b'.');
            let keep = if after_dot && b.get(i) == Some(&b'(') {
                !methods.contains(word)
            } else if after_dot {
                foreign.contains(word) || (map.get(word).is_some_and(|r| r.starts_with('v')) && !fields.contains(word))
            } else {
                false
            };
            match map.get(word) {
                Some(r) if !keep => out.push_str(r),
                _ => out.push_str(word),
            }
        } else if c.is_ascii_digit() {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_' || (b[i] == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit))) {
                i += 1;
            }
            out.push_str(&e[start..i]);
        } else {
            let ch = e[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// Rewrite identifiers outside strings and comments by the map.
pub fn nameless(src: &str) -> Result<String, String> {
    let Names { map, fields, methods, own, fns, struct_field_lines, foreign } = renames(src)?;
    // the byte offset at which each line starts (line 1 at index 0)
    let line_starts: Vec<usize> = std::iter::once(0).chain(src.match_indices('\n').map(|(k, _)| k + 1)).collect();
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    // for each open parenthesis, what it follows: one of the module's own
    // functions (its labels are parameters), one of its own types, variants,
    // events or errors (its labels are fields), or another module's name
    #[derive(Clone, Copy, PartialEq)]
    enum Callee {
        OwnFn,
        OwnType,
        Other,
        /// a map or set literal `{…}`: `NAME:` in it is a key expression
        Brace,
    }
    let mut parens: Vec<Callee> = vec![];
    let mut last_word: Option<(usize, Callee)> = None; // end, and what it names
    // `NAME[…](`: generic arguments sit between a name and its parentheses
    let mut brackets: Vec<Option<Callee>> = vec![];
    while i < b.len() {
        let c = b[i] as char;
        if c == '[' {
            brackets.push(match last_word {
                Some((e, k)) if e == i => Some(k),
                _ => None,
            });
            out.push(c);
            i += 1;
        } else if c == ']' {
            if let Some(Some(k)) = brackets.pop() {
                last_word = Some((i + 1, k));
            }
            out.push(c);
            i += 1;
        } else if c == '(' {
            parens.push(match last_word {
                Some((e, k)) if e == i => k,
                _ => Callee::Other,
            });
            out.push(c);
            i += 1;
        } else if c == ')' {
            parens.pop();
            out.push(c);
            i += 1;
        } else if c == '{' {
            parens.push(Callee::Brace);
            out.push(c);
            i += 1;
        } else if c == '}' {
            parens.pop();
            out.push(c);
            i += 1;
        } else if c == '"' && i > 0 && b[i - 1] == b't' && (i < 2 || !(b[i - 2].is_ascii_alphanumeric() || b[i - 2] == b'_')) {
            // a template literal `t"…{expr}…"` (F.6.1): the text is copied as
            // written, and each `{expr}` is projected like code (`{{` and `}}`
            // are literal braces)
            out.push('"');
            i += 1;
            while i < b.len() && b[i] != b'"' {
                if b[i] == b'\\' && i + 1 < b.len() {
                    out.push_str(&src[i..i + 2]);
                    i += 2;
                } else if (b[i] == b'{' || b[i] == b'}') && b.get(i + 1) == Some(&b[i]) {
                    out.push_str(&src[i..i + 2]);
                    i += 2;
                } else if b[i] == b'{' {
                    let end = src[i..].find('}').map_or(b.len(), |k| i + k);
                    out.push('{');
                    out.push_str(&project_expr(&src[i + 1..end], &map, &fields, &methods, &foreign));
                    i = end;
                } else {
                    let ch = src[i..].chars().next().unwrap();
                    out.push(ch);
                    i += ch.len_utf8();
                }
            }
            if i < b.len() {
                out.push('"');
                i += 1;
            }
        } else if c == '"' {
            // string literal body: copy verbatim with escapes
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
        } else if c.is_ascii_digit() {
            // a numeric literal is copied verbatim: its letters (an exponent
            // `e`, a radix prefix or digit) are not names, so `1.5e-3` keeps
            // its `e` even when the program declares a name `e`
            let start = i;
            while i < b.len() {
                let x = b[i];
                let exp_sign = (x == b'+' || x == b'-')
                    && matches!(b[i - 1], b'e' | b'E')
                    && i + 1 < b.len()
                    && b[i + 1].is_ascii_digit()
                    && !src[start..i].starts_with("0x");
                if x.is_ascii_alphanumeric()
                    || x == b'_'
                    || (x == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit())
                    || exp_sign
                {
                    i += 1;
                } else {
                    break;
                }
            }
            out.push_str(&src[start..i]);
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            let word = &src[start..i];
            // a `use` line names modules by path, and a module path is not a
            // declaration of this program, so it keeps its names
            let line_start = src[..start].rfind('\n').map_or(0, |k| k + 1);
            if src[line_start..start].trim_start().starts_with("use ") {
                out.push_str(word);
                continue;
            }
            // `..` is a range, not a member access
            let after_dot = start > 0 && b[start - 1] == b'.' && !(start > 1 && b[start - 2] == b'.');
            let called = b.get(i) == Some(&b'(');
            let rest = src[i..].trim_start_matches([' ', '\t']);
            let before = src[..start].trim_end_matches([' ', '\t']);
            let line_head = before.is_empty() || before.ends_with('\n');
            // inside brackets a line break is not a token, so a label may
            // start a line of its own as well as follow `(` or `,`
            let label = rest.starts_with(':')
                && !rest.starts_with("::")
                && (matches!(before.chars().last(), Some('(') | Some(',')) || (line_head && !parens.is_empty()));
            let keep = if after_dot && called {
                // `x.m(…)` reaches the module's own function only as a method
                // of an `impl` block (5.15); otherwise it is a standard
                // library method or a function of another module, which keep
                // their names whatever this module declares
                !methods.contains(word)
            } else if after_dot {
                // `x.f` names a field: one of the module's own types, or of a
                // type another module declares, which keeps its name; a field
                // name an imported type also declares keeps it everywhere it
                // names a field, as the receiver's type is not known here
                foreign.contains(word) || (map.get(word).is_some_and(|r| r.starts_with('v')) && !fields.contains(word))
            } else if label && !parens.is_empty() {
                // `NAME:` inside parentheses labels a parameter or field of
                // the callee: renamed only when the callee is the module's own
                match parens.last().copied().unwrap_or(Callee::Other) {
                    Callee::OwnFn | Callee::Brace => false,
                    Callee::OwnType => foreign.contains(word),
                    Callee::Other => true,
                }
            } else if rest.starts_with(':') && !rest.starts_with("::") && src[line_start..start].trim().is_empty() {
                // a field declared at the start of a line in a type block: a
                // struct field keeps a name an imported type's field also
                // has, as its uses after `.` and as a label do; a field of a
                // state or actor is renamed, as its bare uses are
                let line = line_starts.partition_point(|&k| k <= start);
                foreign.contains(word) && struct_field_lines.contains(&line)
            } else {
                false
            };
            let local = (map.contains_key(word) || own.contains(word)) && (!after_dot || methods.contains(word));
            let callee = if word == "fn" && !after_dot {
                // a lambda `fn(x: T) => …`: its parameters are its own
                Callee::OwnFn
            } else if !local {
                Callee::Other
            } else if fns.contains(word) || src[..start].trim_end().strip_suffix("fn").is_some_and(|p| p.ends_with(|c: char| !c.is_ascii_alphanumeric() && c != '_') || p.is_empty()) {
                // a function of the module, or any name declared by `fn` here
                Callee::OwnFn
            } else {
                Callee::OwnType
            };
            last_word = Some((i, callee));
            match map.get(word) {
                Some(r) if !keep => out.push_str(r),
                _ => out.push_str(word),
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
    Ok(out)
}
