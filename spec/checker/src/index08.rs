//! Version 0.8 indexed values (spec 5.16; CORE-PROPERTIES.md P1; Lean
//! RCE/Label.lean): `Text[C]` context-typed text, `Secret[τ]` opaque
//! secrets, `Cap[A]` capabilities and `Guarded[τ, A]` state fields.
//! This module holds the encoders (shared by the reference evaluator) and
//! the declaration-level rules W-CTX-1, W-SEC-1, W-CAP-1, W-CAP-2, which
//! are syntactic over the module; the expression rules are in check.rs.
use crate::ast::*;
use std::collections::{HashMap, HashSet};

/// The closed set of contexts (5.16).
pub const CTX: &[&str] = &["Html", "Url", "Sql", "Json"];
/// The contexts a version 0.9 module adds (F.5.2). `ShellArg` and `Prompt`
/// have `lit` and `encode`; `Shell` is made only by `Shell.cmd`.
pub const CTX09: &[&str] = &["ShellArg", "Prompt", "Shell", "Header"];

/// Whether `c` names a context in a module of the given version.
pub fn is_ctx(c: &str) -> bool {
    CTX.contains(&c) || CTX09.contains(&c)
}

/// The deterministic data-block wrapper of `Prompt.encode` (F.5.2): the
/// delimiter is derived from the SHA-256 of the field and is extended by
/// re-hashing until the field text does not contain it.
pub fn prompt_encode(s: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut h: Vec<u8> = Sha256::digest(s.as_bytes()).to_vec();
    loop {
        let hex: String = h.iter().map(|b| format!("{b:02x}")).collect();
        let d = format!("data-{hex}");
        if !s.contains(&d) {
            return format!("<<{d}\n{s}\n{d}>>");
        }
        h = Sha256::digest(&h).to_vec();
    }
}

/// `Shell.cmd(program, args)` as canonical JSON text: the host runs exactly
/// this argument vector, with no shell parsing (F.5.2).
pub fn shell_cmd(program: &str, args: &[String]) -> String {
    let mut o = String::from("{\"args\":[");
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            o.push(',');
        }
        o.push_str(&encode("Json", a));
    }
    o.push_str("],\"program\":");
    o.push_str(&encode("Json", program));
    o.push('}');
    o
}

/// Type names that are indexed types in a 0.8 module.
pub const INDEXED: &[&str] = &["Text", "Secret", "Cap", "Guarded"];

/// Encode text for a context (5.16, normative; every runtime agrees).
pub fn encode(ctx: &str, s: &str) -> String {
    match ctx {
        "ShellArg" => s.to_string(),
        "Prompt" => prompt_encode(s),
        // F.5.7: a header value is its text; CR and LF trap in the caller
        "Header" => s.to_string(),
        "Html" => {
            let mut o = String::new();
            for c in s.chars() {
                match c {
                    '&' => o.push_str("&amp;"),
                    '<' => o.push_str("&lt;"),
                    '>' => o.push_str("&gt;"),
                    '"' => o.push_str("&quot;"),
                    '\'' => o.push_str("&#39;"),
                    c => o.push(c),
                }
            }
            o
        }
        "Url" => {
            let mut o = String::new();
            for b in s.bytes() {
                if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
                    o.push(b as char);
                } else {
                    o.push_str(&format!("%{b:02X}"));
                }
            }
            o
        }
        "Sql" => format!("'{}'", s.replace('\'', "''")),
        _ => {
            let mut o = String::from("\"");
            for c in s.chars() {
                match c {
                    '"' => o.push_str("\\\""),
                    '\\' => o.push_str("\\\\"),
                    '\n' => o.push_str("\\n"),
                    '\r' => o.push_str("\\r"),
                    '\t' => o.push_str("\\t"),
                    c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
                    c => o.push(c),
                }
            }
            o.push('"');
            o
        }
    }
}

fn head(t: &Ty) -> Option<(&str, &Vec<Ty>)> {
    match t {
        Ty::Path(p, a) if p.len() == 1 => Some((p[0].as_str(), a)),
        _ => None,
    }
}

/// Whether t mentions the indexed type `name` anywhere.
pub fn mentions(t: &Ty, name: &str) -> bool {
    match t {
        Ty::Path(p, a) => (p.len() == 1 && p[0] == name) || a.iter().any(|x| mentions(x, name)),
        Ty::Tuple(v) => v.iter().any(|x| mentions(x, name)),
        Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Region(x, _) | Ty::Array(x, _) => mentions(x, name),
        Ty::Union(a, b) => mentions(a, name) || mentions(b, name),
        Ty::Fn(a, r) => a.iter().any(|x| mentions(x, name)) || mentions(r, name),
        Ty::Unit => false,
    }
}

/// Whether t is `Cap[A]` at the top (optionally under one option `?`... no:
/// exactly `Cap[A]`).
fn is_cap(t: &Ty) -> Option<String> {
    match head(t) {
        Some(("Cap", a)) if a.len() == 1 => head(&a[0]).map(|(n, _)| n.to_string()),
        _ => None,
    }
}

/// A struct transitively mentions Secret (so emitting it would expose it).
pub fn struct_secret(m: &Module) -> HashSet<String> {
    let mut out: HashSet<String> = HashSet::new();
    loop {
        let before = out.len();
        for it in &m.items {
            let (name, tys): (&String, Vec<&Ty>) = match it {
                Item::Struct { name, fields, .. } => (name, fields.iter().map(|f| &f.ty).collect()),
                Item::Kind { name, variants, .. } => (name, variants.iter().flat_map(|(_, p)| p.iter().map(|(_, t)| t)).collect()),
                _ => continue,
            };
            if tys.iter().any(|t| mentions(t, "Secret") || out.iter().any(|s| mentions(t, s))) {
                out.insert(name.clone());
            }
        }
        if out.len() == before {
            return out;
        }
    }
}

const READS: &[&str] = &[
    "len", "get", "contains", "isempty", "keys", "values", "tostr", "map", "filter", "fold", "sum", "utf8",
    "with", "or", "issome", "isnone", "abs", "max", "min", "startswith", "endswith", "split", "lines",
    "words", "chars", "strip", "upper", "lower", "toint", "sha256", "isdigit", "isalpha", "isspace",
    "join", "checked_add", "checked_sub", "checked_mul", "checked_div", "saturating_add",
    "saturating_sub", "saturating_mul", "wrapping_add", "wrapping_sub", "wrapping_mul", "ceil",
    "floor", "round", "sqrt", "collect", "raw",
];

fn writes(body: &Body, field: &str, only_field: bool) -> bool {
    let mut w = false;
    let mut on_expr = |e: &Expr, w: &mut bool| {
        crate::check::walk_expr(e, &mut |x| match x {
            Expr::Borrow(true, p, _) if crate::check::place_root(p).as_deref() == Some(field) => *w = true,
            Expr::Call(f, _, _, _) => {
                if let Expr::Member(r, m, _) = &**f
                    && crate::check::place_root(r).as_deref() == Some(field)
                    && !READS.contains(&m.as_str())
                {
                    *w = true;
                }
            }
            _ => {}
        });
    };
    match body {
        Body::Arrow(e) => {
            if only_field {
                return true;
            }
            on_expr(e, &mut w);
        }
        Body::Block(b) => crate::check::walk_stmts(b, &mut |s| {
            if let Stmt::Assign(_, p, _) = s
                && crate::check::place_root(p).as_deref() == Some(field)
            {
                w = true;
            }
            for e in crate::check::stmt_direct_exprs(s) {
                on_expr(e, &mut w);
            }
        }),
    }
    w
}

fn pat_events(p: &Pat, out: &mut Vec<String>) -> bool {
    match p {
        Pat::Wild => false,
        Pat::Path(v) | Pat::Ctor(v, _) => {
            out.push(v.last().cloned().unwrap_or_default());
            true
        }
        Pat::Alt(v) => v.iter().all(|x| pat_events(x, out)),
        _ => false,
    }
}

/// The declaration-level rules of 5.16 for a version 0.8 module.
pub fn check(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    let mut err = |l: usize, r: &str, s: String| errs.push(format!("line {l}: {r} {s}"));
    let structs: HashSet<String> = m.items.iter().filter_map(|i| if let Item::Struct { name, .. } = i { Some(name.clone()) } else { None }).collect();
    // the indexed type names and contexts are the language's in 0.8
    for it in &m.items {
        let (n, l) = match it {
            Item::Struct { name, line, .. } | Item::Kind { name, line, .. } | Item::TypeAlias { name, line, .. }
            | Item::State { name, line, .. } | Item::Event { name, line, .. } | Item::Error { name, line, .. }
            | Item::Actor { name, line, .. } | Item::Trait { name, line, .. } => (name, *line),
            Item::Fn(fd) => (&fd.sig.name, fd.sig.line),
            _ => continue,
        };
        if INDEXED.contains(&n.as_str()) || is_ctx(n) {
            err(l, "W-NAME-2", format!("`{n}` is a version 0.8 indexed type or context (5.16) and cannot be declared"));
        }
    }
    // well-formed indexed types wherever a type is written
    let mut all: Vec<(usize, &Ty, &'static str)> = vec![];
    for it in &m.items {
        match it {
            Item::Struct { fields, line, .. } => fields.iter().for_each(|f| all.push((*line, &f.ty, "struct field"))),
            Item::State { fields, line, .. } | Item::Actor { fields, line, .. } => fields.iter().for_each(|f| all.push((*line, &f.ty, "state"))),
            Item::Event { payload, line, .. } => payload.iter().for_each(|(_, t)| all.push((*line, t, "event"))),
            Item::Error { payload, line, .. } => payload.iter().for_each(|(_, t)| all.push((*line, t, "error"))),
            Item::Kind { variants, line, .. } => variants.iter().flat_map(|(_, p)| p.iter()).for_each(|(_, t)| all.push((*line, t, "variant"))),
            Item::TypeAlias { ty, line, .. } => all.push((*line, ty, "alias")),
            Item::Fn(fd) => {
                for p in &fd.sig.params {
                    if let Param::Named { ty, .. } = p {
                        all.push((fd.sig.line, ty, "param"));
                    }
                }
                if let Some(r) = &fd.sig.ret {
                    all.push((fd.sig.line, r, "result"));
                }
            }
            Item::Extern { sig, line, .. } => {
                for p in &sig.params {
                    if let Param::Named { ty, .. } = p {
                        all.push((*line, ty, "foreign"));
                    }
                }
            }
            _ => {}
        }
    }
    fn each<'t>(t: &'t Ty, f: &mut dyn FnMut(&'t Ty)) {
        f(t);
        match t {
            Ty::Path(_, a) | Ty::Tuple(a) => a.iter().for_each(|x| each(x, f)),
            Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Region(x, _) | Ty::Array(x, _) => each(x, f),
            Ty::Union(a, b) => { each(a, f); each(b, f) }
            Ty::Fn(a, r) => { a.iter().for_each(|x| each(x, f)); each(r, f) }
            Ty::Unit => {}
        }
    }
    for (l, t, what) in &all {
        each(t, &mut |x| match head(x) {
            Some(("Text", a)) => {
                if a.len() != 1 || !matches!(head(&a[0]), Some((c, b)) if is_ctx(c) && b.is_empty()) {
                    err(*l, "W-CTX-1", format!("`Text` takes one context, one of {}", CTX.join(", ")));
                }
            }
            Some(("Secret", a)) if a.len() != 1 => err(*l, "W-SEC-1", "`Secret` takes one type".into()),
            Some(("Cap", a)) | Some(("Guarded", a)) => {
                let want = if matches!(head(x), Some(("Cap", _))) { 1 } else { 2 };
                let auth = a.last().and_then(head).map(|(n, _)| n.to_string());
                if a.len() != want || !auth.as_ref().is_some_and(|n| structs.contains(n)) {
                    err(*l, "W-CAP-1", format!("the authority of `{}` must be a declared struct", head(x).unwrap().0));
                }
            }
            _ => {}
        });
        // placement
        if *what == "event" && mentions(t, "Text") {
            err(*l, "W-CTX-1", "an event payload carries plain text; context-typed text is made only by literals and encoders".into());
        }
        if matches!(*what, "error" | "foreign") && mentions(t, "Secret") {
            err(*l, "W-SEC-1", format!("a secret cannot leave through an {} ({} payloads are outputs)", if *what == "error" { "error payload" } else { "foreign call" }, *what));
        }
        if mentions(t, "Secret") {
            each(t, &mut |x| if let Ty::Chan(i) = x && mentions(i, "Secret") { err(*l, "W-SEC-1", "a channel cannot carry a secret".into()) });
        }
        let top_cap = is_cap(t).is_some() && matches!(*what, "event" | "param");
        if mentions(t, "Cap") && !top_cap {
            err(*l, "W-CAP-1", format!("a capability lives for one event: `Cap` appears only as an event payload field or a parameter type, not in a {what}"));
        }
        let top_guard = matches!(head(t), Some(("Guarded", _))) && *what == "state";
        if mentions(t, "Guarded") && !top_guard {
            err(*l, "W-CAP-1", "`Guarded` is the type of a state field only".into());
        }
    }
    // W-CAP-2: a step that writes a guarded field matches only events that carry its capability
    let mut guarded: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let mut nfields: HashMap<String, usize> = HashMap::new();
    let mut caps: HashMap<String, HashSet<String>> = HashMap::new();
    let mut events: Vec<String> = vec![];
    for it in &m.items {
        match it {
            Item::State { name, fields, .. } => {
                nfields.insert(name.clone(), fields.len());
                for f in fields {
                    if let Some(("Guarded", a)) = head(&f.ty)
                        && let Some((auth, _)) = a.last().and_then(head)
                    {
                        guarded.entry(name.clone()).or_default().push((f.name.clone(), auth.to_string()));
                    }
                }
            }
            Item::Event { name, payload, .. } => {
                events.push(name.clone());
                caps.insert(name.clone(), payload.iter().filter_map(|(_, t)| is_cap(t)).collect());
            }
            _ => {}
        }
    }
    for it in &m.items {
        if let Item::Step { region, pat, body, line, .. } = it
            && let Some(gs) = guarded.get(region)
        {
            let mut evs = vec![];
            if !pat_events(pat, &mut evs) {
                evs = events.clone();
            }
            for (f, auth) in gs {
                if writes(body, f, nfields.get(region) == Some(&1)) {
                    for e in &evs {
                        if !caps.get(e).is_some_and(|c| c.contains(auth)) {
                            err(*line, "W-CAP-2", format!("step writes `{region}.{f}` (Guarded by {auth}) on event `{e}`, which carries no Cap[{auth}]"));
                        }
                    }
                }
            }
        }
    }
    errs
}
