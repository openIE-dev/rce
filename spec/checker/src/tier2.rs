//! Version 0.9 Tier 2 static rules (Appendix F.5): S2 file-system capabilities
//! (W-CAP-3, closures), S6 `model` foreign functions (W-BUD-1) and the model
//! result boundary (W-IFC-7, `extern "model" event`, F.5.6). Like `ifc`,
//! they run on the module as written and report findings that the checker
//! emits; nothing here changes what a program means.
use crate::ast::*;
use std::collections::{HashMap, HashSet};

fn err(l: usize, code: &str, m: String) -> String {
    format!("line {l}: {code} {m}")
}

fn head(t: &Ty) -> Option<(&str, &Vec<Ty>)> {
    match t {
        Ty::Path(p, a) if p.len() == 1 => Some((p[0].as_str(), a)),
        Ty::Ref(_, x) | Ty::Region(x, _) => head(x),
        _ => None,
    }
}

fn is_world(t: &Ty) -> bool {
    matches!(head(t), Some(("World", a)) if a.is_empty())
}

fn dir_cap(t: &Ty) -> Option<&str> {
    match head(t) {
        Some(("Cap", a)) if a.len() == 1 => match head(&a[0]) {
            Some((n @ ("DirRead" | "DirWrite"), _)) => Some(n),
            _ => None,
        },
        _ => None,
    }
}

fn holds_authority(t: &Ty) -> bool {
    is_world(t) || matches!(head(t), Some(("Cap", _)))
}

/// A foreign function takes a path when a parameter is named `path` or ends in `_path`.
fn path_param(p: &Param) -> bool {
    matches!(p, Param::Named { name, ty, .. }
        if (name == "path" || name.ends_with("_path")) && matches!(head(ty), Some(("text", _))))
}

/// F.5.1 W-CAP-3 and the closure rule of W-CAP-1.
pub fn s2(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    // DirRead and DirWrite are declared authority structs with `root: text`
    for it in &m.items {
        if let Item::Struct { name, fields, line, .. } = it
            && (name == "DirRead" || name == "DirWrite")
            && !fields.iter().any(|f| f.name == "root" && matches!(head(&f.ty), Some(("text", _))))
        {
            errs.push(err(*line, "W-CAP-3", format!("`{name}` is an authority struct with a `root: text` field (F.5.1)")));
        }
    }
    let declared: HashSet<&str> = m
        .items
        .iter()
        .filter_map(|i| if let Item::Struct { name, .. } = i { Some(name.as_str()) } else { None })
        .collect();
    for it in &m.items {
        if let Item::Extern { sig, line, abi } = it
            && abi != "model"
            && sig.params.iter().any(path_param)
        {
            let after_world = sig.params.iter().skip(1);
            let caps: Vec<&str> = after_world.filter_map(|p| if let Param::Named { ty, .. } = p { dir_cap(ty) } else { None }).collect();
            if caps.is_empty() {
                errs.push(err(*line, "W-CAP-3", format!("the foreign function `{}` reads or writes a path and takes no `Cap[DirRead]` or `Cap[DirWrite]` after its `&World` (F.5.1)", sig.name)));
            }
            for c in caps {
                if !declared.contains(c) {
                    errs.push(err(*line, "W-CAP-3", format!("`{c}` is not declared: declare `struct {c}` with `root: text`")));
                }
            }
        }
    }
    // a closure that captures `&World`, a `Cap` or labeled data (by declared type)
    let mut fns: Vec<&FnDecl> = vec![];
    for it in &m.items {
        match it {
            Item::Fn(f) => fns.push(f),
            Item::Impl { members, .. } => fns.extend(members.iter()),
            Item::Actor { fns: f, .. } => fns.extend(f.iter()),
            _ => {}
        }
    }
    for f in fns {
        let mut held: HashMap<String, String> = HashMap::new();
        for p in &f.sig.params {
            if let Param::Named { name, ty, .. } = p
                && holds_authority(ty)
            {
                held.insert(name.clone(), if is_world(ty) { "&World".into() } else { "a capability".into() });
            }
        }
        if let Body::Block(b) = &f.body {
            collect_let_caps(b, &mut held);
        }
        if held.is_empty() {
            continue;
        }
        let mut each = |e: &Expr| {
            crate::v09::each_expr(e, false, &mut |x, _| {
                if let Expr::Lambda(ps, body, l) = x {
                    let own: HashSet<&str> = ps.iter().filter_map(|p| if let Param::Named { name, .. } = p { Some(name.as_str()) } else { None }).collect();
                    crate::v09::each_expr(body, true, &mut |y, _| {
                        if let Expr::Name(n, _) = y
                            && !own.contains(n.as_str())
                            && let Some(what) = held.get(n)
                        {
                            errs.push(err(*l, "W-CAP-1", format!("a closure captures `{n}`, {what}: authority held by a closure must be visible in its capture list, which RCE does not have, so the closure requires change (F.5.1)")));
                        }
                    });
                }
            });
        };
        match &f.body {
            Body::Arrow(e) => each(e),
            Body::Block(b) => crate::v09::each_stmt_expr(b, &mut |e| each(e)),
        }
    }
    errs
}

fn collect_let_caps(b: &[Stmt], held: &mut HashMap<String, String>) {
    for s in b {
        if let Stmt::Let(Pat::Path(p), Some(t), _, _) = s
            && p.len() == 1
            && holds_authority(t)
        {
            held.insert(p[0].clone(), if is_world(t) { "&World".into() } else { "a capability".into() });
        }
    }
}

/// F.5.5 W-BUD-1: a `model` foreign function is called only inside `with budget`.
pub fn s6(m: &Module) -> Vec<String> {
    let models: HashSet<&str> = m
        .items
        .iter()
        .filter_map(|i| if let Item::Extern { abi, sig, .. } = i && abi == "model" { Some(sig.name.as_str()) } else { None })
        .collect();
    let mut errs = vec![];
    if models.is_empty() {
        return errs;
    }
    let mut fns: Vec<&FnDecl> = vec![];
    for it in &m.items {
        match it {
            Item::Fn(f) => fns.push(f),
            Item::Impl { members, .. } => fns.extend(members.iter()),
            Item::Actor { fns: f, .. } => fns.extend(f.iter()),
            _ => {}
        }
    }
    fn calls(e: &Expr, models: &HashSet<&str>, inside: bool, errs: &mut Vec<String>) {
        crate::v09::each_expr(e, false, &mut |x, _| {
            if let Expr::Call(f, _, l, _) = x
                && let Expr::Name(n, _) = &**f
                && models.contains(n.as_str())
                && !inside
            {
                errs.push(err(*l, "W-BUD-1", format!("`{n}` is a `model` foreign function: its cost is chosen by the outside, so it is called only inside `with budget` (F.5.5)")));
            }
        });
    }
    fn walk(ss: &[Stmt], models: &HashSet<&str>, inside: bool, errs: &mut Vec<String>) {
        for s in ss {
            match s {
                Stmt::Arena(n, _, body, _) if n == "$budget" => walk(body, models, true, errs),
                _ => {
                    for e in crate::v09::stmt_exprs(s) {
                        calls(e, models, inside, errs);
                    }
                    for b in crate::v09::stmt_blocks(s) {
                        walk(b, models, inside, errs);
                    }
                }
            }
        }
    }
    for f in fns {
        match &f.body {
            Body::Arrow(e) => calls(e, &models, false, &mut errs),
            Body::Block(b) => walk(b, &models, false, &mut errs),
        }
    }
    for it in &m.items {
        if let Item::Step { body, .. } = it {
            match body {
                Body::Arrow(e) => calls(e, &models, false, &mut errs),
                Body::Block(b) => walk(b, &models, false, &mut errs),
            }
        }
    }
    errs
}

/// The labels written on the top of a type, when any: `T @ {A, B}`.
fn written_labels(t: &Ty) -> Option<Vec<&str>> {
    match t {
        Ty::Region(_, s) => s.strip_prefix("@{").and_then(|x| x.strip_suffix('}')).map(|x| x.split(',').filter(|n| !n.is_empty()).collect()),
        _ => None,
    }
}

fn mentions_cap(t: &Ty) -> bool {
    match t {
        Ty::Path(p, a) => (p.len() == 1 && (p[0] == "Cap" || p[0] == "Guarded")) || a.iter().any(mentions_cap),
        Ty::Tuple(a) => a.iter().any(mentions_cap),
        Ty::Region(x, _) | Ty::Slice(x) | Ty::Chan(x) | Ty::Iter(x) | Ty::Ref(_, x) | Ty::Opt(x) | Ty::Array(x, _) => mentions_cap(x),
        Ty::Union(a, b) => mentions_cap(a) || mentions_cap(b),
        Ty::Fn(a, r) => a.iter().any(mentions_cap) || mentions_cap(r),
        Ty::Unit => false,
    }
}

/// F.5.6 W-IFC-7: what a model runtime answers is an input of no authority.
/// A field of an `extern "model" event` and the result of an `extern "model"`
/// function are `untrusted` (a label written on them names `untrusted`, so
/// `@ {}` cannot endorse them by declaration), and such an event carries no
/// `Cap` or `Guarded` value (authority is minted only by events the host
/// delivers from a grant, HC-5). Only the `model` ABI declares an event.
pub fn s8(m: &Module) -> Vec<String> {
    let mut errs = vec![];
    for it in &m.items {
        match it {
            Item::Event { name, payload, line, foreign: Some(abi), .. } => {
                if abi != "model" {
                    errs.push(err(*line, "W-FFI-1", format!("`extern \"{abi}\" event`: only a `model` runtime answers on a declared event (F.5.6)")));
                    continue;
                }
                for (f, t) in payload {
                    if mentions_cap(t) {
                        errs.push(err(*line, "W-IFC-7", format!("field `{f}` of the model result event `{name}` holds a capability: a model answer carries no authority (HC-5)")));
                    }
                    if let Some(ls) = written_labels(t)
                        && !ls.contains(&"untrusted")
                    {
                        errs.push(err(*line, "W-IFC-7", format!("field `{f}` of the model result event `{name}` is declared trusted: a model answer is `untrusted`; `endorse` it under a held capability")));
                    }
                }
            }
            Item::Extern { abi, sig, line } if abi == "model" => {
                if let Some(ls) = sig.ret.as_ref().and_then(written_labels)
                    && !ls.contains(&"untrusted")
                {
                    errs.push(err(*line, "W-IFC-7", format!("the result of the `model` foreign function `{}` is declared trusted: a model answer is `untrusted`; `endorse` it under a held capability", sig.name)));
                }
            }
            _ => {}
        }
    }
    errs
}
