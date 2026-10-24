//! The standard library, built in (4.1): every module of `stdlib/lib` is
//! compiled into the checker by build.rs and resolves by `use`, like
//! `std.text`. A library module is parsed once, on first use, with its
//! `check`, `example` and `intent` items dropped (its own checks run in the
//! library's test suite, not in every program that imports it). There is no
//! directive, flag or feature: a program names a module and the checker has
//! it, and the library carries no version (F.5.3: it is pinned by hash).
use crate::ast::{Item, Module};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

include!(concat!(env!("OUT_DIR"), "/stdlib_gen.rs"));

/// The module paths of the built-in library, in sorted order.
pub fn names() -> Vec<&'static str> {
    let mut v: Vec<&str> = LIB.iter().map(|(n, _)| *n).collect();
    v.sort();
    v
}

pub fn has(path: &str) -> bool {
    LIB.iter().any(|(n, _)| *n == path)
}

/// The source text of a library module.
pub fn source(path: &str) -> Option<&'static str> {
    LIB.iter().find(|(n, _)| *n == path).map(|(_, s)| *s)
}

/// The parsed library module, without its checks, examples and intents.
pub fn module(path: &str) -> Option<&'static Module> {
    static PARSED: OnceLock<Mutex<HashMap<String, &'static Module>>> = OnceLock::new();
    let cache = PARSED.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(m) = cache.lock().unwrap().get(path) {
        return Some(m);
    }
    let src = source(path)?;
    let mut m = crate::parse::parse(src).ok()?;
    m.items.retain(|it| !matches!(it, Item::Check(_) | Item::Example(_) | Item::Intent(..)));
    let leaked: &'static Module = Box::leak(Box::new(m));
    cache.lock().unwrap().insert(path.to_string(), leaked);
    Some(leaked)
}

/// The libraries a `#!lib NAME` directive line names (the form the shipped
/// binary accepts: the library's items are placed in front of the program,
/// unqualified; `pq`, `hpke` and `mls` bring `crypto` with them). The line
/// is a comment to every other tool.
pub fn directives(src: &str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = vec![];
    for l in src.lines() {
        if let Some(n) = l.strip_prefix("#!lib ") {
            let n = n.trim();
            if matches!(n, "pq" | "hpke" | "mls") && !out.contains(&"crypto") {
                out.push("crypto");
            }
            if let Some((name, _)) = LIB.iter().find(|(k, _)| *k == n) {
                if !out.contains(name) {
                    out.push(name);
                }
            }
        }
    }
    out
}

/// The result of placing directive libraries in front of a module: the
/// module, and the names of the library functions and types it now carries,
/// which the checker treats as imported (their bodies are checked by the
/// library's own suite, not under the program's names).
pub struct WithLibs {
    pub module: Module,
    pub fns: std::collections::HashSet<String>,
    pub types: std::collections::HashSet<String>,
}

/// `m` with the items of the directive libraries in front (checks, examples,
/// intents and module lines dropped; an item `m` declares itself wins).
pub fn with_directives(m: &Module, src: &str) -> WithLibs {
    let libs = directives(src);
    let mut fns = std::collections::HashSet::new();
    let mut types = std::collections::HashSet::new();
    if libs.is_empty() {
        return WithLibs { module: m.clone(), fns, types };
    }
    let own: std::collections::HashSet<String> = m.items.iter().filter_map(crate::link::item_name).collect();
    let mut seen = own.clone();
    let mut items = vec![];
    for l in libs {
        if let Some(lm) = module(l) {
            for it in &lm.items {
                if matches!(it, Item::Use(..)) {
                    continue;
                }
                if let Some(n) = crate::link::item_name(it) {
                    if !seen.insert(n.clone()) {
                        continue;
                    }
                    match it {
                        Item::Fn(_) | Item::Extern { .. } => fns.insert(n),
                        _ => types.insert(n),
                    };
                } else if let Item::Impl { ty, .. } = it {
                    types.insert(crate::check::ty_head(ty));
                }
                items.push(it.clone());
            }
        }
    }
    items.extend(m.items.iter().cloned());
    WithLibs { module: Module { items, ..m.clone() }, fns, types }
}
