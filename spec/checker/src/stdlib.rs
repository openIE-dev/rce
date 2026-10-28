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

/// The library's manifest as embedded with it (`stdlib/lib/MANIFEST.json`,
/// F.5.3), without the one final line feed the file may carry.
pub fn manifest_text() -> &'static str {
    MANIFEST.strip_suffix('\n').unwrap_or(MANIFEST)
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
