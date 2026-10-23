//! The reference checker reads the spec's own data (W2, `spec/data/
//! spec-data.json`, generated from the book by `rce-spec-data`) for the
//! tables it enforces, rather than keeping a second copy by hand:
//!
//! * the 5.15 standard library table decides which builtin methods exist on
//!   each receiver (W-NAME-1 for any other, before the method is typed);
//! * the C.6.4 complexity classes decide `conc::method_work`;
//! * the language server offers the 5.15 methods after `.`.
//!
//! What is compiled in (`include_str!`) is `spec/data/spec-tables.json`,
//! the `methods` and `method_work` members of spec-data.json and nothing
//! else (`tables_of`), written by `rce-spec-data` next to it so the wasm
//! build does not carry the grammar tables. `rce-spec-data --check` and
//! `tests/spec_data.rs` fail when either file is stale against the book,
//! and `tests/spec_read.rs` checks that every table entry is typed and
//! evaluated.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

pub const SPEC_TABLES: &str = include_str!("../../data/spec-tables.json");

/// The members of spec-data.json the checker reads, as compact JSON.
pub fn tables_of(spec_data: &str) -> String {
    let d: Value = serde_json::from_str(spec_data).expect("spec-data.json is JSON");
    serde_json::json!({ "methods": d["methods"], "method_work": d["method_work"] }).to_string()
}

/// Complexity class of a method (C.6.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Work {
    /// n + k
    Linear,
    /// n · ⌈log₂ n⌉
    NLogN,
    /// 0
    Constant,
}

pub struct Tables {
    /// 5.15: receiver (`Int`, `F`, `Bool`, `Text`, `F32`, `Gen`, `List`,
    /// `Map`, `Opt`) to its methods
    pub methods: BTreeMap<String, BTreeSet<String>>,
    /// C.6.4: (receiver word of the class: `text`, `list`, `map`; method) to
    /// its class; every pair not listed is `Constant`
    pub work: BTreeMap<(String, String), Work>,
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect()).unwrap_or_default()
}

pub static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    let d: Value = serde_json::from_str(SPEC_TABLES).expect("spec-tables.json is JSON");
    let mut methods = BTreeMap::new();
    for r in d["methods"]["table"].as_array().expect("5.15 table") {
        methods.insert(r["receiver"].as_str().expect("receiver").to_string(), strs(&r["methods"]).into_iter().collect());
    }
    let mut work = BTreeMap::new();
    for r in d["method_work"]["table"].as_array().expect("C.6.4 table") {
        let class = r["class"].as_str().expect("class");
        let w = if class.starts_with("linear") {
            Work::Linear
        } else if class.starts_with("n log n") {
            Work::NLogN
        } else {
            continue;
        };
        // "linear, text receiver", "n log n, list receiver", "linear, map receiver (...)"
        let recv = class.split(", ").nth(1).and_then(|s| s.split(' ').next()).expect("receiver word");
        for m in strs(&r["methods"]) {
            work.insert((recv.to_string(), m), w);
        }
    }
    assert!(!methods.is_empty() && !work.is_empty(), "spec-data.json: empty method tables");
    Tables { methods, work }
});

/// Whether 5.15 lists method `m` for receiver `recv`.
pub fn has_method(recv: &str, m: &str) -> bool {
    TABLES.methods.get(recv).is_some_and(|s| s.contains(m))
}

/// The C.6.4 class of method `m` on a receiver of kind `recv` (`text`,
/// `list`, `map`, or anything else).
pub fn work(recv: &str, m: &str) -> Work {
    TABLES.work.get(&(recv.to_string(), m.to_string())).copied().unwrap_or(Work::Constant)
}

/// Every method of 5.15, sorted (language server completion).
pub fn all_methods() -> Vec<String> {
    let s: BTreeSet<&String> = TABLES.methods.values().flatten().collect();
    s.into_iter().cloned().collect()
}
