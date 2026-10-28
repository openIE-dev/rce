//! The catalogue of `rce-form/1` (rce-form.md §4), read from
//! `spec/data/form-catalogue.json` (`rce-catalogue/1`), which the checker
//! carries. Its key is the SHA-256 of the file's bytes, which are canonical
//! JSON (`tests/form_catalogue.rs` checks that they are).
use serde_json::Value as J;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The catalogue file as the checker carries it.
pub const TEXT: &str = include_str!("../../../data/form-catalogue.json");

/// The mode of an operand port (§8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Consume,
    ReadReturn,
    UpdateReturn,
    Order,
}

/// One entry of the catalogue.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: String,
    pub cell: bool,
    pub effect: u8,
    /// the kind of each type argument: body, frame, int, int-bitwise, int-zero, sum or tup
    pub params: Vec<String>,
    /// each operand's mode and type pattern
    pub operands: Vec<(Mode, J)>,
    /// each result's returned operand (if any) and type pattern
    pub results: Vec<(Option<usize>, J)>,
    pub symmetric: Vec<usize>,
    /// the domain fact and its tests, for a partial entry
    pub domain: Option<(String, Vec<String>)>,
    pub traps: Vec<String>,
}

struct Cat {
    key: String,
    entries: BTreeMap<String, Entry>,
}

fn load() -> Cat {
    let v: J = serde_json::from_str(TEXT).expect("the catalogue the checker carries is JSON");
    let mut entries = BTreeMap::new();
    for e in v["entries"].as_array().expect("catalogue entries") {
        let s = |k: &str| e[k].as_str().unwrap_or_default().to_string();
        let mode = |m: &str| match m {
            "consume" => Mode::Consume,
            "read-return" => Mode::ReadReturn,
            "update-return" => Mode::UpdateReturn,
            _ => Mode::Order,
        };
        let id = s("id");
        let ent = Entry {
            id: id.clone(),
            cell: e["cell"].as_bool().unwrap_or(false),
            effect: e["effect"].as_u64().unwrap_or(0) as u8,
            params: e["params"].as_array().into_iter().flatten().map(|p| p.as_str().unwrap_or_default().to_string()).collect(),
            operands: e["operands"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|o| (mode(o["mode"].as_str().unwrap_or_default()), o["type"].clone()))
                .collect(),
            results: e["results"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|r| (r["of"].as_u64().map(|x| x as usize), r["type"].clone()))
                .collect(),
            symmetric: e["symmetric"].as_array().into_iter().flatten().filter_map(|x| x.as_u64()).map(|x| x as usize).collect(),
            domain: e["domain"].as_object().map(|d| {
                (
                    d["fact"].as_str().unwrap_or_default().to_string(),
                    d["tests"].as_array().into_iter().flatten().map(|t| t.as_str().unwrap_or_default().to_string()).collect(),
                )
            }),
            traps: e["traps"].as_array().into_iter().flatten().map(|t| t.as_str().unwrap_or_default().to_string()).collect(),
        };
        entries.insert(id, ent);
    }
    Cat { key: crate::interp::d_sha(TEXT), entries }
}

fn cat() -> &'static Cat {
    static C: OnceLock<Cat> = OnceLock::new();
    C.get_or_init(load)
}

/// The key a module names to say it was written against this catalogue.
pub fn key() -> &'static str {
    &cat().key
}

/// The entry of a name, if the catalogue has it.
pub fn entry(id: &str) -> Option<&'static Entry> {
    cat().entries.get(id)
}
