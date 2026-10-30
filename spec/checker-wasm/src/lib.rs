#![allow(dead_code, unused_mut, unused_assignments)]
//! Browser surface for rce-check. One call: `check(src)` returns a JSON string.
#[path = "../../checker/src/ast.rs"]
mod ast;
#[path = "../../checker/src/borrow.rs"]
mod borrow;
#[path = "../../checker/src/check.rs"]
mod check;
#[path = "../../checker/src/check06b.rs"]
mod check06b;
#[path = "../../checker/src/conc.rs"]
mod conc;
#[path = "../../checker/src/interp.rs"]
mod interp;
#[path = "../../checker/src/link.rs"]
mod link;
#[path = "../../checker/src/parse.rs"]
mod parse;
#[path = "../../checker/src/index08.rs"]
pub mod index08;
#[path = "../../checker/src/refine.rs"]
pub mod refine;
#[path = "../../checker/src/types.rs"]
mod types;
#[path = "../../checker/src/specread.rs"]
mod specread;
#[path = "../../checker/src/session_prio.rs"]
mod session_prio;
#[path = "../../checker/src/v09.rs"]
pub mod v09;
#[path = "../../checker/src/ifc.rs"]
pub mod ifc;
#[path = "../../checker/src/tier2.rs"]
pub mod tier2;
#[path = "../../checker/src/tier3.rs"]
pub mod tier3;
#[path = "../../checker/src/apidiff.rs"]
pub mod apidiff;
#[path = "../../checker/src/unparse.rs"]
pub mod unparse;
#[path = "../../checker/src/nameless.rs"]
pub mod nameless;
#[path = "../../checker/src/stdlib.rs"]
pub mod stdlib;
use std::collections::HashSet;
use wasm_bindgen::prelude::*;

fn js(s: &str) -> String {
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

fn rule_of(e: &str) -> String {
    // first token that looks like a rule id, e.g. W-INT-1, T-MUT-2, R3
    for w in e.split(|c: char| !(c.is_ascii_alphanumeric() || c == '-')) {
        let up = w.chars().next().is_some_and(|c| c.is_ascii_uppercase());
        if up && w.contains('-') && w.chars().any(|c| c.is_ascii_digit()) {
            return w.to_string();
        }
    }
    String::new()
}

/// Parse, type-check and admit one module. Returns JSON:
/// {"stage":"parse"|"check"|"admit"|"ok","ok":bool,"errors":[{"msg","rule"}],"examples":[...]}
#[wasm_bindgen]
pub fn check(src: &str) -> String {
    check_with(src, "")
}

/// Same as `check`, with `known` a comma-separated list of other module paths in the program (for `use`).
#[wasm_bindgen]
pub fn check_with(src: &str, known_mods: &str) -> String {
    let m = match parse::parse(src) {
        Ok(m) => m,
        Err(e) => {
            return format!(
                "{{\"ok\":false,\"stage\":\"parse\",\"errors\":[{{\"msg\":{},\"rule\":{}}}],\"examples\":[]}}",
                js(&e),
                js(&rule_of(&e))
            );
        }
    };
    let mut known: HashSet<String> = m.path.iter().map(|q| q.join(".")).collect();
    known.extend(known_list(known_mods));
    // the program as `rce` prepares it: `use` imports linked in from the
    // built-in standard library (4.1)
    let lk = link::link(&m, &[]);
    let m = lk.module;
    let mut c = check::Checker::new();
    c.imported_fns = lk.imported_fns;
    c.imported_types = lk.imported_types;
    for e in lk.errs {
        c.e(e);
    }
    c.check_module(&m, &known);
    let mut stage = "check";
    if c.errs.is_empty() {
        stage = "admit";
        c.errs = interp::admit(&m);
    }
    let ok = c.errs.is_empty();
    let errs: Vec<String> = c
        .errs
        .iter()
        .map(|e| format!("{{\"msg\":{},\"rule\":{}}}", js(e), js(&rule_of(e))))
        .collect();
    let mut exs = vec![];
    if stage == "admit" {
        for (line, steps) in interp::trace_examples(&m) {
            let ss: Vec<String> = steps
                .iter()
                .map(|s| match &s.result {
                    Ok(st) => {
                        let regions: Vec<String> = st
                            .iter()
                            .map(|(r, fs)| {
                                let f: Vec<String> = fs
                                    .iter()
                                    .map(|(n, v)| format!("{}:{}", js(n), js(v)))
                                    .collect();
                                format!("{}:{{{}}}", js(r), f.join(","))
                            })
                            .collect();
                        format!(
                            "{{\"event\":{},\"state\":{{{}}}}}",
                            js(&s.event),
                            regions.join(",")
                        )
                    }
                    Err(r) => format!("{{\"event\":{},\"kept\":{}}}", js(&s.event), js(r)),
                })
                .collect();
            exs.push(format!("{{\"line\":{line},\"steps\":[{}]}}", ss.join(",")));
        }
    }
    let form = if ok { interp::form_json(&m) } else { "null".into() };
    format!(
        "{{\"ok\":{ok},\"stage\":{},\"errors\":[{}],\"examples\":[{}],\"form\":{form}}}",
        js(if ok { "ok" } else { stage }),
        errs.join(","),
        exs.join(",")
    )
}

fn known_list(k: &str) -> Vec<String> {
    k.split(',').map(str::trim).filter(|s| !s.is_empty()).map(String::from).collect()
}
