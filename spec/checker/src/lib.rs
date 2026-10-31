#![allow(dead_code, unused_mut, unused_assignments)]
//! RCE reference parser, type and well-formedness checker (RCE-SPEC.md),
//! plus the tree-sitter port of the grammar for cross-checks.
/// The git commit the checker was built from (short hash), set by build.rs;
/// "unknown" outside a git checkout. Named in emitted code (CC-4).
pub const GIT_COMMIT: &str = env!("RCE_GIT_COMMIT");

pub mod adapters;
pub mod alter;
pub mod anf;
pub mod apidiff;
pub mod borrow;
pub mod ast;
pub mod book;
pub mod brick;
pub mod brick_lower;
pub mod brick_vec;
pub mod check;
pub mod check06b;
pub mod cli;
pub mod conc;
pub mod core_elab;
pub mod core_rules;
pub mod dim;
pub mod nameless;
pub mod one;
pub mod form;
pub mod emit;
pub mod host;
pub mod ifc;
pub mod tier2;
pub mod tier3;
pub mod caretaker;
pub mod manifest;
pub mod index08;
pub mod interp;
pub mod lalr;
pub mod link;
pub mod lsp;
pub mod parse;
pub mod raft;
pub mod rcedb;
pub mod refine;
pub mod repl;
pub mod sandbox;
pub mod specdata;
pub mod mpst;
pub mod mpst_async;
pub mod session_prio;
pub mod specread;
pub mod stdlib;
pub mod tls;
#[cfg(feature = "tree-sitter")]
pub mod ts;
pub mod typed;
pub mod types;
pub mod unparse;
pub mod v09;
pub mod wasm_float;
use std::collections::HashSet;

/// Check `files` (name, source) as one program, as `rce-check` does: every
/// file is parsed, `use` resolves among them, then each is type checked and
/// admitted. Returns one verdict per file, in order: Ok or what requires change.
pub fn check_program(
    files: &[(String, String)],
    parse_only: bool,
) -> Vec<(String, Result<(), String>)> {
    let mut out = vec![];
    let mut parsed = vec![];
    for (p, src) in files {
        match parse::parse(src) {
            Ok(m) => parsed.push((p.clone(), m)),
            Err(e) => out.push((p.clone(), Err(e))),
        }
    }
    let known: HashSet<String> = parsed
        .iter()
        .filter_map(|(_, m)| m.path.as_ref().map(|q| q.join(".")))
        .collect();
    let mods: Vec<&ast::Module> = parsed.iter().map(|(_, m)| m).collect();
    for (p, m) in &parsed {
        if parse_only {
            out.push((p.clone(), Ok(())));
            continue;
        }
        // imported library items are linked in (4.1, src/link.rs)
        let lk = link::link(m, &mods);
        let mut c = check::Checker::new();
        c.imported_fns = lk.imported_fns;
        c.imported_types = lk.imported_types;
        for e in lk.errs {
            c.e(e);
        }
        c.check_module(&lk.module, &known);
        if c.errs.is_empty() {
            c.errs = interp::admit(&lk.module);
        }
        out.push((
            p.clone(),
            if c.errs.is_empty() {
                Ok(())
            } else {
                Err(c.errs.join("; "))
            },
        ));
    }
    // Same order as the input.
    let pos = |n: &String| files.iter().position(|(p, _)| p == n).unwrap_or(usize::MAX);
    out.sort_by_key(|(n, _)| pos(n));
    out
}

/// One source text as the program the tools run: parsed, and its `use`
/// imports linked in from the built-in standard library (4.1). This is what
/// `rce run`, `test`, `trace`, `emit`, `dim` and `host` interpret;
/// `check_program` checks the same text.
pub fn program(src: &str) -> Result<ast::Module, String> {
    let m = parse::parse(src)?;
    let lk = link::link(&m, &[]);
    if let Some(e) = lk.errs.iter().find(|e| e.contains("W-MOD-2")) {
        return Err(e.clone());
    }
    Ok(lk.module)
}

/// Check one source text alone.
pub fn check_one(src: &str) -> Result<(), String> {
    check_program(&[("p.rce".into(), src.into())], false)
        .remove(0)
        .1
}
