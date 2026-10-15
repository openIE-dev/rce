#![allow(dead_code, unused_mut, unused_assignments)]
//! RCE reference parser, type and well-formedness checker (RCE-SPEC.md),
//! plus the tree-sitter port of the grammar for cross-checks.
pub mod adapters;
pub mod anf;
pub mod ast;
pub mod book;
pub mod brick;
pub mod brick_lower;
pub mod brick_vec;
pub mod check;
pub mod check06b;
pub mod cli;
pub mod conc;
pub mod emit;
pub mod host;
pub mod index08;
pub mod interp;
pub mod lalr;
pub mod link;
pub mod lsp;
pub mod parse;
pub mod rcedb;
pub mod refine;
pub mod sandbox;
pub mod specdata;
pub mod specread;
pub mod tls;
#[cfg(feature = "tree-sitter")]
pub mod ts;
pub mod types;
pub mod unparse;
pub mod wasm_float;
use std::collections::HashSet;

/// Check `files` (name, source) as one program, as `rce-check` does: every
/// file is parsed, `use` resolves among them, then each is type checked and
/// admitted. Returns one verdict per file, in order: Ok or the refusal.
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

/// Check one source text alone.
pub fn check_one(src: &str) -> Result<(), String> {
    check_program(&[("p.rce".into(), src.into())], false)
        .remove(0)
        .1
}
