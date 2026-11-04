#![allow(dead_code, unused_mut, unused_assignments)]
//! RCE reference parser, type and well-formedness checker (RCE-SPEC.md),
//! plus the tree-sitter port of the grammar for cross-checks.
/// The git commit the checker was built from (short hash), set by build.rs;
/// "unknown" outside a git checkout. Named in emitted code (CC-4).
pub const GIT_COMMIT: &str = env!("RCE_GIT_COMMIT");

pub mod adapters;
pub mod admission;
pub mod alter;
pub mod anf;
pub mod apidiff;
pub mod args;
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
pub mod graph;
pub mod wasm_term;
pub mod frame;
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
pub mod lift_authority;
pub mod lift_gguf;
pub mod law;
pub mod link;
pub mod lsp;
pub mod parse;
pub mod raft;
pub mod rap;
pub mod rcedb;
pub mod receipt;
pub mod refine;
pub mod proposals;
pub mod repair;
pub mod repl;
pub mod sandbox;
pub mod specdata;
pub mod mpst;
pub mod mutants;
pub mod mpst_async;
pub mod session_prio;
pub mod specread;
pub mod stdlib;
pub mod status;
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
    check_program_with(files, parse_only, true)
}

/// `check_program`, with the mutation gate (W-MUT-1, 6.12) run on each
/// admitted file when `mutation` is true; `rce mutants` reads the scores of
/// a program the gate would refuse, so it checks without it.
pub fn check_program_with(
    files: &[(String, String)],
    parse_only: bool,
    mutation: bool,
) -> Vec<(String, Result<(), String>)> {
    if parse_only {
        return files.iter().map(|(p, src)| (p.clone(), parse::parse(src).map(|_| ()))).collect();
    }
    // one admission for every tool (admission.rs): imported library items
    // are linked in (4.1, src/link.rs), then the rules in force, their modes
    // (7.1) and the admission gate decide
    admission::program(admission::RuleSet::in_force(), files, mutation)
        .into_iter()
        .map(|(p, a)| {
            let v = match a {
                Err(e) => Err(e),
                Ok(a) if a.errs.is_empty() => Ok(()),
                Ok(a) => Err(a.errs.join("; ")),
            };
            (p, v)
        })
        .collect()
}

/// Admission (6.9, 6.11) of a linked program whose static reports are
/// `errs`, under the rules in force and their modes (7.1), and then, when
/// `mutation` is true and every obligation holds, the mutation gate (6.12,
/// W-MUT-1) on its first `own` items: the reports that require change.
pub fn admit_gated(lk: link::Linked, errs: Vec<String>, own: usize, known: &HashSet<String>, mutation: bool) -> Vec<String> {
    admission::linked(admission::RuleSet::in_force(), lk, errs, own, known, mutation).errs
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

/// One source text parsed, linked and type-checked, without admission:
/// the module `rce law` decides (6.11). The errors are those of `rce check`
/// before its admission gate.
pub fn checked_program(src: &str) -> Result<ast::Module, String> {
    let m = parse::parse(src)?;
    let lk = link::link(&m, &[&m]);
    let known: HashSet<String> = m.path.iter().map(|q| q.join(".")).collect();
    let mut c = check::Checker::new();
    c.imported_fns = lk.imported_fns;
    c.imported_types = lk.imported_types;
    for e in lk.errs {
        c.e(e);
    }
    c.check_module(&lk.module, &known);
    if c.errs.is_empty() { Ok(lk.module) } else { Err(c.errs.join("; ")) }
}

/// Check one source text alone.
pub fn check_one(src: &str) -> Result<(), String> {
    check_program(&[("p.rce".into(), src.into())], false)
        .remove(0)
        .1
}
