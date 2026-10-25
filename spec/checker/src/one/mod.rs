//! The one emitter (spec/proposals/one-emitter.md): every program is lowered
//! once to the core `rce-core/1`, and every language prints that core.
//!
//! - `ir`: the data types of the core (section 3.2)
//! - `json`: its JSON form and the keys of its groups (3.4, W5)
//! - `wf`: its well-formedness check, W1 to W6 (3.3)
//! - `eval`: a reference evaluator of the core
//! - `trace`: the reference trace a printed program must reproduce (3.6)
//! - `build`, `rt`: building core functions, and the runtime functions a
//!   lowering calls, written in the core
pub mod build;
pub mod eval;
pub mod ir;
pub mod json;
pub mod rt;
pub mod trace;
pub mod wf;

/// Evaluate a core module and return what it writes.
pub fn run_core(c: &ir::Module) -> Result<String, String> {
    let r = eval::run(c)?;
    if r.halt.is_some() {
        return Err("the core program halted".into());
    }
    String::from_utf8(r.out).map_err(|_| "the core program wrote bytes that are not UTF-8".into())
}
