//! The one emitter (spec/proposals/one-emitter.md): every program is lowered
//! once to the core `rce-core/1`, and every language prints that core.
//!
//! - `ir`: the data types of the core (section 3.2)
//! - `json`: its JSON form and the keys of its groups (3.4, W5)
//! - `wf`: its well-formedness check, W1 to W6 (3.3)
//! - `eval`: a reference evaluator of the core
//! - `trace`: the reference trace a printed program must reproduce (3.6)
//! - `build`, `rt`: building core functions, and the runtime functions the
//!   lowering calls, written in the core
//! - `lower`: the lowering of an RCE module to the core (3.5)
pub mod build;
pub mod eval;
pub mod ir;
pub mod json;
pub mod lower;
pub mod rt;
pub mod trace;
pub mod wf;

/// `rce emit core`: the core of a module, as canonical JSON.
pub fn core_of(m: &crate::ast::Module) -> Result<ir::Module, String> {
    let c = lower::lower(m)?;
    if let Err(e) = wf::check(&c) {
        if let Some(p) = std::env::var_os("RCE_CORE_DUMP") {
            let _ = std::fs::write(p, json::text(&c));
        }
        return Err(format!("internal: the lowered core is not well formed: {e}"));
    }
    Ok(c)
}

/// `rce emit trace`: the trace printed by evaluating the core of a module.
pub fn trace_of(m: &crate::ast::Module) -> Result<String, String> {
    run_core(&core_of(m)?)
}

/// Evaluate a core module and return what it writes.
pub fn run_core(c: &ir::Module) -> Result<String, String> {
    let r = eval::run(c)?;
    if r.halt.is_some() {
        return Err("the core program halted".into());
    }
    String::from_utf8(r.out).map_err(|_| "the core program wrote bytes that are not UTF-8".into())
}
