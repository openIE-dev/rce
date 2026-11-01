//! Emitters for `rce emit rust|python|js|scala` (spec 10.5); `verilog`, the
//! step function as a hardware module, is in `verilog.rs`.
//!
//! Each emitter turns a checked module into one self-contained source file.
//! The file starts with a runtime that follows the reference evaluator
//! (src/interp.rs) value by value. Then comes one target function for each
//! RCE function, method, step and constant. Expressions become calls to
//! runtime helpers, with operands in evaluation order. Statements become the
//! target language's own control flow.
//!
//! The Rust output has the same shape as the verified emitter of T9
//! (spec/lean/RCE/Emit.lean): one uniform value type, `Result` with `?` for
//! traps and kept steps, checked i128 arithmetic against each integer type's
//! range, and one counter per bounded loop.
//!
//! Running an emitted file prints the program's observations (`observe`).
//! These cover every constant, every `check`, every zero-argument function,
//! every example run and `fn main`. The emitter tests compare those lines
//! with the reference evaluator's.
use crate::ast::*;
use crate::interp::{Interp, V, why};
use std::collections::HashMap;

mod base;
mod conc06;
mod expr;
mod probe;
mod prog;
mod stmt;
mod tasks;
pub mod verilog;
pub use probe::with_probes;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Lang {
    Rust,
    Python,
    Js,
    /// Scala 3 on the JVM (the functional-language target; see rt.scala)
    Scala,
}

impl Lang {
    pub fn parse(s: &str) -> Option<Lang> {
        match s {
            "rust" => Some(Lang::Rust),
            "python" => Some(Lang::Python),
            "js" => Some(Lang::Js),
            "scala" => Some(Lang::Scala),
            _ => None,
        }
    }
    pub fn ext(self) -> &'static str {
        match self {
            Lang::Rust => "rs",
            Lang::Python => "py",
            Lang::Js => "mjs",
            Lang::Scala => "scala",

        }
    }
}

/// Emit `m` (already admitted by the checker) as one source file. An `Err`
/// names a construct this emitter does not translate.
pub fn emit(m: &Module, lang: Lang) -> Result<String, String> {
    if let Some(e) = crate::v09::unemittable(m) {
        return Err(e);
    }
    // CC-4: the first line names the program (its hash, name-free) and the
    // checker that printed it, so an emitted file is tied to its source.
    let provenance = format!(
        "{} rce-program {} · emitted by rce commit {}",
        match lang {
            Lang::Python => "#",
            _ => "//",
        },
        crate::apidiff::program_hash(std::slice::from_ref(m)),
        crate::GIT_COMMIT,
    );
    // one shared, order-preserving lowering for all three emitters (6.2)
    // a `let &mut` borrow names its place (5.6), the same rewrite the reference evaluator makes
    let m = crate::borrow::lower_module(m);
    let m = crate::anf::lower(&m);
    let body = base::G::new(&m, lang).program()?;
    Ok(format!("{provenance}\n{body}"))
}

fn show_state(s: &[crate::interp::Region]) -> String {
    s.iter()
        .map(|(r, fs)| {
            let f: Vec<String> = fs
                .iter()
                .map(|(n, v, _)| format!("{n} = {}", v.show()))
                .collect();
            format!("{r} {{ {} }}", f.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

pub(crate) fn zero_arg(f: &FnDecl) -> bool {
    f.sig.params.is_empty() && f.sig.generics.is_empty() && f.sig.name != "main"
}

fn res(out: &mut Vec<String>, label: String, r: Result<V, crate::interp::Flow>) {
    match r {
        Ok(v) => out.push(format!("{label} = {}", v.show())),
        Err(f) => out.push(format!("{label} {}", why(f))),
    }
}

/// What an admitted module does, according to the reference evaluator. There
/// is one line per constant, `check`, zero-argument function, example event
/// and `fn main` output, in item order, with `main` last.
pub fn observe(m: &Module) -> Vec<String> {
    observe_with(m, false, false)
}

/// `observe`, with a line `fuel N` after each observation giving the steps
/// (fuel ticks, 6.7) it used; an emitted program run with `fuel-trace`
/// prints the same lines (tests/fuel.rs).
pub fn observe_fuel(m: &Module) -> Vec<String> {
    observe_with(m, true, false)
}

/// `observe_fuel`, with the log D (6.1) of each example: the lines an
/// emitted program prints when run with `trace fuel-trace`. Before the
/// line of each event δ applied or kept comes its entry of D,
/// `d applied HASH BODY` or `d kept HASH BODY`, where BODY is the canonical
/// JSON of `{event, prev_hash, seq, state_hash}` over the positional forms
/// and HASH is its SHA-256; after the example's last event comes
/// `trace HEAD`, the hash of the last entry (64 zeros when there is none).
/// Each line of an example block is part of its whole trace: the entries,
/// the state after each event, the outputs (λ), the reason an event was
/// kept, and the fuel the example used. The head alone commits to the
/// events and state hashes only.
pub fn observe_trace(m: &Module) -> Vec<String> {
    observe_with(m, true, true)
}

/// The whole trace of example `x` of `m`: its block of `observe_trace`,
/// without the `example LINE` header (whose number depends only on where
/// the example is written).
pub fn trace_example(m: &Module, x: &Example) -> Vec<String> {
    let mut out = vec![];
    obs_example(m, x, &mut out, true, true);
    out.remove(0);
    out
}

fn observe_with(m: &Module, fuel: bool, trace: bool) -> Vec<String> {
    let mut out = vec![];
    let mut main = None;
    for item in &m.items {
        match item {
            Item::Const { name, .. } => {
                let mut it = Interp::new(m);
                let r = it.const_value(name);
                res(&mut out, format!("const {name}"), r);
                if fuel { out.push(format!("fuel {}", it.steps_used())); }
            }
            Item::Check(e) => obs_check(m, e, &mut out, fuel),
            Item::Intent(_, xs, cks) => {
                for e in cks {
                    obs_check(m, e, &mut out, fuel);
                }
                for x in xs {
                    obs_example(m, x, &mut out, fuel, trace);
                }
            }
            Item::Example(x) => obs_example(m, x, &mut out, fuel, trace),
            Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty() => main = Some(f),
            Item::Fn(f) if zero_arg(f) => {
                let mut it = Interp::new(m);
                let r = it.call_fn(f, &[], None);
                res(&mut out, format!("fn {}", f.sig.name), r);
                if fuel { out.push(format!("fuel {}", it.steps_used())); }
            }
            _ => {}
        }
    }
    if let Some(f) = main {
        out.push("main".into());
        let mut it = Interp::new(m);
        it.outputs.clear();
        let r = it.call_fn(f, &[], None);
        for o in std::mem::take(&mut it.outputs) {
            out.push(format!("emit {}", o.show()));
        }
        res(&mut out, "main".into(), r);
        if fuel { out.push(format!("fuel {}", it.steps_used())); }
    }
    out
}

fn obs_check(m: &Module, e: &Expr, out: &mut Vec<String>, fuel: bool) {
    let mut it = Interp::new(m);
    it.scopes = vec![HashMap::new()];
    let r = it.eval(e);
    res(out, format!("check {}", e.line()), r);
    if fuel { out.push(format!("fuel {}", it.steps_used())); }
}

fn obs_example(m: &Module, x: &Example, out: &mut Vec<String>, fuel: bool, trace: bool) {
    let mut it = Interp::new(m);
    obs_example1(m, x, out, &mut it, trace);
    if fuel { out.push(format!("fuel {}", it.steps_used())); }
}

/// The log D of one example as it is written (6.1), for `observe_trace`.
struct DLines {
    prev: String,
    seq: usize,
}

impl DLines {
    /// Append the entry of `ev` (named `evn` in the example) with the state
    /// `s` after it, and print it.
    fn add(&mut self, m: &Module, evn: &str, ev: &V, s: &[crate::interp::Region], kept: bool, out: &mut Vec<String>) {
        let evp = crate::interp::event_pos_json(evn, ev, m);
        let sh = crate::interp::d_sha(&crate::interp::state_pos_json(s, m));
        let body = format!("{{\"event\":{evp},\"prev_hash\":\"{}\",\"seq\":{},\"state_hash\":\"{sh}\"}}", self.prev, self.seq);
        let h = crate::interp::d_sha(&body);
        out.push(format!("d {} {h} {body}", if kept { "kept" } else { "applied" }));
        self.prev = h;
        self.seq += 1;
    }
}

fn obs_example1<'a>(m: &'a Module, x: &'a Example, out: &mut Vec<String>, it: &mut Interp<'a>, trace: bool) {
    out.push(format!("example {}", x.line));
    let mut s = match it.initial_state() {
        Ok(s) => s,
        Err(f) => {
            out.push(format!("initial {}", why(f)));
            return;
        }
    };
    out.push(format!("0 (initial) {}", show_state(&s)));
    let mut d = DLines { prev: crate::interp::D_GENESIS.to_string(), seq: 0 };
    for (k, (evn, args)) in x.events.iter().enumerate() {
        let k = k + 1;
        it.scopes = vec![HashMap::new()];
        let ev = match it.construct(evn, args) {
            Ok(Some(v)) => v,
            Ok(None) => {
                out.push(format!("{k} {evn} not an event"));
                break;
            }
            Err(f) => {
                out.push(format!("{k} {evn} {}", why(f)));
                break;
            }
        };
        match it.delta(&s, &ev) {
            Ok((s2, outs)) => {
                s = s2;
                if trace {
                    d.add(m, evn, &ev, &s, false, out);
                }
                out.push(format!("{k} {} {}", ev.show(), show_state(&s)));
                for o in outs {
                    out.push(format!("  emit {}", o.show()));
                }
            }
            Err(r) => {
                if trace {
                    d.add(m, evn, &ev, &s, true, out);
                }
                // a kept step keeps the state (T8); the next event is applied to it
                out.push(format!("{k} {} kept: {r}", ev.show()));
            }
        }
    }
    if trace {
        out.push(format!("trace {}", d.prev));
    }
}
