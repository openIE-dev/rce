//! The trace of a program (one-emitter.md ONE-5): the bytes the reference
//! interpreter prints for it, which every printed program must reproduce.
//!
//! The trace is the lines that an emitted program prints today under
//! `--trace --fuel-trace`, without the entry of D the emitted program
//! prints before each event (`d applied HASH BODY` or `d kept HASH BODY`,
//! `emit::observe_trace`), which the core does not print yet: one block per observation in item order (each
//! constant, each `check`, each zero-argument function, each example run),
//! then the run of `fn main` when the module has one. An observation starts
//! from a new evaluator, so from full fuel and with no constant evaluated.
//! After each observation comes `fuel N`, the evaluation steps (6.7) it
//! used; after the events of an example comes `trace H`, the head of the
//! example's log chain (6.1).
use crate::ast::*;
use crate::interp::{D_GENESIS, Flow, Interp, Region, V, d_sha, event_pos_json, state_pos_json, why};
use std::collections::HashMap;

/// The reference trace of an admitted module.
pub fn reference(m: &Module) -> String {
    let mut out: Vec<String> = vec![];
    let mut main = None;
    for item in &m.items {
        match item {
            Item::Const { name, .. } => {
                let mut it = Interp::new(m);
                let r = it.const_value(name);
                res(&mut out, format!("const {name}"), r);
                out.push(format!("fuel {}", it.steps_used()));
            }
            Item::Check(e) => check(m, e, &mut out),
            Item::Intent(_, xs, cks) => {
                for e in cks {
                    check(m, e, &mut out);
                }
                for x in xs {
                    example(m, x, &mut out);
                }
            }
            Item::Example(x) => example(m, x, &mut out),
            Item::Fn(f) if f.sig.name == "main" && f.sig.params.is_empty() => main = Some(f),
            Item::Fn(f) if crate::emit::zero_arg(f) => {
                let mut it = Interp::new(m);
                let r = it.call_fn(f, &[], None);
                res(&mut out, format!("fn {}", f.sig.name), r);
                out.push(format!("fuel {}", it.steps_used()));
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
        out.push(format!("fuel {}", it.steps_used()));
    }
    let mut s = out.join("\n");
    if !s.is_empty() {
        s.push('\n');
    }
    s
}

fn res(out: &mut Vec<String>, label: String, r: Result<V, Flow>) {
    match r {
        Ok(v) => out.push(format!("{label} = {}", v.show())),
        Err(f) => out.push(format!("{label} {}", why(f))),
    }
}

fn check(m: &Module, e: &Expr, out: &mut Vec<String>) {
    let mut it = Interp::new(m);
    it.scopes = vec![HashMap::new()];
    let r = it.eval(e);
    res(out, format!("check {}", e.line()), r);
    out.push(format!("fuel {}", it.steps_used()));
}

/// The state as an observation line shows it: `R { f = v, … }`, regions joined by `; `.
pub fn show_state(s: &[Region]) -> String {
    s.iter()
        .map(|(r, fs)| {
            let f: Vec<String> = fs.iter().map(|(n, v, _)| format!("{n} = {}", v.show())).collect();
            format!("{r} {{ {} }}", f.join(", "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The log chain of one example (6.1): each entry hashes the positional
/// event, the previous head, its sequence number and the hash of the
/// positional state after it.
struct DLog {
    prev: String,
    seq: usize,
}

impl DLog {
    fn add(&mut self, m: &Module, name: &str, ev: &V, s: &[Region]) {
        let evp = event_pos_json(name, ev, m);
        let sh = d_sha(&state_pos_json(s, m));
        let body = format!("{{\"event\":{evp},\"prev_hash\":\"{}\",\"seq\":{},\"state_hash\":\"{sh}\"}}", self.prev, self.seq);
        self.prev = d_sha(&body);
        self.seq += 1;
    }
}

fn example(m: &Module, x: &Example, out: &mut Vec<String>) {
    let mut it = Interp::new(m);
    out.push(format!("example {}", x.line));
    match it.initial_state() {
        Err(f) => out.push(format!("initial {}", why(f))),
        Ok(mut s) => {
            out.push(format!("0 (initial) {}", show_state(&s)));
            let mut d = DLog { prev: D_GENESIS.to_string(), seq: 0 };
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
                        d.add(m, evn, &ev, &s);
                        out.push(format!("{k} {} {}", ev.show(), show_state(&s)));
                        for o in outs {
                            out.push(format!("  emit {}", o.show()));
                        }
                    }
                    Err(r) => {
                        d.add(m, evn, &ev, &s);
                        out.push(format!("{k} {} refused: {r}", ev.show()));
                        break;
                    }
                }
            }
            out.push(format!("trace {}", d.prev));
        }
    }
    out.push(format!("fuel {}", it.steps_used()));
}
