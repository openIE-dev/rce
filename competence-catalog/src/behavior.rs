//! Behavior model (Stately/XState statechart parity): a deterministic statechart extracted from the
//! admitted Form — states = program points, labeled transitions = branches/loops/sequence. Humans
//! read it; agents reason about it; it is a projection of the replay-gated lift, not a guess.
//! Unlike Stately, the model is content-addressed and joule-priced via the underlying Form.

use serde_json::{json, Value};

use crate::lut_emit::Stmt;

/// Extract a statechart {states, transitions, initial, final} from a program.
pub fn statechart(s: &Stmt) -> Value {
    let mut states: Vec<String> = vec!["init".into()];
    let mut tr: Vec<Value> = Vec::new();
    let mut n = 0usize;
    fn fresh(states: &mut Vec<String>, n: &mut usize) -> String {
        let id = format!("s{n}");
        *n += 1;
        states.push(id.clone());
        id
    }
    fn edge(tr: &mut Vec<Value>, from: &str, to: &str, event: &str) {
        tr.push(json!({ "from": from, "to": to, "event": event }));
    }
    fn walk(s: &Stmt, cur: &str, states: &mut Vec<String>, tr: &mut Vec<Value>, n: &mut usize) -> String {
        match s {
            Stmt::Seq(v) => {
                let mut c = cur.to_string();
                for x in v { c = walk(x, &c, states, tr, n); }
                c
            }
            Stmt::If(_, t, e) => {
                let a = fresh(states, n);
                let b = fresh(states, n);
                let join = fresh(states, n);
                edge(tr, cur, &a, "true");
                edge(tr, cur, &b, "false");
                let ta = walk(t, &a, states, tr, n);
                let tb = walk(e, &b, states, tr, n);
                edge(tr, &ta, &join, "join");
                edge(tr, &tb, &join, "join");
                join
            }
            Stmt::While(_, b) => {
                let body = fresh(states, n);
                let exit = fresh(states, n);
                edge(tr, cur, &body, "loop");
                edge(tr, cur, &exit, "exit");
                let end = walk(b, &body, states, tr, n);
                edge(tr, &end, cur, "repeat");
                exit
            }
            Stmt::Return(_) => {
                edge(tr, cur, "final", "return");
                cur.to_string()
            }
            other => {
                let next = fresh(states, n);
                let ev = match other {
                    Stmt::Let(..) => "bind",
                    Stmt::Assign(..) => "update",
                    Stmt::Call(..) | Stmt::LetCall(..) => "call",
                    Stmt::LetFn(..) => "define",
                    _ => "step",
                };
                edge(tr, cur, &next, ev);
                next
            }
        }
    }
    states.push("final".into());
    let end = walk(s, "init", &mut states, &mut tr, &mut n);
    if end != "final" { edge(&mut tr, &end, "final", "done"); }
    json!({ "initial": "init", "final": "final", "states": states, "transitions": tr })
}
