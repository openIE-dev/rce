//! Cross-references / call graph derived from the admitted Form (REA cross-refs and JetBrains
//! Find-Usages / call-hierarchy parity), but deterministic and verified: every edge is a real
//! construct of the replay-gated lift, not an indexed heuristic or an agent guess.

use serde_json::{json, Value};

use crate::lut_emit::Stmt;
use rce_spec_check::ast::Expr;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Xref {
    /// (enclosing function or "top", callee)
    pub calls: Vec<(String, String)>,
    pub defs: Vec<String>,
    /// (name, read-count)
    pub uses: Vec<(String, usize)>,
}

fn reads(e: &Expr, uses: &mut Vec<(String, usize)>) {
    match e {
        Expr::Name(n, _) => {
            if let Some(u) = uses.iter_mut().find(|(k, _)| k == n) { u.1 += 1; } else { uses.push((n.clone(), 1)); }
        }
        Expr::Neg(a, _) => reads(a, uses),
        Expr::Bin(_, a, b, _) => { reads(a, uses); reads(b, uses); }
        _ => {}
    }
}

fn walk(s: &Stmt, ctx: &str, x: &mut Xref) {
    match s {
        Stmt::Let(n, e) | Stmt::Assign(n, e) => { x.defs.push(n.clone()); reads(e, &mut x.uses); }
        Stmt::Return(e) => reads(e, &mut x.uses),
        Stmt::If(c, t, e) => { reads(c, &mut x.uses); walk(t, ctx, x); walk(e, ctx, x); }
        Stmt::While(c, b) => { reads(c, &mut x.uses); walk(b, ctx, x); }
        Stmt::For(n, lo, hi, b) => { x.defs.push(n.clone()); reads(lo, &mut x.uses); reads(hi, &mut x.uses); walk(b, ctx, x); }
        Stmt::LetFn(n, _, b) => { x.defs.push(n.clone()); walk(b, n, x); }
        Stmt::Call(n, a) => { x.calls.push((ctx.to_string(), n.clone())); for e in a { reads(e, &mut x.uses); } }
        Stmt::LetCall(n, f, a) => { x.defs.push(n.clone()); x.calls.push((ctx.to_string(), f.clone())); for e in a { reads(e, &mut x.uses); } }
        Stmt::Seq(v) => { for s in v { walk(s, ctx, x); } }
        Stmt::Match(scr, arms) => { reads(scr, &mut x.uses); for (_, b) in arms { walk(b, ctx, x); } }
        _ => {}
    }
}

/// Deterministic cross-references for a program.
pub fn xref(s: &Stmt) -> Xref {
    let mut x = Xref::default();
    walk(s, "top", &mut x);
    x
}

/// Call graph as JSON {nodes, edges} for rendering / MCP exposure.
pub fn call_graph(s: &Stmt) -> Value {
    let x = xref(s);
    let mut nodes: Vec<String> = vec!["top".into()];
    for (a, b) in &x.calls {
        if !nodes.contains(a) { nodes.push(a.clone()); }
        if !nodes.contains(b) { nodes.push(b.clone()); }
    }
    json!({ "nodes": nodes, "edges": x.calls.iter().map(|(a, b)| json!({"from": a, "to": b})).collect::<Vec<_>>() })
}
