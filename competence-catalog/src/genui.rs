//! Constrained declarative generative UI (A2UI / json-render / OpenUI parity), grounded in the
//! admitted construct IR. The model/agent may only emit components from a fixed allow-listed
//! catalog (the construct kinds); the UI tree is a deterministic projection of the replay-gated
//! lift, so every rendered node is admitted competence, not next-token-generated raw markup.
//! Renderers (rcebrowser / hosts via MCP Apps / AG-UI) consume this JSON; they never execute model
//! code. This is the "generative UI" mechanism with the gate underneath.

use serde_json::{json, Value};

use crate::lut_emit::Stmt;
use crate::program::lift_program;

/// The allow-listed component catalog (construct kinds). A renderer refuses anything outside it.
pub fn catalog() -> Vec<&'static str> {
    vec!["program", "sequence", "bind", "update", "result", "branch", "loop", "call", "define", "value"]
}

fn node(component: &str, label: &str, children: Vec<Value>) -> Value {
    json!({ "component": component, "props": { "label": label }, "children": children })
}

fn expr_value(e: &rce_spec_check::ast::Expr) -> Value {
    node("value", &crate::canonical::emit(e), vec![])
}

fn walk(s: &Stmt) -> Value {
    match s {
        Stmt::Let(n, e) => node("bind", n, vec![expr_value(e)]),
        Stmt::Assign(n, e) => node("update", n, vec![expr_value(e)]),
        Stmt::Return(e) => node("result", "return", vec![expr_value(e)]),
        Stmt::If(c, t, e) => node("branch", "if", vec![expr_value(c), walk(t), walk(e)]),
        Stmt::While(c, b) => node("loop", "while", vec![expr_value(c), walk(b)]),
        Stmt::For(n, lo, hi, b) => node("loop", n, vec![expr_value(lo), expr_value(hi), walk(b)]),
        Stmt::LetFn(n, _, b) => node("define", n, vec![walk(b)]),
        Stmt::Call(n, _) => node("call", n, vec![]),
        Stmt::LetCall(n, f, _) => node("call", &format!("{n} = {f}"), vec![]),
        Stmt::Seq(v) => node("sequence", "seq", v.iter().map(walk).collect()),
        _ => node("value", "stmt", vec![]),
    }
}

/// Emit the constrained declarative UI for a program: {catalog, tree}. Every tree component is in
/// the catalog; the tree is a pure function of the lift (deterministic, admitted).
pub fn ui_spec(src: &str) -> Result<String, String> {
    let stmt = lift_program(src)?;
    let tree = match &stmt {
        Stmt::Seq(_) => walk(&stmt),
        other => node("program", "program", vec![walk(other)]),
    };
    Ok(json!({ "catalog": catalog(), "tree": tree }).to_string())
}

/// Validate that a UI tree only uses allow-listed components (the renderer's gate).
pub fn validate(spec: &str) -> Result<(), String> {
    let v: Value = serde_json::from_str(spec).map_err(|e| e.to_string())?;
    let cat = catalog();
    fn check(n: &Value, cat: &[&str]) -> Result<(), String> {
        let c = n["component"].as_str().ok_or("missing component")?;
        if !cat.contains(&c) { return Err(format!("component not in catalog: {c}")); }
        if let Some(ch) = n["children"].as_array() {
            for x in ch { check(x, cat)?; }
        }
        Ok(())
    }
    check(&v["tree"], &cat)
}
