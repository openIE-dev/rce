//! Full-kernel seed — Stmt-level lowering to ARM64 with a real stack frame. Params and lets are
//! spilled to fixed slots (no register-aliasing class of bugs); `for` desugars to let+while+assign;
//! control flow uses the same label/branch machinery as the expression compiler. The expression
//! fragment (arithmetic, comparisons, logicals, conditional select, capability-derived FMA) is
//! reused verbatim via the shared codegen. x86-64 kernel lowering and fn/calls are the next breadth.

use std::collections::BTreeMap;

use rce_spec_check::ast::Expr;

use crate::compiler::{Arch, Cg, Target};
use crate::lut_emit::Stmt;
use crate::program::lift_program;

/// Collect stack slots: params a/b/c occupy 0..2; every let/for binding gets the next slot.
fn collect(s: &Stmt, slots: &mut BTreeMap<String, u32>) {
    match s {
        Stmt::Let(n, _) | Stmt::For(n, ..) => {
            if !slots.contains_key(n) {
                let next = slots.len() as u32;
                slots.insert(n.clone(), next);
            }
        }
        Stmt::If(_, t, e) => { collect(t, slots); collect(e, slots); }
        Stmt::While(_, b) => collect(b, slots),
        Stmt::Seq(v) => { for x in v { collect(x, slots); } }
        _ => {}
    }
}

fn slot_of(slots: &BTreeMap<String, u32>, n: &str) -> Result<u32, String> {
    slots.get(n).copied().ok_or_else(|| format!("unbound name {n}"))
}

fn lower(s: &Stmt, cg: &mut Cg, slots: &BTreeMap<String, u32>, frame: u16) -> Result<(), String> {
    match s {
        Stmt::Let(n, e) | Stmt::Assign(n, e) => {
            let slot = slot_of(slots, n)?;
            cg.gen(e, 0)?;
            cg.emit(&format!("stur w0, [sp, #{}]", slot * 8));
        }
        Stmt::Return(e) => {
            cg.gen(e, 0)?;
            cg.emit(&format!("add sp, sp, #{frame}"));
            cg.emit("ret");
        }
        Stmt::If(c, th, el) => {
            let cd = cg.alloc();
            cg.gen(c, cd)?;
            let z = cg.gen_test(cd)?;
            let lelse = cg.new_label();
            let lend = cg.new_label();
            cg.emit(&format!("b.eq {lelse}"));
            cg.release(z);
            cg.release(cd);
            lower(th, cg, slots, frame)?;
            cg.emit(&format!("b {lend}"));
            cg.label(&lelse);
            lower(el, cg, slots, frame)?;
            cg.label(&lend);
        }
        Stmt::While(c, body) => {
            let ltop = cg.new_label();
            let lend = cg.new_label();
            cg.label(&ltop);
            let cd = cg.alloc();
            cg.gen(c, cd)?;
            let z = cg.gen_test(cd)?;
            cg.emit(&format!("b.eq {lend}"));
            cg.release(z);
            cg.release(cd);
            lower(body, cg, slots, frame)?;
            cg.emit(&format!("b {ltop}"));
            cg.label(&lend);
        }
        Stmt::For(n, lo, hi, body) => {
            // for n in lo..hi { body }  ==  let n = lo; while n < hi { body; n = n + 1; }
            lower(&Stmt::Let(n.clone(), lo.clone()), cg, slots, frame)?;
            let cond = Expr::Bin("<".into(), Box::new(Expr::Name(n.clone(), 0)), Box::new(hi.clone()), 0);
            let inc = Stmt::Assign(
                n.clone(),
                Expr::Bin("+".into(), Box::new(Expr::Name(n.clone(), 0)), Box::new(Expr::Int(1, 0)), 0),
            );
            let mut seq = match body.as_ref() { Stmt::Seq(v) => v.clone(), other => vec![other.clone()] };
            seq.push(inc);
            lower(&Stmt::While(cond, Box::new(Stmt::Seq(seq))), cg, slots, frame)?;
        }
        Stmt::Seq(v) => { for x in v { lower(x, cg, slots, frame)?; } }
        Stmt::Import(_) | Stmt::TypeDef(_) | Stmt::Attribute(_) | Stmt::GenericType(..) => {}
        _ => return Err("construct outside the kernel seed (fn/calls/match/async next)".into()),
    }
    Ok(())
}

/// Lower a whole program (Stmt) to ARM64 assembly with a stack frame.
pub fn compile_stmt(s: &Stmt, t: &Target) -> Result<String, String> {
    if t.arch != Arch::Arm64 {
        return Err("kernel seed lowers ARM64 first; x86-64 kernel is the next breadth".into());
    }
    let mut slots = BTreeMap::new();
    slots.insert("a".to_string(), 0u32);
    slots.insert("b".to_string(), 1);
    slots.insert("c".to_string(), 2);
    collect(s, &mut slots);
    if slots.len() > 31 {
        return Err("kernel seed: >31 slots exceeds the STUR/LDUR imm9 frame".into());
    }
    let frame = (((slots.len() * 8) + 15) / 16 * 16) as u16;
    let mut cg = Cg {
        out: String::new(), tmp: 3, free: Vec::new(), max: 3,
        arch: t.arch, has_fma: t.has_fma, lbl: 0, slots: Some(slots.clone()),
    };
    cg.emit(&format!("sub sp, sp, #{frame}"));
    cg.emit("stur w0, [sp, #0]");
    cg.emit("stur w1, [sp, #8]");
    cg.emit("stur w2, [sp, #16]");
    lower(s, &mut cg, &slots, frame)?;
    // Fall-through (no return on this path): result 0.
    cg.emit("mov w0, #0");
    cg.emit(&format!("add sp, sp, #{frame}"));
    cg.emit("ret");
    Ok(cg.out)
}

/// Lower program source (let/assign/if/while/for/return) to ARM64 assembly.
pub fn compile_program(src: &str, t: &Target) -> Result<String, String> {
    compile_stmt(&lift_program(src)?, t)
}