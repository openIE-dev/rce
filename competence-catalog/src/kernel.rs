//! Full-kernel seed — Stmt-level lowering to ARM64 and x86-64 with real stack frames, local
//! functions and calls (recursion included). Params and lets are spilled to fixed slots (no
//! register-aliasing class of bugs); call args go through frame scratch slots so parallel-move
//! clobbering cannot happen; `for` desugars to let+while+assign. Internal calling convention:
//! args in w0..w2 / edi,esi,edx (System V integer regs on x86-64), result in w0 / eax. The
//! expression fragment (arithmetic, comparisons, logicals, conditional select, capability-derived
//! FMA) is reused verbatim via the shared codegen. >3 args, closures, and >28-slot ARM64 frames
//! are the next breadth.

use std::collections::BTreeMap;

use rce_spec_check::ast::Expr;

use crate::compiler::{Arch, Cg, Target};
use crate::lut_emit::Stmt;
use crate::program::lift_program;

struct FnDef {
    label: String,
    params: Vec<String>,
    body: Stmt,
}

/// Collect stack slots for lets/fors (does not descend into fn bodies — they own their frames).
fn collect(s: &Stmt, slots: &mut BTreeMap<String, u32>) {
    match s {
        Stmt::Let(n, _) | Stmt::LetCall(n, ..) | Stmt::For(n, ..) => {
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

fn arg_reg(arch: Arch, k: usize) -> &'static str {
    match arch {
        Arch::Arm64 => ["w0", "w1", "w2"][k],
        Arch::X86 => ["edi", "esi", "edx"][k],
    }
}

fn result_reg(arch: Arch) -> &'static str {
    match arch {
        Arch::Arm64 => "w0",
        Arch::X86 => "eax",
    }
}

fn emit_store(cg: &mut Cg, regname: &str, off: u32) {
    match cg.arch {
        Arch::Arm64 => cg.emit(&format!("stur {}, [sp, #{}]", regname, off)),
        Arch::X86 => cg.emit(&format!("mov [rsp+{}], {}", off, regname)),
    }
}

fn emit_load(cg: &mut Cg, regname: &str, off: u32) {
    match cg.arch {
        Arch::Arm64 => cg.emit(&format!("ldur {}, [sp, #{}]", regname, off)),
        Arch::X86 => cg.emit(&format!("mov {}, [rsp+{}]", regname, off)),
    }
}

fn emit_frame(cg: &mut Cg, frame: u16, setup: bool) {
    match cg.arch {
        Arch::Arm64 => cg.emit(&format!("{} sp, sp, #{}", if setup { "sub" } else { "add" }, frame)),
        Arch::X86 => cg.emit(&format!("{} rsp, {}", if setup { "sub" } else { "add" }, frame)),
    }
}

/// ARM64: BL clobbers x30, so any function that may call must save/restore the link register
/// in its frame (last slot). x86 CALL pushes the return address on the stack — nothing to save.
fn emit_lr_save(cg: &mut Cg, frame: u16) {
    if cg.arch == Arch::Arm64 {
        cg.emit(&format!("stur x30, [sp, #{}]", frame - 8));
    }
}

fn emit_lr_restore(cg: &mut Cg, frame: u16) {
    if cg.arch == Arch::Arm64 {
        cg.emit(&format!("ldur x30, [sp, #{}]", frame - 8));
    }
}

fn emit_branch(cg: &mut Cg, cond_eq: bool, lbl: &str) {
    match cg.arch {
        Arch::Arm64 => cg.emit(&format!("{} {}", if cond_eq { "b.eq" } else { "b" }, lbl)),
        Arch::X86 => cg.emit(&format!("{} {}", if cond_eq { "je" } else { "jmp" }, lbl)),
    }
}

fn lower(
    s: &Stmt,
    cg: &mut Cg,
    slots: &BTreeMap<String, u32>,
    frame: u16,
    fns: &BTreeMap<String, FnDef>,
) -> Result<(), String> {
    match s {
        Stmt::Let(n, e) | Stmt::Assign(n, e) => {
            let slot = slot_of(slots, n)?;
            cg.gen(e, 0)?;
            emit_store(cg, result_reg(cg.arch), slot * 8);
        }
        Stmt::Return(e) => {
            cg.gen(e, 0)?;
            emit_lr_restore(cg, frame);
            emit_frame(cg, frame, false);
            cg.emit("ret");
        }
        Stmt::If(c, th, el) => {
            let cd = cg.alloc();
            cg.gen(c, cd)?;
            let z = cg.gen_test(cd)?;
            let lelse = cg.new_label();
            let lend = cg.new_label();
            emit_branch(cg, true, &lelse);
            cg.release(z);
            cg.release(cd);
            lower(th, cg, slots, frame, fns)?;
            emit_branch(cg, false, &lend);
            cg.label(&lelse);
            lower(el, cg, slots, frame, fns)?;
            cg.label(&lend);
        }
        Stmt::While(c, body) => {
            let ltop = cg.new_label();
            let lend = cg.new_label();
            cg.label(&ltop);
            let cd = cg.alloc();
            cg.gen(c, cd)?;
            let z = cg.gen_test(cd)?;
            emit_branch(cg, true, &lend);
            cg.release(z);
            cg.release(cd);
            lower(body, cg, slots, frame, fns)?;
            emit_branch(cg, false, &ltop);
            cg.label(&lend);
        }
        Stmt::For(n, lo, hi, body) => {
            lower(&Stmt::Let(n.clone(), lo.clone()), cg, slots, frame, fns)?;
            let cond = Expr::Bin("<".into(), Box::new(Expr::Name(n.clone(), 0)), Box::new(hi.clone()), 0);
            let inc = Stmt::Assign(
                n.clone(),
                Expr::Bin("+".into(), Box::new(Expr::Name(n.clone(), 0)), Box::new(Expr::Int(1, 0)), 0),
            );
            let mut seq = match body.as_ref() { Stmt::Seq(v) => v.clone(), other => vec![other.clone()] };
            seq.push(inc);
            lower(&Stmt::While(cond, Box::new(Stmt::Seq(seq))), cg, slots, frame, fns)?;
        }
        Stmt::Call(f, args) => { lower_call(f, args, cg, slots, fns)?; }
        Stmt::LetCall(n, f, args) => {
            lower_call(f, args, cg, slots, fns)?;
            let slot = slot_of(slots, n)?;
            emit_store(cg, result_reg(cg.arch), slot * 8);
        }
        Stmt::Seq(v) => { for x in v { lower(x, cg, slots, frame, fns)?; } }
        Stmt::LetFn(..) => {} // hoisted: emitted as separate labeled frames
        Stmt::Import(_) | Stmt::TypeDef(_) | Stmt::Attribute(_) | Stmt::GenericType(..) => {}
        _ => return Err("construct outside the kernel seed (match/async next)".into()),
    }
    Ok(())
}

/// Call lowering: args evaluated left-to-right into frame scratch slots (memory, so no
/// parallel-move clobbering), then loaded into the arg registers, then bl/call. Result in w0/eax.
fn lower_call(
    f: &str,
    args: &[Expr],
    cg: &mut Cg,
    slots: &BTreeMap<String, u32>,
    fns: &BTreeMap<String, FnDef>,
) -> Result<(), String> {
    let fd = fns.get(f).ok_or_else(|| format!("unknown function {f}"))?;
    if args.len() > 3 { return Err("kernel seed: >3 args (stack-passing next)".into()); }
    if args.len() != fd.params.len() { return Err(format!("arity mismatch calling {f}")); }
    let scratch = slots.len() as u32;
    for (k, a) in args.iter().enumerate() {
        let tt = cg.alloc();
        cg.gen(a, tt)?;
        let name = crate::compiler::reg(cg.arch, tt);
        emit_store(cg, &name, (scratch + k as u32) * 8);
        cg.release(tt);
    }
    for k in 0..args.len() {
        emit_load(cg, arg_reg(cg.arch, k), (scratch + k as u32) * 8);
    }
    match cg.arch {
        Arch::Arm64 => cg.emit(&format!("bl {}", fd.label)),
        Arch::X86 => cg.emit(&format!("call {}", fd.label)),
    }
    Ok(())
}

fn new_cg(arch: Arch, has_fma: bool, slots: BTreeMap<String, u32>, lbl: u32) -> Cg {
    Cg { out: String::new(), tmp: 3, free: Vec::new(), max: 3, arch, has_fma, lbl, slots: Some(slots) }
}

/// Lower a whole program (Stmt) to target assembly with stack frames for main and each local fn.
pub fn compile_stmt(s: &Stmt, t: &Target) -> Result<String, String> {
    let items = match s { Stmt::Seq(v) => v.clone(), other => vec![other.clone()] };
    let mut fns: BTreeMap<String, FnDef> = BTreeMap::new();
    let mut main = Vec::new();
    for st in items {
        if let Stmt::LetFn(n, params, body) = st {
            fns.insert(n.clone(), FnDef { label: format!("fn_{n}"), params, body: *body });
        } else {
            main.push(st);
        }
    }

    let mut slots = BTreeMap::new();
    slots.insert("a".to_string(), 0u32);
    slots.insert("b".to_string(), 1);
    slots.insert("c".to_string(), 2);
    for st in &main { collect(st, &mut slots); }
    // +3 scratch slots for call args, +1 for the ARM64 link register; STUR/LDUR imm9 caps at 255B.
    if t.arch == Arch::Arm64 && (slots.len() + 4) * 8 > 255 {
        return Err("kernel seed: frame exceeds the STUR/LDUR imm9 range".into());
    }
    let frame = (((slots.len() + 4) * 8 + 15) / 16 * 16) as u16;

    let mut cg = new_cg(t.arch, t.has_fma, slots.clone(), 0);
    emit_frame(&mut cg, frame, true);
    emit_lr_save(&mut cg, frame);
    for k in 0..3 {
        emit_store(&mut cg, arg_reg(t.arch, k), k as u32 * 8);
    }
    let main_seq = Stmt::Seq(main);
    lower(&main_seq, &mut cg, &slots, frame, &fns)?;
    // Fall-through (no return on this path): result 0.
    match t.arch {
        Arch::Arm64 => cg.emit("mov w0, #0"),
        Arch::X86 => cg.emit("mov eax, 0"),
    }
    emit_lr_restore(&mut cg, frame);
    emit_frame(&mut cg, frame, false);
    cg.emit("ret");
    let mut next_lbl = cg.lbl;

    for fd in fns.values() {
        let mut fslots = BTreeMap::new();
        for (i, p) in fd.params.iter().enumerate() {
            fslots.insert(p.clone(), i as u32);
        }
        collect(&fd.body, &mut fslots);
        if t.arch == Arch::Arm64 && (fslots.len() + 4) * 8 > 255 {
            return Err("kernel seed: fn frame exceeds the STUR/LDUR imm9 range".into());
        }
        let fframe = (((fslots.len() + 4) * 8 + 15) / 16 * 16) as u16;
        let mut fcg = new_cg(t.arch, t.has_fma, fslots.clone(), next_lbl);
        fcg.label(&fd.label);
        emit_frame(&mut fcg, fframe, true);
        emit_lr_save(&mut fcg, fframe);
        for (k, _) in fd.params.iter().enumerate() {
            emit_store(&mut fcg, arg_reg(t.arch, k), k as u32 * 8);
        }
        lower(&fd.body, &mut fcg, &fslots, fframe, &fns)?;
        match t.arch {
            Arch::Arm64 => fcg.emit("mov w0, #0"),
            Arch::X86 => fcg.emit("mov eax, 0"),
        }
        emit_lr_restore(&mut fcg, fframe);
        emit_frame(&mut fcg, fframe, false);
        fcg.emit("ret");
        next_lbl = fcg.lbl;
        cg.out.push_str(&fcg.out);
    }
    Ok(cg.out)
}

/// Lower program source (let/assign/if/while/for/fn/calls/return) to target assembly.
pub fn compile_program(src: &str, t: &Target) -> Result<String, String> {
    compile_stmt(&lift_program(src)?, t)
}