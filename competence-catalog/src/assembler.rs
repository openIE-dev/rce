//! Own assemblers (ARM64 + x86-64) — encode the instruction subsets RCE-Compile emits, so even
//! verification need not rely on a system assembler. Known-vector tested against the ARMv8 /
//! x86-64 encodings; `system_words` is an OPTIONAL cross-check (system assembler used only as a
//! verification tool; the compiler+assemblers are sovereign).

use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum Ins {
    MovImm(u32, u16),
    MovReg(u32, u32),
    Add(u32, u32, u32),
    Sub(u32, u32, u32),
    Subs(u32, u32, u32),
    Neg(u32, u32),
    Mul(u32, u32, u32),
    Madd(u32, u32, u32, u32),
    Msub(u32, u32, u32, u32),
    Sdiv(u32, u32, u32),
    And(u32, u32, u32),
    Orr(u32, u32, u32),
    Cset(u32, u8),
    /// ADD SP, SP, #imm (64-bit immediate, for frame teardown).
    AddSp(u16),
    /// SUB SP, SP, #imm (64-bit immediate, for frame setup).
    SubSp(u16),
    /// ADD Wd, Wn, #imm (32-bit immediate).
    AddImm(u32, u32, u16),
    /// STUR Wt, [Xn, #off] (unscaled signed 9-bit offset).
    Stur(u32, u32, i16),
    /// LDUR Wt, [Xn, #off].
    Ldur(u32, u32, i16),
    Ret,
}

fn regnum(s: &str) -> Result<u32, String> {
    let s = s.trim();
    if s == "wzr" || s == "sp" { return Ok(31); }
    if let Some(n) = s.strip_prefix('w') {
        n.parse::<u32>().map_err(|e| e.to_string())
    } else {
        Err(format!("not a W register: {s}"))
    }
}

/// Parse `[sp, #8]` (already split on commas: base part and offset part).
fn memref(base: &str, off: &str) -> Result<(u32, i16), String> {
    let rn = regnum(base.trim().trim_start_matches('['))?;
    let o = off.trim().trim_end_matches(']').strip_prefix('#')
        .ok_or_else(|| format!("bad memref offset {off}"))?;
    Ok((rn, o.parse::<i16>().map_err(|e| e.to_string())?))
}

/// ARM64 condition codes (the encoding's cond field).
pub fn cond_code(s: &str) -> Result<u8, String> {
    Ok(match s {
        "eq" => 0, "ne" => 1, "hs" => 2, "lo" => 3, "mi" => 4, "pl" => 5,
        "vs" => 6, "vc" => 7, "hi" => 8, "ls" => 9, "ge" => 10, "lt" => 11,
        "gt" => 12, "le" => 13, "al" => 14,
        _ => return Err(format!("unknown cond {s}")),
    })
}

fn cond_invert(c: u8) -> u8 {
    match c {
        0 => 1, 1 => 0, 2 => 3, 3 => 2, 4 => 5, 5 => 4, 6 => 7, 7 => 6,
        8 => 9, 9 => 8, 10 => 11, 11 => 10, 12 => 13, 13 => 12, _ => 14,
    }
}

pub fn parse_line(line: &str) -> Result<Ins, String> {
    let line = line.trim();
    if line == "ret" { return Ok(Ins::Ret); }
    let (op, rest) = line.split_once(' ').ok_or_else(|| format!("bad line {line}"))?;
    let args: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
    let r = |i: usize| regnum(args[i]);
    match op {
        "mov" => {
            if args[1].starts_with('#') {
                Ok(Ins::MovImm(r(0)?, args[1][1..].parse::<u16>().map_err(|e| e.to_string())?))
            } else {
                Ok(Ins::MovReg(r(0)?, r(1)?))
            }
        }
        "add" => {
            if args[2].starts_with('#') {
                let imm = args[2][1..].parse::<u16>().map_err(|e| e.to_string())?;
                if args[0].trim() == "sp" { Ok(Ins::AddSp(imm)) } else { Ok(Ins::AddImm(r(0)?, r(1)?, imm)) }
            } else {
                Ok(Ins::Add(r(0)?, r(1)?, r(2)?))
            }
        }
        "sub" => {
            if args[2].starts_with('#') {
                let imm = args[2][1..].parse::<u16>().map_err(|e| e.to_string())?;
                if args[0].trim() == "sp" { Ok(Ins::SubSp(imm)) } else { Err("sub immediate only for sp".into()) }
            } else {
                Ok(Ins::Sub(r(0)?, r(1)?, r(2)?))
            }
        }
        "subs" => Ok(Ins::Subs(r(0)?, r(1)?, r(2)?)),
        "stur" => { let (rn, off) = memref(args[1], args[2])?; Ok(Ins::Stur(r(0)?, rn, off)) }
        "ldur" => { let (rn, off) = memref(args[1], args[2])?; Ok(Ins::Ldur(r(0)?, rn, off)) }
        "neg" => Ok(Ins::Neg(r(0)?, r(1)?)),
        "mul" => Ok(Ins::Mul(r(0)?, r(1)?, r(2)?)),
        "madd" => Ok(Ins::Madd(r(0)?, r(1)?, r(2)?, r(3)?)),
        "msub" => Ok(Ins::Msub(r(0)?, r(1)?, r(2)?, r(3)?)),
        "sdiv" => Ok(Ins::Sdiv(r(0)?, r(1)?, r(2)?)),
        "and" => Ok(Ins::And(r(0)?, r(1)?, r(2)?)),
        "orr" => Ok(Ins::Orr(r(0)?, r(1)?, r(2)?)),
        "cset" => Ok(Ins::Cset(r(0)?, cond_code(args[1])?)),
        _ => Err(format!("unknown op {op}")),
    }
}

pub fn encode(i: &Ins) -> u32 {
    match *i {
        Ins::MovImm(d, imm) => 0x5280_0000 | ((imm as u32) << 5) | d,
        Ins::MovReg(d, m) => 0x2A00_0000 | (m << 16) | (31 << 5) | d, // ORR Wd,WZR,Wm
        Ins::Add(d, n, m) => 0x0B00_0000 | (m << 16) | (n << 5) | d,
        Ins::Sub(d, n, m) => 0x4B00_0000 | (m << 16) | (n << 5) | d,
        Ins::Subs(d, n, m) => 0x6B00_0000 | (m << 16) | (n << 5) | d,
        Ins::Neg(d, m) => 0x4B00_0000 | (m << 16) | (31 << 5) | d, // SUB Wd,WZR,Wm
        Ins::Mul(d, n, m) => 0x1B00_0000 | (m << 16) | (31 << 10) | (n << 5) | d, // MADD ..,WZR
        Ins::Madd(d, n, m, a) => 0x1B00_0000 | (m << 16) | (a << 10) | (n << 5) | d,
        Ins::Msub(d, n, m, a) => 0x1B00_8000 | (m << 16) | (a << 10) | (n << 5) | d,
        Ins::Sdiv(d, n, m) => 0x1AC0_0C00 | (m << 16) | (n << 5) | d,
        Ins::And(d, n, m) => 0x0A00_0000 | (m << 16) | (n << 5) | d,
        Ins::Orr(d, n, m) => 0x2A00_0000 | (m << 16) | (n << 5) | d,
        // CSET Wd,cond = CSINC Wd,WZR,WZR,invert(cond)
        Ins::Cset(d, c) => 0x1A80_0400 | ((cond_invert(c) as u32) << 12) | (31 << 16) | (31 << 5) | d,
        Ins::AddSp(imm) => 0x9100_0000 | ((imm as u32) << 10) | (31 << 5) | 31,
        Ins::SubSp(imm) => 0xD100_0000 | ((imm as u32) << 10) | (31 << 5) | 31,
        Ins::AddImm(d, n, imm) => 0x1100_0000 | ((imm as u32) << 10) | (n << 5) | d,
        Ins::Stur(t, n, off) => 0xB800_0000 | (((off as u32) & 0x1FF) << 12) | (n << 5) | t,
        Ins::Ldur(t, n, off) => 0xB840_0000 | (((off as u32) & 0x1FF) << 12) | (n << 5) | t,
        Ins::Ret => 0xD65F_03C0,
    }
}

enum P {
    Ins(Ins),
    B(String),
    Bcond(u8, String),
}

/// Assemble RCE-Compile ARM64 output to little-endian machine code (two-pass label resolution).
pub fn assemble(asm: &str) -> Result<Vec<u8>, String> {
    let mut prog: Vec<P> = Vec::new();
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    for line in asm.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('.') { continue; }
        if let Some(name) = t.strip_suffix(':') {
            labels.insert(name.trim().to_string(), prog.len());
            continue;
        }
        if let Some(rest) = t.strip_prefix("b.") {
            let (cond, lbl) = rest.split_once(' ').ok_or_else(|| format!("bad branch {t}"))?;
            prog.push(P::Bcond(cond_code(cond.trim())?, lbl.trim().to_string()));
        } else if let Some(lbl) = t.strip_prefix("b ") {
            prog.push(P::B(lbl.trim().to_string()));
        } else {
            prog.push(P::Ins(parse_line(t)?));
        }
    }
    let mut out = Vec::new();
    for (i, p) in prog.iter().enumerate() {
        let w = match p {
            P::Ins(ins) => encode(ins),
            P::B(lbl) => {
                let t = *labels.get(lbl).ok_or_else(|| format!("undefined label {lbl}"))?;
                0x1400_0000 | (((t as i64 - i as i64) as u32) & 0x03FF_FFFF)
            }
            P::Bcond(c, lbl) => {
                let t = *labels.get(lbl).ok_or_else(|| format!("undefined label {lbl}"))?;
                0x5400_0000 | ((((t as i64 - i as i64) as u32) & 0x7_FFFF) << 5) | *c as u32
            }
        };
        out.extend_from_slice(&w.to_le_bytes());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// x86-64 (32-bit operand subset RCE-Compile emits; System V / macOS calling shape)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum InsX86 {
    MovImm(u32, i32),
    MovReg(u32, u32),
    Add(u32, u32),
    Sub(u32, u32),
    Cmp(u32, u32),
    And(u32, u32),
    Or(u32, u32),
    Neg(u32),
    Imul(u32, u32),
    Cdq,
    Idiv(u32),
    Setcc(u8),  // into al
    Movzx(u32), // from al
    Ret,
}

fn regnum_x86(s: &str) -> Result<u32, String> {
    const R: [&str; 16] = [
        "eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi",
        "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d",
    ];
    R.iter().position(|n| *n == s.trim()).map(|i| i as u32)
        .ok_or_else(|| format!("not an x86-64 r32 register: {s}"))
}

fn cond_code_x86(s: &str) -> Result<u8, String> {
    Ok(match s {
        "o" => 0, "no" => 1, "b" => 2, "ae" => 3, "e" => 4, "ne" => 5,
        "be" => 6, "a" => 7, "s" => 8, "ns" => 9, "p" => 10, "np" => 11,
        "l" => 12, "ge" => 13, "le" => 14, "g" => 15,
        _ => return Err(format!("unknown x86 cond {s}")),
    })
}

fn rex(reg_field: u32, rm_field: u32) -> Option<u8> {
    if reg_field >= 8 || rm_field >= 8 {
        Some(0x40 | (((reg_field >= 8) as u8) << 2) | ((rm_field >= 8) as u8))
    } else {
        None
    }
}

fn modrm(reg: u32, rm: u32) -> u8 {
    0xC0 | (((reg & 7) as u8) << 3) | ((rm & 7) as u8)
}

pub fn encode_x86(i: &InsX86) -> Vec<u8> {
    let mut b = Vec::new();
    match *i {
        InsX86::MovImm(d, imm) => {
            if let Some(r) = rex(0, d) { b.push(r); }
            b.push(0xB8 + ((d & 7) as u8));
            b.extend_from_slice(&imm.to_le_bytes());
        }
        InsX86::MovReg(d, s) => {
            if let Some(r) = rex(s, d) { b.push(r); }
            b.push(0x89);
            b.push(modrm(s, d));
        }
        InsX86::Add(d, s) | InsX86::Sub(d, s) | InsX86::Cmp(d, s) | InsX86::And(d, s) | InsX86::Or(d, s) => {
            let op = match i {
                InsX86::Add(..) => 0x01, InsX86::Sub(..) => 0x29, InsX86::Cmp(..) => 0x39,
                InsX86::And(..) => 0x21, _ => 0x09,
            };
            if let Some(r) = rex(s, d) { b.push(r); }
            b.push(op);
            b.push(modrm(s, d));
        }
        InsX86::Neg(d) => {
            if let Some(r) = rex(0, d) { b.push(r); }
            b.push(0xF7);
            b.push(0xD8 | ((d & 7) as u8)); // /3
        }
        InsX86::Imul(d, s) => {
            if let Some(r) = rex(d, s) { b.push(r); }
            b.push(0x0F);
            b.push(0xAF);
            b.push(modrm(d, s));
        }
        InsX86::Cdq => b.push(0x99),
        InsX86::Idiv(s) => {
            if let Some(r) = rex(0, s) { b.push(r); }
            b.push(0xF7);
            b.push(0xF8 | ((s & 7) as u8)); // /7
        }
        InsX86::Setcc(cc) => {
            b.push(0x0F);
            b.push(0x90 + cc);
            b.push(0xC0); // al
        }
        InsX86::Movzx(d) => {
            if let Some(r) = rex(d, 0) { b.push(r); }
            b.push(0x0F);
            b.push(0xB6);
            b.push(modrm(d, 0)); // from al
        }
        InsX86::Ret => b.push(0xC3),
    }
    b
}

pub fn parse_line_x86(line: &str) -> Result<InsX86, String> {
    let line = line.trim();
    if line == "ret" { return Ok(InsX86::Ret); }
    if line == "cdq" { return Ok(InsX86::Cdq); }
    let (op, rest) = line.split_once(' ').ok_or_else(|| format!("bad line {line}"))?;
    let args: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
    let r = |i: usize| regnum_x86(args[i]);
    if let Some(cc) = op.strip_prefix("set") {
        if args[0] != "al" { return Err("setcc target must be al".into()); }
        return Ok(InsX86::Setcc(cond_code_x86(cc)?));
    }
    match op {
        "mov" => {
            if let Ok(imm) = args[1].parse::<i32>() {
                Ok(InsX86::MovImm(r(0)?, imm))
            } else {
                Ok(InsX86::MovReg(r(0)?, r(1)?))
            }
        }
        "movzx" => {
            if args[1] != "al" { return Err("movzx source must be al".into()); }
            Ok(InsX86::Movzx(r(0)?))
        }
        "add" => Ok(InsX86::Add(r(0)?, r(1)?)),
        "sub" => Ok(InsX86::Sub(r(0)?, r(1)?)),
        "cmp" => Ok(InsX86::Cmp(r(0)?, r(1)?)),
        "and" => Ok(InsX86::And(r(0)?, r(1)?)),
        "or" => Ok(InsX86::Or(r(0)?, r(1)?)),
        "neg" => Ok(InsX86::Neg(r(0)?)),
        "imul" => Ok(InsX86::Imul(r(0)?, r(1)?)),
        "idiv" => Ok(InsX86::Idiv(r(0)?)),
        _ => Err(format!("unknown x86 op {op}")),
    }
}

enum XP {
    Ins(InsX86),
    Je(String),
    Jmp(String),
}

impl XP {
    fn lbl(&self) -> Option<&str> {
        match self { XP::Je(l) | XP::Jmp(l) => Some(l), _ => None }
    }
}

/// Assemble RCE-Compile x86-64 output to machine code. Branch relaxation: je/jmp use rel8 (2B)
/// when the displacement fits, else rel32 (6B) — sizes are solved to a fixed point (shrinking
/// only brings targets closer, so the pass is monotone and terminates).
pub fn assemble_x86(asm: &str) -> Result<Vec<u8>, String> {
    let mut prog: Vec<XP> = Vec::new();
    let mut labels: BTreeMap<String, usize> = BTreeMap::new();
    for line in asm.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('.') { continue; }
        if let Some(name) = t.strip_suffix(':') {
            labels.insert(name.trim().to_string(), prog.len());
            continue;
        }
        let p = if let Some(lbl) = t.strip_prefix("je ") {
            XP::Je(lbl.trim().to_string())
        } else if let Some(lbl) = t.strip_prefix("jmp ") {
            XP::Jmp(lbl.trim().to_string())
        } else {
            XP::Ins(parse_line_x86(t)?)
        };
        prog.push(p);
    }
    let mut relaxed = vec![false; prog.len()];
    let addr_of = |relaxed: &[bool]| -> Vec<usize> {
        let mut addr = Vec::new();
        let mut at = 0usize;
        for (i, p) in prog.iter().enumerate() {
            addr.push(at);
            at += match p {
                XP::Ins(ins) => encode_x86(ins).len(),
                _ => if relaxed[i] { 2 } else { 6 },
            };
        }
        addr
    };
    loop {
        let addr = addr_of(&relaxed);
        let mut changed = false;
        for (i, p) in prog.iter().enumerate() {
            if relaxed[i] { continue; }
            if let Some(lbl) = p.lbl() {
                let ti = *labels.get(lbl).ok_or_else(|| format!("undefined label {lbl}"))?;
                let rel = addr[ti] as i64 - (addr[i] as i64 + 6);
                if (-128..=127).contains(&rel) {
                    relaxed[i] = true;
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }
    let addr = addr_of(&relaxed);
    let mut out = Vec::new();
    for (i, p) in prog.iter().enumerate() {
        match p {
            XP::Ins(ins) => out.extend_from_slice(&encode_x86(ins)),
            XP::Je(lbl) | XP::Jmp(lbl) => {
                let ti = *labels.get(lbl).ok_or_else(|| format!("undefined label {lbl}"))?;
                let je = matches!(p, XP::Je(_));
                if relaxed[i] {
                    let rel = (addr[ti] as i64 - (addr[i] as i64 + 2)) as i8;
                    out.push(if je { 0x74 } else { 0xEB });
                    out.push(rel as u8);
                } else {
                    let rel = (addr[ti] as i64 - (addr[i] as i64 + 6)) as i32;
                    if je { out.extend_from_slice(&[0x0F, 0x84]); } else { out.push(0xE9); }
                    out.extend_from_slice(&rel.to_le_bytes());
                }
            }
        }
    }
    Ok(out)
}

/// OPTIONAL cross-check: assemble with the system toolchain (clang -target … + otool) and return
/// the __text bytes. Verification only — never used by the sovereign compile path. Err when the
/// toolchain is unavailable (callers skip honestly).
pub fn system_bytes(asm: &str, arch: &str) -> Result<Vec<u8>, String> {
    let dir = std::env::temp_dir();
    let tag = format!("rce_xcheck_{}", std::process::id());
    let s = dir.join(format!("{tag}_{arch}.s"));
    let o = dir.join(format!("{tag}_{arch}.o"));
    let target = if arch == "arm64" { "arm64-apple-macosx11.0" } else { "x86_64-apple-macosx10.15" };
    let wrap = if arch == "arm64" {
        format!(".p2align 2\n.globl _rce_x\n_rce_x:\n{asm}\n")
    } else {
        format!(".intel_syntax noprefix\n.globl _rce_x\n_rce_x:\n{asm}\n")
    };
    std::fs::write(&s, wrap).map_err(|e| e.to_string())?;
    let ok = std::process::Command::new("clang")
        .args(["-target", target, "-c", s.to_str().unwrap(), "-o", o.to_str().unwrap()])
        .output().map(|out| out.status.success()).unwrap_or(false);
    if !ok { return Err("system assembler unavailable".into()); }
    let out = std::process::Command::new("otool")
        .args(["-s", "__TEXT", "__text", o.to_str().unwrap()])
        .output().map_err(|e| e.to_string())?;
    if !out.status.success() { return Err("otool unavailable".into()); }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut bytes = Vec::new();
    for tok in text.split_whitespace() {
        let h = tok.strip_prefix("0x").unwrap_or(tok);
        if !h.chars().all(|c| c.is_ascii_hexdigit()) { continue; }
        match h.len() {
            8 => bytes.extend_from_slice(&u32::from_str_radix(h, 16).map_err(|e| e.to_string())?.to_le_bytes()),
            2 => bytes.push(u8::from_str_radix(h, 16).map_err(|e| e.to_string())?),
            _ => {}
        }
    }
    let _ = std::fs::remove_file(&s);
    let _ = std::fs::remove_file(&o);
    Ok(bytes)
}

/// Group a byte stream into little-endian u32 words, zero-padding the tail (matches otool -s).
pub fn le_words(bytes: &[u8]) -> Vec<u32> {
    let mut padded = bytes.to_vec();
    while padded.len() % 4 != 0 { padded.push(0); }
    padded.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}