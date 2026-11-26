//! Own ARM64 assembler — encodes the instruction subset RCE-Compile emits, so even verification
//! need not rely on a system assembler. Known-vector tested against the ARMv8 encoding.
//! (System assembler remains an optional cross-check; the compiler+assembler are sovereign.)

#[derive(Debug, Clone, PartialEq)]
pub enum Ins {
    MovImm(u32, u16),
    MovReg(u32, u32),
    Add(u32, u32, u32),
    Sub(u32, u32, u32),
    Neg(u32, u32),
    Mul(u32, u32, u32),
    Madd(u32, u32, u32, u32),
    Msub(u32, u32, u32, u32),
    Sdiv(u32, u32, u32),
    Ret,
}

fn regnum(s: &str) -> Result<u32, String> {
    let s = s.trim();
    if let Some(n) = s.strip_prefix('w') {
        n.parse::<u32>().map_err(|e| e.to_string())
    } else {
        Err(format!("not a W register: {s}")
        )
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
        "add" => Ok(Ins::Add(r(0)?, r(1)?, r(2)?)),
        "sub" => Ok(Ins::Sub(r(0)?, r(1)?, r(2)?)),
        "neg" => Ok(Ins::Neg(r(0)?, r(1)?)),
        "mul" => Ok(Ins::Mul(r(0)?, r(1)?, r(2)?)),
        "madd" => Ok(Ins::Madd(r(0)?, r(1)?, r(2)?, r(3)?)),
        "msub" => Ok(Ins::Msub(r(0)?, r(1)?, r(2)?, r(3)?)),
        "sdiv" => Ok(Ins::Sdiv(r(0)?, r(1)?, r(2)?)),
        _ => Err(format!("unknown op {op}")),
    }
}

pub fn encode(i: &Ins) -> u32 {
    match *i {
        Ins::MovImm(d, imm) => 0x5280_0000 | ((imm as u32) << 5) | d,
        Ins::MovReg(d, m) => 0x2A00_0000 | (m << 16) | (31 << 5) | d, // ORR Wd,WZR,Wm
        Ins::Add(d, n, m) => 0x0B00_0000 | (m << 16) | (n << 5) | d,
        Ins::Sub(d, n, m) => 0x4B00_0000 | (m << 16) | (n << 5) | d,
        Ins::Neg(d, m) => 0x4B00_0000 | (m << 16) | (31 << 5) | d, // SUB Wd,WZR,Wm
        Ins::Mul(d, n, m) => 0x1B00_0000 | (m << 16) | (31 << 10) | (n << 5) | d, // MADD ..,WZR
        Ins::Madd(d, n, m, a) => 0x1B00_0000 | (m << 16) | (a << 10) | (n << 5) | d,
        Ins::Msub(d, n, m, a) => 0x1B00_8000 | (m << 16) | (a << 10) | (n << 5) | d,
        Ins::Sdiv(d, n, m) => 0x1AC0_0C00 | (m << 16) | (n << 5) | d,
        Ins::Ret => 0xD65F_03C0,
    }
}

/// Assemble RCE-Compile ARM64 output to little-endian machine code.
pub fn assemble(asm: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for line in asm.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with('.') || t.ends_with(':') { continue; }
        let ins = parse_line(t)?;
        out.extend_from_slice(&encode(&ins).to_le_bytes());
    }
    Ok(out)
}
