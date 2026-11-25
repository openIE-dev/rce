//! The RCE Compiler (seed) — our own compiler that obviates the need for any other.
//!
//! Because the destination is the bounded, costed primitive floor, the Form can be lowered
//! **directly** to machine primitives (ARM64 / x86-64 assembly) without LLVM/GCC/clang in the
//! loop. The compiler is a deterministic graph→floor projection plus placement (living
//! optimization): it emits the fused multiply-add primitive iff the capability lookup says the
//! fabric has it — the carve-out is derived, not flagged. This is the seed of the dedicated
//! RCE Compiler; breadth (full kernel, more arches, register allocation) is the open work.

use rce_spec_check::ast::Expr;

use crate::canonical::normalize;
use crate::lift::lift_expr;

/// The target architecture (the primitive floor's encoding).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    Arm64,
    X86,
}

/// A compilation target: architecture + capability (a lookup result).
pub struct Target {
    pub arch: Arch,
    pub has_fma: bool,
}

fn reg(arch: Arch, i: u32) -> String {
    match arch {
        Arch::Arm64 => format!("w{}", i),
        Arch::X86 => {
            const R: [&str; 14] = ["eax", "ecx", "edx", "ebx", "esi", "edi", "r8d", "r9d", "r10d", "r11d", "r12d", "r13d", "r14d", "r15d"];
            R[i as usize % R.len()].to_string()
        }
    }
}

struct Cg {
    out: String,
    tmp: u32,
    arch: Arch,
    has_fma: bool,
}

impl Cg {
    fn emit(&mut self, s: &str) {
        self.out.push_str("    ");
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn alloc(&mut self) -> u32 {
        let t = self.tmp;
        self.tmp += 1;
        t
    }
    fn param(&self, n: &str) -> Option<u32> {
        match n {
            "a" => Some(0),
            "b" => Some(1),
            "c" => Some(2),
            _ => None,
        }
    }

    fn gen(&mut self, e: &Expr, dst: u32) -> Result<(), String> {
        use Expr::*;
        let d = reg(self.arch, dst);
        match e {
            Int(n, _) => match self.arch {
                Arch::Arm64 => self.emit(&format!("mov {}, #{}", d, n)),
                Arch::X86 => self.emit(&format!("mov {}, {}", d, n)),
            },
            Name(n, _) => {
                let src = self.param(n).ok_or_else(|| format!("unbound name {n}"))?;
                if src != dst {
                    let s = reg(self.arch, src);
                    self.emit(&format!("mov {}, {}", d, s));
                }
            }
            Neg(a, _) => {
                self.gen(a, dst)?;
                match self.arch {
                    Arch::Arm64 => self.emit(&format!("neg {}, {}", d, d)),
                    Arch::X86 => self.emit(&format!("neg {}", d)),
                }
            }
            Bin(op, l, r, _) => {
                // Placement-derived fusion: a*b + c -> one madd where the fabric has FMA.
                if op == "+" && self.has_fma && self.arch == Arch::Arm64 {
                    if let Bin(m, a, b, _) = l.as_ref() {
                        if m == "*" {
                            let t1 = self.alloc();
                            let t2 = self.alloc();
                            let t3 = self.alloc();
                            self.gen(a, t1)?;
                            self.gen(b, t2)?;
                            self.gen(r, t3)?;
                            let (r1, r2, r3) = (reg(self.arch, t1), reg(self.arch, t2), reg(self.arch, t3));
                            self.emit(&format!("madd {}, {}, {}, {}", d, r1, r2, r3));
                            return Ok(());
                        }
                    }
                }
                self.gen(l, dst)?;
                let t = self.alloc();
                self.gen(r, t)?;
                let s = reg(self.arch, t);
                match (self.arch, op.as_str()) {
                    (Arch::Arm64, "+") => self.emit(&format!("add {}, {}, {}", d, d, s)),
                    (Arch::Arm64, "-") => self.emit(&format!("sub {}, {}, {}", d, d, s)),
                    (Arch::Arm64, "*") => self.emit(&format!("mul {}, {}, {}", d, d, s)),
                    (Arch::Arm64, "/") => self.emit(&format!("sdiv {}, {}, {}", d, d, s)),
                    (Arch::Arm64, "%") => {
                        let q = self.alloc();
                        let qq = reg(self.arch, q);
                        self.emit(&format!("sdiv {}, {}, {}", qq, d, s));
                        self.emit(&format!("msub {}, {}, {}, {}", d, qq, s, d));
                    }
                    (Arch::X86, "+") => self.emit(&format!("add {}, {}", d, s)),
                    (Arch::X86, "-") => self.emit(&format!("sub {}, {}", d, s)),
                    (Arch::X86, "*") => self.emit(&format!("imul {}, {}", d, s)),
                    (Arch::X86, "/") | (Arch::X86, "%") => {
                        // dividend -> eax, divisor in a safe reg; cdq; idiv.
                        if dst != 0 {
                            self.emit(&format!("mov eax, {}", d));
                        }
                        self.emit("cdq");
                        self.emit(&format!("idiv {}", s));
                        let res = if op == "/" { "eax" } else { "edx" };
                        if dst != 0 && !(op == "/" && dst == 0) {
                            self.emit(&format!("mov {}, {}", d, res));
                        }
                    }
                    _ => return Err(format!("no primitive for {op}")),
                }
            }
            _ => return Err("construct outside the compiler fragment".into()),
        }
        Ok(())
    }
}

/// Compile a canonical Form directly to target assembly — no external compiler.
pub fn compile(e: &Expr, t: &Target) -> Result<String, String> {
    let mut cg = Cg { out: String::new(), tmp: 3, arch: t.arch, has_fma: t.has_fma };
    cg.gen(&normalize(e.clone()), 0)?;
    cg.emit(if t.arch == Arch::Arm64 { "ret" } else { "ret" });
    Ok(cg.out)
}

/// Compile source (any surface in the fragment) to target assembly.
pub fn compile_src(src: &str, t: &Target) -> Result<String, String> {
    compile(&lift_expr(src)?, t)
}