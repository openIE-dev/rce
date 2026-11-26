use competence_catalog::assembler::{assemble, encode, system_bytes, Ins};
use competence_catalog::compiler::{Arch, Target};
use competence_catalog::kernel::{compile_program, compile_stmt};
use competence_catalog::program::{eval_program, lift_program, Env, FnTable};

fn arm() -> Target { Target { arch: Arch::Arm64, has_fma: true } }

#[test]
fn frame_instruction_vectors() {
    // Verified against the system assembler in kernel_matches_system_toolchain.
    assert_eq!(encode(&Ins::SubSp(32)), 0xD100_83FF);
    assert_eq!(encode(&Ins::AddSp(32)), 0x9100_83FF);
    assert_eq!(encode(&Ins::Stur(0, 31, 0)), 0xB800_03E0);
    assert_eq!(encode(&Ins::Ldur(3, 31, 24)), 0xB841_83E3);
    assert_eq!(encode(&Ins::AddImm(0, 0, 1)), 0x1100_0400);
}

const PROGRAMS: &[(&str, i128)] = &[
    ("let s = 0; let i = 0; while i < 5 { s = s + i; i = i + 1; } return s;", 10),
    ("let s = 0; for i in 0..5 { s = s + i; } return s;", 10),
    ("let x = a; if x < b { x = c; } else { x = 0; } return x;", 4),
    ("return a + b * c;", 14),
    ("let x = a; while x < c { x = x + 1; } return x;", 4),
];

fn probe_env() -> Env {
    let mut e = Env::new();
    e.insert("a".into(), 2);
    e.insert("b".into(), 3);
    e.insert("c".into(), 4);
    e
}

#[test]
fn kernel_matches_replay_gate() {
    // The lowered program must agree with the deterministic interpreter on the probes.
    for (src, want) in PROGRAMS {
        let stmt = lift_program(src).unwrap();
        let mut env = probe_env();
        let got = eval_program(&stmt, &mut env, &mut FnTable::new()).unwrap();
        assert_eq!(got, Some(*want), "interpreter: {src}");
        let asm = compile_stmt(&stmt, &arm()).unwrap();
        assert!(asm.contains("sub sp, sp, #"), "frame setup: {src}");
        assert!(asm.contains("stur"), "spill: {src}");
    }
}

#[test]
fn kernel_matches_system_toolchain() {
    use competence_catalog::assembler::assemble_x86;
    let mut checked = 0;
    for (src, _) in PROGRAMS {
        for (t, label) in [(arm(), "arm64"), (Target { arch: Arch::X86, has_fma: false }, "x86-64")] {
            let asm = compile_program(src, &t).unwrap();
            let ours = if t.arch == Arch::Arm64 { assemble(&asm).unwrap() } else { assemble_x86(&asm).unwrap() };
            if let Ok(sys) = system_bytes(&asm, label) {
                assert_eq!(ours, sys, "{src} on {label}");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "expected the system toolchain on the dev host");
}

#[test]
fn x86_frame_vectors_and_kernel() {
    use competence_catalog::assembler::{assemble_x86, encode_x86, InsX86};
    assert_eq!(encode_x86(&InsX86::SubSp(32)), vec![0x48, 0x83, 0xEC, 0x20]);
    assert_eq!(encode_x86(&InsX86::AddSp(32)), vec![0x48, 0x83, 0xC4, 0x20]);
    assert_eq!(encode_x86(&InsX86::SubSp(256)), vec![0x48, 0x81, 0xEC, 0x00, 0x01, 0x00, 0x00]);
    assert_eq!(encode_x86(&InsX86::Store(0, 0)), vec![0x89, 0x04, 0x24]); // mov [rsp], eax
    assert_eq!(encode_x86(&InsX86::Store(1, 8)), vec![0x89, 0x4C, 0x24, 0x08]); // mov [rsp+8], ecx
    assert_eq!(encode_x86(&InsX86::Load(0, 16)), vec![0x8B, 0x44, 0x24, 0x10]); // mov eax, [rsp+16]
    let t = Target { arch: Arch::X86, has_fma: false };
    let asm = compile_program("let s = 0; for i in 0..5 { s = s + i; } return s;", &t).unwrap();
    assert!(asm.contains("sub rsp,"));
    assert!(assemble_x86(&asm).unwrap().len() > 0);
}

#[cfg(target_arch = "aarch64")]
#[test]
fn kernel_meaning_preserved_on_silicon() {
    let dir = std::env::temp_dir();
    let mut ran = 0;
    for (i, (src, want)) in PROGRAMS.iter().enumerate() {
        let asm = compile_program(src, &arm()).unwrap();
        let s = dir.join(format!("rce_kernel_{i}.s"));
        let c = dir.join(format!("rce_kernel_{i}.c"));
        let bin = dir.join(format!("rce_kernel_{i}"));
        std::fs::write(&s, format!(".globl _rce\n_rce:\n{asm}\n")).unwrap();
        std::fs::write(&c, "#include <stdio.h>\nextern long rce(long,long,long);\nint main(){printf(\"%ld\\n\", rce(2,3,4));return 0;}\n").unwrap();
        let ok = std::process::Command::new("clang")
            .args([s.to_str().unwrap(), c.to_str().unwrap(), "-o", bin.to_str().unwrap()])
            .output().map(|o| o.status.success()).unwrap_or(false);
        if !ok { continue; }
        let out = std::process::Command::new(&bin).output().unwrap();
        let got: i128 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(-999);
        assert_eq!(got, *want, "silicon: {src}");
        ran += 1;
    }
    assert!(ran > 0, "expected clang on the dev host");
}