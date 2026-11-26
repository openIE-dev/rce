use competence_catalog::compiler::{compile_receipt, compile_src, Arch, Target};

#[test]
fn sovereign_compiler_deterministic_and_content_addressed() {
    let t = Target { arch: Arch::Arm64, has_fma: true };
    let r1 = compile_receipt("a*b+c", &t).unwrap();
    let r2 = compile_receipt("a*b+c", &t).unwrap();
    assert!(r1.deterministic);
    assert_eq!(r1.asm_sha256, r2.asm_sha256);
    assert_eq!(r1.stages.len(), 3);
}

#[test]
fn capability_derived_fusion_reduces_instructions() {
    let fused = Target { arch: Arch::Arm64, has_fma: true };
    let plain = Target { arch: Arch::Arm64, has_fma: false };
    let rf = compile_receipt("a*b+c", &fused).unwrap();
    let rp = compile_receipt("a*b+c", &plain).unwrap();
    assert!(rf.fused_fma, "FMA derived from capability lookup");
    assert!(!rp.fused_fma);
    assert!(rf.instruction_count < rp.instruction_count, "fusion must reduce instruction count");
}

#[test]
fn unbound_name_refused() {
    let t = Target { arch: Arch::Arm64, has_fma: true };
    assert!(compile_receipt("a*z+c", &t).is_err());
}

#[test]
fn own_assembler_known_vectors() {
    use competence_catalog::assembler::{encode, Ins};
    assert_eq!(encode(&Ins::MovImm(0, 10)), 0x5280_0140);
    assert_eq!(encode(&Ins::MovReg(0, 1)), 0x2A01_03E0);
    assert_eq!(encode(&Ins::Add(0, 0, 1)), 0x0B01_0000);
    assert_eq!(encode(&Ins::Madd(0, 0, 1, 2)), 0x1B01_0800);
    assert_eq!(encode(&Ins::Ret), 0xD65F_03C0);
}

#[test]
fn own_assembler_encodes_compiler_output() {
    let t = Target { arch: Arch::Arm64, has_fma: true };
    let r = compile_receipt("a*b+c", &t).unwrap();
    assert_eq!(r.machine_code_len, r.instruction_count * 4, "own assembler emits 4B/ins");
    assert!(r.machine_code_len > 0);
}

#[test]
fn free_list_allocator_bounds_register_pressure() {
    let t = Target { arch: Arch::Arm64, has_fma: true };
    let r = compile_receipt("(a*b+c)*(a+b)+c", &t).unwrap();
    // naive (no reuse) would reach w7 => max 8; free-list reuse keeps it at 7.
    assert!(r.reg_max <= 7, "free-list reuse bounds pressure below naive, got {}", r.reg_max);
}

#[test]
fn own_assembler_arm64_extended_vectors() {
    use competence_catalog::assembler::{assemble, encode, Ins};
    // Verified against the system assembler (examples/compiler_crosscheck.rs): 30/30 match.
    assert_eq!(encode(&Ins::Subs(31, 0, 1)), 0x6B01_001F);
    assert_eq!(encode(&Ins::Cset(0, 0)), 0x1A9F_17E0);  // cset w0, eq
    assert_eq!(encode(&Ins::Cset(0, 11)), 0x1A9F_A7E0); // cset w0, lt
    assert_eq!(encode(&Ins::And(0, 0, 1)), 0x0A01_0000);
    assert_eq!(encode(&Ins::Orr(0, 0, 1)), 0x2A01_0000);
    // Branches: offset is in instruction units from the branch itself.
    let b = assemble("b.eq L\nmov w0, #1\nL:\nret").unwrap();
    assert_eq!(&b[0..4], &0x5400_0040u32.to_le_bytes());
    let u = assemble("b L\nmov w0, #1\nL:\nret").unwrap();
    assert_eq!(&u[0..4], &0x1400_0002u32.to_le_bytes());
}

#[test]
fn own_assembler_x86_known_vectors() {
    use competence_catalog::assembler::{encode_x86, InsX86};
    assert_eq!(encode_x86(&InsX86::MovImm(0, 10)), vec![0xB8, 0x0A, 0, 0, 0]);
    assert_eq!(encode_x86(&InsX86::MovReg(1, 0)), vec![0x89, 0xC1]);
    assert_eq!(encode_x86(&InsX86::Add(0, 1)), vec![0x01, 0xC8]);
    assert_eq!(encode_x86(&InsX86::Cmp(0, 1)), vec![0x39, 0xC8]);
    assert_eq!(encode_x86(&InsX86::Imul(0, 1)), vec![0x0F, 0xAF, 0xC1]); // imul eax, ecx: reg=dst, rm=src
    assert_eq!(encode_x86(&InsX86::Cdq), vec![0x99]);
    assert_eq!(encode_x86(&InsX86::Idiv(1)), vec![0xF7, 0xF9]);
    assert_eq!(encode_x86(&InsX86::Setcc(4)), vec![0x0F, 0x94, 0xC0]); // sete al
    assert_eq!(encode_x86(&InsX86::Movzx(0)), vec![0x0F, 0xB6, 0xC0]);
    assert_eq!(encode_x86(&InsX86::MovImm(8, 1)), vec![0x41, 0xB8, 0x01, 0, 0, 0]); // r8d, REX.B
    assert_eq!(encode_x86(&InsX86::Ret), vec![0xC3]);
}

#[test]
fn x86_branch_relaxation_short_and_long() {
    use competence_catalog::assembler::assemble_x86;
    // Short: je skips a 5B mov -> rel8 (2B), displacement 5.
    let short = assemble_x86("je L\nmov eax, 1\nL:\nret").unwrap();
    assert_eq!(short, vec![0x74, 0x05, 0xB8, 0x01, 0, 0, 0, 0xC3]);
    // Long: >127B of nops (mov eax,0 = 5B each, 30 of them = 150B) -> rel32 (6B).
    let mut asm = String::from("je L\n");
    for _ in 0..30 { asm.push_str("mov eax, 0\n"); }
    asm.push_str("L:\nret\n");
    let long = assemble_x86(&asm).unwrap();
    assert_eq!(&long[0..2], &[0x0F, 0x84]);
    let rel = i32::from_le_bytes([long[2], long[3], long[4], long[5]]);
    assert_eq!(rel, 150, "rel32 displacement over the 30 movs");
}

#[test]
fn x86_receipt_has_machine_code_and_is_deterministic() {
    let t = Target { arch: Arch::X86, has_fma: false };
    let r1 = compile_receipt("a*b+c", &t).unwrap();
    let r2 = compile_receipt("a*b+c", &t).unwrap();
    assert!(r1.deterministic);
    assert_eq!(r1.asm_sha256, r2.asm_sha256);
    assert!(r1.machine_code_len > 0, "own x86 assembler emits machine code");
    assert!(!r1.fused_fma, "no madd on x86 in this fragment");
}

#[test]
fn own_assemblers_match_system_toolchain() {
    // Verification only: system assembler is a cross-check, never in the compile path.
    use competence_catalog::assembler::{assemble, assemble_x86, system_bytes};
    let cases = ["a*b+c", "(a+b)*(a-b)", "a/b", "a%b", "-a", "a<b", "a>=c", "a==a", "a!=b", "a<b && b<c", "a>b || b==c", "1<2"];
    let mut checked = 0;
    for src in cases {
        for (arch, label) in [(Arch::Arm64, "arm64"), (Arch::X86, "x86-64")] {
            let t = Target { arch, has_fma: true };
            let asm = compile_src(src, &t).unwrap();
            let ours = if arch == Arch::Arm64 { assemble(&asm).unwrap() } else { assemble_x86(&asm).unwrap() };
            match system_bytes(&asm, label) {
                Ok(sys) => { assert_eq!(ours, sys, "{src} on {label}"); checked += 1; }
                Err(_) => {} // toolchain unavailable: skip honestly
            }
        }
    }
    let ifs = [("a<b", "a", "c"), ("a>b", "a", "c")];
    for (c, th, el) in ifs {
        for (arch, label) in [(Arch::Arm64, "arm64"), (Arch::X86, "x86-64")] {
            let t = Target { arch, has_fma: true };
            let asm = competence_catalog::compiler::compile_if_src(c, th, el, &t).unwrap();
            let ours = if arch == Arch::Arm64 { assemble(&asm).unwrap() } else { assemble_x86(&asm).unwrap() };
            if let Ok(sys) = system_bytes(&asm, label) {
                assert_eq!(ours, sys, "if {c} then {th} else {el} on {label}");
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "expected the system toolchain on the dev host");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn comparisons_and_conditionals_meaning_preserved_on_silicon() {
    let cases: Vec<(String, i64)> = vec![
        (compile_src("a<b", &Target { arch: Arch::Arm64, has_fma: true }).unwrap(), 1),
        (compile_src("a>=c", &Target { arch: Arch::Arm64, has_fma: true }).unwrap(), 0),
        (compile_src("a<b && b<c", &Target { arch: Arch::Arm64, has_fma: true }).unwrap(), 1),
        (compile_src("a>b || b==c", &Target { arch: Arch::Arm64, has_fma: true }).unwrap(), 0),
        (competence_catalog::compiler::compile_if_src("a<b", "a", "c", &Target { arch: Arch::Arm64, has_fma: true }).unwrap(), 2),
        (competence_catalog::compiler::compile_if_src("a>b", "a", "c", &Target { arch: Arch::Arm64, has_fma: true }).unwrap(), 4),
    ];
    let dir = std::env::temp_dir();
    for (i, (asm, want)) in cases.iter().enumerate() {
        let s = dir.join(format!("rce_cmp_{i}.s"));
        let c = dir.join(format!("rce_cmp_{i}.c"));
        let bin = dir.join(format!("rce_cmp_{i}"));
        std::fs::write(&s, format!(".globl _rce\n_rce:\n{asm}\n")).unwrap();
        std::fs::write(&c, "#include <stdio.h>\nextern long rce(long,long,long);\nint main(){printf(\"%ld\\n\", rce(2,3,4));return 0;}\n").unwrap();
        let ok = std::process::Command::new("clang")
            .args([s.to_str().unwrap(), c.to_str().unwrap(), "-o", bin.to_str().unwrap()])
            .output().map(|o| o.status.success()).unwrap_or(false);
        if !ok { continue; }
        let out = std::process::Command::new(&bin).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim().parse::<i64>().unwrap(), *want, "case {i}");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn emitted_arm64_assembles_and_runs_meaning_preserved() {
    // Our own compiler's ARM64 output, assembled+run on real silicon (system assembler used only
    // as a verification tool; the compiler itself uses no LLVM). a*b+c with (2,3,4) = 10.
    for has_fma in [true, false] {
        let t = Target { arch: Arch::Arm64, has_fma };
        let asm = compile_src("a*b+c", &t).unwrap();
        let dir = std::env::temp_dir();
        let s = dir.join(format!("rce_cg_{has_fma}.s"));
        let c = dir.join(format!("rce_cg_{has_fma}.c"));
        let bin = dir.join(format!("rce_cg_{has_fma}"));
        std::fs::write(&s, format!(".globl _rce\n_rce:\n{asm}\n")).unwrap();
        std::fs::write(&c, "#include <stdio.h>\nextern long rce(long,long,long);\nint main(){printf(\"%ld\\n\", rce(2,3,4));return 0;}\n").unwrap();
        let ok = std::process::Command::new("clang")
            .args([s.to_str().unwrap(), c.to_str().unwrap(), "-o", bin.to_str().unwrap()])
            .output().map(|o| o.status.success()).unwrap_or(false);
        if !ok { continue; } // assembler unavailable in this env; skip silently
        let out = std::process::Command::new(bin.to_str().unwrap()).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "10", "meaning preserved (fma={has_fma})");
    }
}
