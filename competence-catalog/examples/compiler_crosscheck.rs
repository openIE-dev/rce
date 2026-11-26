//! Sovereign-compiler cross-check: own assemblers (ARM64 + x86-64) vs the system toolchain
//! (verification tool only), plus meaning-preserved runs on real ARM64 silicon.

use competence_catalog::assembler::{assemble, assemble_x86, system_bytes};
use competence_catalog::compiler::{compile, Arch, Target};
use competence_catalog::lift::lift_expr;
use rce_spec_check::ast::Expr;

fn bin(op: &str, l: Expr, r: Expr) -> Expr {
    Expr::Bin(op.into(), Box::new(l), Box::new(r), 0)
}

fn main() {
    let n = |s: &str| lift_expr(s).unwrap();
    let a = || Expr::Name("a".into(), 0);
    let b = || Expr::Name("b".into(), 0);
    let c = || Expr::Name("c".into(), 0);

    let cases: Vec<(String, Expr)> = vec![
        ("a*b+c".into(), n("a*b+c")),
        ("(a+b)*(a-b)".into(), n("(a+b)*(a-b)")),
        ("a/b".into(), n("a/b")),
        ("a%b".into(), n("a%b")),
        ("-a".into(), n("-a")),
        ("a<b".into(), n("a<b")),
        ("a>=c".into(), n("a>=c")),
        ("a==a".into(), n("a==a")),
        ("a!=b".into(), n("a!=b")),
        ("a<b && b<c".into(), n("a<b && b<c")),
        ("a>b || b==c".into(), n("a>b || b==c")),
        ("1<2 (folds to Bool)".into(), n("1<2")),
        ("not a".into(), Expr::Not(Box::new(a()), 0)),
        ("if a<b then a else c".into(), Expr::If(Box::new(bin("<", a(), b())), Box::new(a()), Box::new(c()), 0)),
        ("if a>b then a else c".into(), Expr::If(Box::new(bin(">", a(), b())), Box::new(a()), Box::new(c()), 0)),
    ];

    let mut checked = 0;
    let mut matched = 0;
    let mut skipped = 0;
    println!("== own assembler vs system toolchain (verification only) ==");
    for (name, e) in &cases {
        for (arch, label) in [(Arch::Arm64, "arm64"), (Arch::X86, "x86-64")] {
            let t = Target { arch, has_fma: true };
            let asm = compile(e, &t).unwrap();
            let ours = if arch == Arch::Arm64 { assemble(&asm).unwrap() } else { assemble_x86(&asm).unwrap() };
            match system_bytes(&asm, label) {
                Ok(sys) => {
                    checked += 1;
                    let ok = sys == ours;
                    if ok { matched += 1; }
                    println!("{:<26} {:<7} {} ({} bytes)", name, label, if ok { "MATCH" } else { "MISMATCH" }, ours.len());
                    if !ok {
                        println!("  ours: {:02x?}", ours);
                        println!("  sys : {:02x?}", sys);
                    }
                }
                Err(_) => {
                    skipped += 1;
                    println!("{:<26} {:<7} SKIP (no system toolchain)", name, label);
                }
            }
        }
    }
    println!("cross-check: {matched}/{checked} match, {skipped} skipped");

    println!("\n== meaning preserved on real ARM64 silicon (a=2,b=3,c=4) ==");
    let expect: Vec<(&str, Expr, i64)> = vec![
        ("a*b+c", n("a*b+c"), 10),
        ("a/b", n("a/b"), 0),
        ("a%b", n("a%b"), 2),
        ("a<b", n("a<b"), 1),
        ("a>=c", n("a>=c"), 0),
        ("a==a", n("a==a"), 1),
        ("a!=b", n("a!=b"), 1),
        ("a<b && b<c", n("a<b && b<c"), 1),
        ("a>b || b==c", n("a>b || b==c"), 0),
        ("1<2", n("1<2"), 1),
        ("not a", Expr::Not(Box::new(a()), 0), 0),
        ("if a<b then a else c", Expr::If(Box::new(bin("<", a(), b())), Box::new(a()), Box::new(c()), 0), 2),
        ("if a>b then a else c", Expr::If(Box::new(bin(">", a(), b())), Box::new(a()), Box::new(c()), 0), 4),
    ];
    let dir = std::env::temp_dir();
    let mut ran = 0;
    let mut correct = 0;
    for (i, (name, e, want)) in expect.iter().enumerate() {
        for has_fma in [true, false] {
            let t = Target { arch: Arch::Arm64, has_fma };
            let asm = compile(e, &t).unwrap();
            let s = dir.join(format!("rce_mean_{i}_{has_fma}.s"));
            let cc = dir.join(format!("rce_mean_{i}_{has_fma}.c"));
            let binp = dir.join(format!("rce_mean_{i}_{has_fma}"));
            std::fs::write(&s, format!(".globl _rce\n_rce:\n{asm}\n")).unwrap();
            std::fs::write(&cc, "#include <stdio.h>\nextern long rce(long,long,long);\nint main(){printf(\"%ld\\n\", rce(2,3,4));return 0;}\n").unwrap();
            let ok = std::process::Command::new("clang")
                .args([s.to_str().unwrap(), cc.to_str().unwrap(), "-o", binp.to_str().unwrap()])
                .output().map(|o| o.status.success()).unwrap_or(false);
            if !ok { println!("{name} (fma={has_fma}): SKIP (no clang)"); continue; }
            let out = std::process::Command::new(&binp).output().unwrap();
            let got: i64 = String::from_utf8_lossy(&out.stdout).trim().parse().unwrap_or(-999);
            ran += 1;
            if got == *want { correct += 1; }
            println!("{:<26} fma={:<5} got {:>3} want {:>3} {}", name, has_fma, got, want, if got == *want { "OK" } else { "FAIL" });
        }
    }
    println!("meaning: {correct}/{ran} correct on silicon");
}