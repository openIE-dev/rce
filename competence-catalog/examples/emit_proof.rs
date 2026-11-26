//! Emit one Form to several real languages and write runnable programs, so the emission
//! can be checked by the actual toolchains (clang / rustc / python3), not by assertion.

fn main() {
    let src = "2+3*4";
    let lut = |n: &str| competence_catalog::lut_emit::builtin_luts().into_iter().find(|l| l.name == n).unwrap();
    let c = competence_catalog::lut_emit::emit_expr(src, &lut("c")).unwrap();
    let rs = competence_catalog::lut_emit::emit_expr(src, &lut("rust")).unwrap();
    let py = competence_catalog::lut_emit::emit_expr(src, &lut("python")).unwrap();
    std::fs::write("/tmp/proof.c", format!("#include <stdio.h>\nint main(void){{printf(\"%lld\\n\",(long long)({c}));return 0;}}\n")).unwrap();
    std::fs::write("/tmp/proof.rs", format!("fn main(){{println!(\"{{}}\",{rs});}}\n")).unwrap();
    std::fs::write("/tmp/proof.py", format!("print({py})\n")).unwrap();
    println!("c: {c}");
    println!("rust: {rs}");
    println!("python: {py}");
}
