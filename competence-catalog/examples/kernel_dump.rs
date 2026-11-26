use competence_catalog::compiler::{Arch, Target};
use competence_catalog::kernel::compile_program;

fn main() {
    let src = "fn add(x, y) { return x + y; } let r = add(a, b); return r * c;";
    for (arch, label) in [(Arch::Arm64, "arm64"), (Arch::X86, "x86-64")] {
        println!("== {label} ==");
        println!("{}", compile_program(src, &Target { arch, has_fma: true }).unwrap());
    }
}