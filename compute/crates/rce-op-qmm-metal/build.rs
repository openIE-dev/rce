// Link Metal + Foundation on Apple platforms; set the `rce_qmm_metal`
// cfg so the imp module compiles in only there. Cross-target builds get
// a stub returning None from `MetalQmmContext::new`.
fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(target_os.as_str(), "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-cfg=rce_qmm_metal");
    }
    println!("cargo:rustc-check-cfg=cfg(rce_qmm_metal)");
}
