// Link Metal + Foundation on macOS/iOS. We use the `metal` crate for
// MTLDevice / MTLLibrary / MTLComputePipelineState etc., which already
// brings the framework link, but we keep an explicit pass here for
// the cfg flag so non-Metal builds compile cleanly with stubs.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(target_os.as_str(), "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-cfg=rce_metal");
    }
    println!("cargo:rustc-check-cfg=cfg(rce_metal)");
}
