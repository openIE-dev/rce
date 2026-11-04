// Link MetalPerformanceShaders + Metal + Foundation on macOS/iOS.
//
// We do this explicitly (rather than transitively through the `metal`
// crate) because the `metal` crate currently doesn't expose
// MPSMatrixMultiplication, so we issue raw objc messages to the
// MetalPerformanceShaders Objective-C classes — and that needs the
// MPS framework on the link line.

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if matches!(target_os.as_str(), "macos" | "ios") {
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-cfg=rce_mps");
    }
    println!("cargo:rustc-check-cfg=cfg(rce_mps)");
}
