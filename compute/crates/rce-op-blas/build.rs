// Emit per-platform link flags for the platform BLAS:
//   macOS  → Apple Accelerate (system framework, always available)
//   linux  → OpenBLAS when the system has libopenblas in a standard
//            library directory; otherwise no BLAS link, as on other
//            targets (the dispatch returns Unsupported and the CPU
//            backend's own matmul runs)
//   other  → no BLAS link; runtime returns Unsupported and the cpu
//            backend's naive matmul handles every size

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-cfg=rce_blas_accelerate");
        }
        "linux" => {
            let dirs = [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/aarch64-linux-gnu",
                "/usr/lib64",
                "/usr/lib",
                "/usr/local/lib",
                "/opt/OpenBLAS/lib",
            ];
            let found = dirs.iter().any(|d| {
                ["libopenblas.so", "libopenblas.a"].iter().any(|f| std::path::Path::new(d).join(f).exists())
            });
            if found {
                println!("cargo:rustc-link-lib=openblas");
                println!("cargo:rustc-cfg=rce_blas_cblas");
            }
        }
        _ => {
            // No BLAS — the dispatch returns Unsupported and the existing
            // pure-Rust matmul in rce-vm-backend-cpu handles the workload.
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
    // Tell rustc which cfg keys we may emit so #[cfg] guards don't warn.
    println!("cargo:rustc-check-cfg=cfg(rce_blas_accelerate)");
    println!("cargo:rustc-check-cfg=cfg(rce_blas_cblas)");
}
