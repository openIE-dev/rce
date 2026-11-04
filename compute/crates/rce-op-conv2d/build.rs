fn main() {
    // On macOS, link Apple's Accelerate.framework for cblas_sgemm (~1.5 TFLOPS
    // single-precision GEMM via Apple's hand-tuned kernels — orders of magnitude
    // faster than a pure-Rust matmul for tall/skinny conv im2col shapes).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
