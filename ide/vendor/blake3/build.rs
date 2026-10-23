// rce-ide vendored build script: pure Rust only, no C compiler, no `cc`.
//
// Upstream blake3 1.8.7 always lists `cc` as a build dependency, even with
// the `pure` feature, which only stops it from being invoked. This copy keeps
// the upstream library source unchanged and replaces build.rs with the
// `pure` path of the upstream script: on x86/x86_64 it selects the Rust
// intrinsics implementations (SSE2, SSE4.1, AVX2; runtime-detected), never
// the C/assembly ones, and never AVX-512 or the C NEON code. wasm32 SIMD
// stays opt-in through the `wasm32_simd` feature, as upstream.
use std::env;

fn main() {
    for c in [
        "blake3_sse2_ffi",
        "blake3_sse2_rust",
        "blake3_sse41_ffi",
        "blake3_sse41_rust",
        "blake3_avx2_ffi",
        "blake3_avx2_rust",
        "blake3_avx512_ffi",
        "blake3_neon",
        "blake3_wasm32_simd",
    ] {
        println!("cargo::rustc-check-cfg=cfg({c}, values(none()))");
    }
    if env::var_os("CARGO_FEATURE_NEON").is_some() {
        panic!("this vendored blake3 is pure Rust; the C `neon` feature is not available");
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch == "x86_64" || arch == "x86" {
        println!("cargo:rustc-cfg=blake3_sse2_rust");
        println!("cargo:rustc-cfg=blake3_sse41_rust");
        println!("cargo:rustc-cfg=blake3_avx2_rust");
    }
    if arch == "wasm32" && env::var_os("CARGO_FEATURE_WASM32_SIMD").is_some() {
        println!("cargo:rustc-cfg=blake3_wasm32_simd");
    }
    println!("cargo:rerun-if-changed=build.rs");
}
