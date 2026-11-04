// Link CUDA + cuBLAS on Linux when the toolkit is present. The probe
// is a presence check on `libcuda.so` / `libcublas.so` in the
// standard CUDA install prefixes; if either is missing we silently
// skip the linker flags and the crate compiles to a runtime stub
// that always returns Unsupported (so non-Linux / non-CUDA hosts
// build without any toolkit installed).

use std::path::Path;

fn cuda_lib_dir() -> Option<&'static str> {
    // Standard CUDA install prefixes on Linux + a couple of common
    // override paths. Stop at the first one that contains
    // `libcudart.so` (a sentinel — every CUDA install ships it).
    for dir in &[
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/cuda-13/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ] {
        let p = Path::new(dir).join("libcudart.so");
        if p.exists() {
            // Leak the dir as &'static str — tiny, only happens once
            // at build time.
            return Some(Box::leak(dir.to_string().into_boxed_str()));
        }
    }
    None
}

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        if let Some(libdir) = cuda_lib_dir() {
            println!("cargo:rustc-link-search=native={}", libdir);
            // libcuda.so is the driver API, libcublas.so is cuBLAS.
            // Both are runtime-shipped so we link dynamically.
            println!("cargo:rustc-link-lib=dylib=cuda");
            println!("cargo:rustc-link-lib=dylib=cublas");
            println!("cargo:rustc-cfg=rce_cuda");
        }
    }
    println!("cargo:rustc-check-cfg=cfg(rce_cuda)");
}
