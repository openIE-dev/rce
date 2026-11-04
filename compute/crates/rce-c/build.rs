use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    // Generate the header into OUT_DIR for the build, and also into the
    // crate's `include/` directory so it's readily available to consumers
    // checking out the repo (no `cargo install`-only flow).
    let include_dir = PathBuf::from(&crate_dir).join("include");
    let _ = std::fs::create_dir_all(&include_dir);
    let header_path = include_dir.join("rce.h");

    let config = cbindgen::Config::from_file(format!("{crate_dir}/cbindgen.toml")).unwrap();

    if let Ok(b) = cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        b.write_to_file(&header_path);
        b.write_to_file(format!("{out_dir}/rce.h"));
    }

    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
}
