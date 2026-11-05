//! The BLAKE3 digest of a file, in lowercase hexadecimal (used by
//! `scripts/build-wasm-kernels.sh` to pin the kernel library).
fn main() {
    let p = std::env::args().nth(1).expect("FILE");
    let b = std::fs::read(&p).expect("read FILE");
    println!("{}", blake3::hash(&b).to_hex());
}
