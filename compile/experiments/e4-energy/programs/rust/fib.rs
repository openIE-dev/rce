//! E4 kernel: the n-th Fibonacci number modulo 2^32 (n at most 30), by naive recursion.
#![cfg_attr(target_arch = "wasm32", no_std)]

fn fib(n: u32) -> u32 {
    if n < 2 { n } else { fib(n - 1).wrapping_add(fib(n - 2)) }
}

#[no_mangle]
pub extern "C" fn run(n: u32) -> u32 {
    fib(if n > 30 { 30 } else { n })
}

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

// Native driver: calls run(n) R times and prints run(n) and a checksum of all R calls.
#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 3 {
        eprintln!("usage: PROGRAM N R");
        std::process::exit(2);
    }
    let n: u32 = a[1].parse().expect("N");
    let r: u64 = a[2].parse().expect("R");
    let first = run(std::hint::black_box(n));
    let mut x = 0u32;
    for i in 0..r {
        x ^= run(std::hint::black_box(n)).wrapping_add(i as u32);
    }
    println!("{} {}", first, x);
}
