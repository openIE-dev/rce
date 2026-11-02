//! E4 kernel: total Collatz steps of the starting values 1..n, each capped at 1000 steps.
#![cfg_attr(target_arch = "wasm32", no_std)]

#[no_mangle]
pub extern "C" fn run(n: u32) -> u32 {
    let mut total: u32 = 0;
    let mut v: u32 = 1;
    while v <= n {
        let mut x = v;
        let mut steps: u32 = 0;
        while x != 1 && steps < 1000 {
            x = if x & 1 == 1 { x.wrapping_mul(3).wrapping_add(1) } else { x >> 1 };
            steps += 1;
        }
        total = total.wrapping_add(steps);
        v += 1;
    }
    total
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
