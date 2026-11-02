//! E4 kernel: the total number of set bits in n values from a linear congruential generator,
//! counted by clearing the lowest set bit.
#![cfg_attr(target_arch = "wasm32", no_std)]

#[no_mangle]
pub extern "C" fn run(n: u32) -> u32 {
    let mut x: u32 = 7;
    let mut total: u32 = 0;
    let mut i: u32 = 0;
    while i < n {
        x = x.wrapping_mul(22695477).wrapping_add(1);
        let mut y = x;
        while y != 0 {
            y &= y - 1;
            total = total.wrapping_add(1);
        }
        i += 1;
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
