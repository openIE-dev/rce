//! E4 kernel: the number of primes below n (n at most 8192), by the sieve of Eratosthenes.
//! A composite is marked with this call's stamp, so the array needs no clearing between calls.
#![cfg_attr(target_arch = "wasm32", no_std)]

const CAP: u32 = 8192;
static mut MARK: [u32; CAP as usize] = [0; CAP as usize];
static mut STAMP: u32 = 0;

#[no_mangle]
pub extern "C" fn run(n: u32) -> u32 {
    let n = if n > CAP { CAP } else { n };
    unsafe {
        STAMP = STAMP.wrapping_add(1);
        let s = STAMP;
        let mut count: u32 = 0;
        let mut i: u32 = 2;
        while i < n {
            if MARK[i as usize] != s {
                count += 1;
                let mut j = i * i;
                while j < n {
                    MARK[j as usize] = s;
                    j += i;
                }
            }
            i += 1;
        }
        count
    }
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
