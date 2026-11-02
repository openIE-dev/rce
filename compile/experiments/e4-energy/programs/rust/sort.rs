//! E4 kernel: insertion sort of n (at most 1024) values from a linear congruential generator,
//! returning the sum of value times (position + 1), modulo 2^32.
#![cfg_attr(target_arch = "wasm32", no_std)]

const CAP: u32 = 1024;
static mut A: [u32; CAP as usize] = [0; CAP as usize];

#[no_mangle]
pub extern "C" fn run(n: u32) -> u32 {
    let n = if n > CAP { CAP } else { n } as usize;
    unsafe {
        let mut x: u32 = 12345;
        let mut i = 0;
        while i < n {
            x = x.wrapping_mul(1103515245).wrapping_add(12345);
            A[i] = (x >> 16) & 0x7fff;
            i += 1;
        }
        i = 1;
        while i < n {
            let v = A[i];
            let mut j = i;
            while j > 0 && A[j - 1] > v {
                A[j] = A[j - 1];
                j -= 1;
            }
            A[j] = v;
            i += 1;
        }
        let mut sum: u32 = 0;
        i = 0;
        while i < n {
            sum = sum.wrapping_add(A[i].wrapping_mul(i as u32 + 1));
            i += 1;
        }
        sum
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
