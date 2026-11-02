//! E4 kernel: the product of two n-by-n integer matrices (n at most 24), returning the sum of
//! each entry times (row + column + 1), modulo 2^32.
#![cfg_attr(target_arch = "wasm32", no_std)]

const CAP: usize = 24;
static mut A: [[u32; CAP]; CAP] = [[0; CAP]; CAP];
static mut B: [[u32; CAP]; CAP] = [[0; CAP]; CAP];
static mut C: [[u32; CAP]; CAP] = [[0; CAP]; CAP];

#[no_mangle]
pub extern "C" fn run(n: u32) -> u32 {
    let n = if n as usize > CAP { CAP } else { n as usize };
    unsafe {
        for i in 0..n {
            for j in 0..n {
                A[i][j] = (i as u32).wrapping_mul(j as u32).wrapping_add(1);
                B[i][j] = (i as u32).wrapping_add(2 * j as u32).wrapping_add(3);
            }
        }
        for i in 0..n {
            for j in 0..n {
                let mut s: u32 = 0;
                for k in 0..n {
                    s = s.wrapping_add(A[i][k].wrapping_mul(B[k][j]));
                }
                C[i][j] = s;
            }
        }
        let mut sum: u32 = 0;
        for i in 0..n {
            for j in 0..n {
                sum = sum.wrapping_add(C[i][j].wrapping_mul((i + j + 1) as u32));
            }
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
