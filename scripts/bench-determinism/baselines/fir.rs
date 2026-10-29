//! Hand-written baseline of scripts/bench-determinism/programs/fir_filter.rce.
//! `--cfg bench_fma` uses fused multiply-adds where the RCE program
//! multiplies then adds.
#[path = "common.rs"]
mod common;
use std::hint::black_box;

const TAPS: usize = 16;
struct Filter { xs: [f64; TAPS], head: usize, y: f64, rms: f64 }

fn sqrt_newton(x: f64) -> f64 {
    let mut g = x;
    for _ in 0..30 {
        g = 0.5 * (g + x / g);
    }
    g
}

fn filter(xs: &[f64; TAPS], head: usize) -> f64 {
    let mut s = 0.0f64;
    for i in 0..TAPS {
        #[cfg(bench_fma)]
        { s = 0.0625f64.mul_add(xs[(head + i) % TAPS], s); }
        #[cfg(not(bench_fma))]
        { s += 0.0625 * xs[(head + i) % TAPS]; }
    }
    s
}

fn step(f: &mut Filter, x: f64) {
    f.xs[f.head] = x;
    f.head = (f.head + 1) % TAPS;
    f.y = filter(&f.xs, f.head);
    #[cfg(bench_fma)]
    let a = (0.5 * f.rms).mul_add(f.rms, (0.5 * f.y).mul_add(f.y, 1.0));
    #[cfg(not(bench_fma))]
    let a = 0.5 * f.rms * f.rms + 0.5 * f.y * f.y + 1.0;
    f.rms = sqrt_newton(a);
}

fn pos(f: &Filter) -> String {
    let xs: Vec<String> = f.xs.iter().map(|x| common::fj(*x)).collect();
    format!("[[[{}],{},{},{}]]", xs.join(","), f.head, common::fj(f.y), common::fj(f.rms))
}

fn main() {
    let (evs, reps) = common::input();
    let evs: Vec<f64> = evs.iter().map(|e| e.real("x")).collect();
    let zero = || Filter { xs: [0.0; TAPS], head: 0, y: 0.0, rms: 0.0 };
    let (mut last, mut head) = (zero(), String::new());
    for _ in 0..reps {
        let mut ch = common::Chain::new();
        let mut f = zero();
        for x in black_box(&evs) {
            step(&mut f, *x);
            if cfg!(bench_log) { ch.add(&format!("{{\"payload\":[{}],\"type\":0}}", common::fj(*x)), &pos(&f)); }
        }
        last = f;
        head = ch.prev;
    }
    if cfg!(bench_log) { println!("head {head}"); } else { println!("state {}", pos(&last)); }
}
