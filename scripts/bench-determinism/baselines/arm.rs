//! Hand-written baseline of programs/body/arm.rce. `--cfg bench_fma` uses a
//! fused multiply-add where the RCE program multiplies then adds.
#[path = "common.rs"]
mod common;
use std::hint::black_box;

struct Arm { angle: f64, target: f64 }

fn step(s: &mut Arm, r: f64) -> Result<(), ()> {
    if r < -3.2 || r > 3.2 { return Err(()); }
    #[cfg(bench_fma)]
    { s.angle = 0.5f64.mul_add(s.target - r, s.angle); }
    #[cfg(not(bench_fma))]
    { s.angle += 0.5 * (s.target - r); }
    Ok(())
}

fn pos(s: &Arm) -> String {
    format!("[[{},{}]]", common::fj(s.angle), common::fj(s.target))
}

fn main() {
    let (evs, reps) = common::input();
    let evs: Vec<f64> = evs.iter().map(|e| e.real("reading")).collect();
    let (mut last, mut head) = (Arm { angle: 0.0, target: 1.0 }, String::new());
    for _ in 0..reps {
        let mut ch = common::Chain::new();
        let mut s = Arm { angle: 0.0, target: 1.0 };
        for r in black_box(&evs) {
            black_box(step(&mut s, *r)).ok();
            if cfg!(bench_log) { ch.add(&format!("{{\"payload\":[{}],\"type\":0}}", common::fj(*r)), &pos(&s)); }
        }
        last = s;
        head = ch.prev;
    }
    if cfg!(bench_log) { println!("head {head}"); } else { println!("state {}", pos(&last)); }
}
