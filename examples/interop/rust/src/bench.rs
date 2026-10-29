//! Per-call cost at the boundary: JSON text to an event value, the step,
//! and the state to JSON text, each timed alone over the stream, n times.
use std::time::Instant;

pub fn run<P, E>(stream: &[serde_json::Value], n: usize, new: impl Fn() -> P, event: impl Fn(&P, &serde_json::Value) -> E,
    step: impl Fn(&mut P, &E), state: impl Fn(&P) -> String) -> String {
    let texts: Vec<String> = stream.iter().map(|e| e.to_string()).collect();
    let (mut dec, mut st, mut enc, mut calls) = (0u128, 0u128, 0u128, 0u128);
    for _ in 0..n {
        let mut p = new();
        for t in &texts {
            let t0 = Instant::now();
            let j: serde_json::Value = serde_json::from_str(t).unwrap();
            let e = event(&p, &j);
            let t1 = Instant::now();
            step(&mut p, &e);
            let t2 = Instant::now();
            let s = state(&p);
            let t3 = Instant::now();
            std::hint::black_box(s);
            dec += (t1 - t0).as_nanos();
            st += (t2 - t1).as_nanos();
            enc += (t3 - t2).as_nanos();
            calls += 1;
        }
    }
    format!("{{\"calls\":{calls},\"decode_ns\":{},\"step_ns\":{},\"encode_ns\":{}}}\n", dec / calls, st / calls, enc / calls)
}
