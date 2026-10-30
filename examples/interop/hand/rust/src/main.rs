//! Run a hand-written program on a stream and print what `rce run` prints.
//!   rce-interop-hand drive NAME STREAM.json
//!   rce-interop-hand bench NAME STREAM.json N
mod programs;
use programs::*;
use serde_json::Value as J;
use std::time::Instant;

/// An event as `rce run` displays it, from its JSON object (fields in order).
fn show_event(j: &J) -> String {
    let o = j.as_object().unwrap();
    let fs: Vec<String> = o.iter().filter(|(k, _)| *k != "type").map(|(k, v)| format!("{k}: {}", show_json(v))).collect();
    let n = o["type"].as_str().unwrap();
    if fs.is_empty() { n.to_string() } else { format!("{n}({})", fs.join(", ")) }
}

fn show_json(v: &J) -> String {
    match v {
        J::String(s) => format!("{s:?}"),
        J::Number(n) if n.is_f64() => format!("{}", n.as_f64().unwrap()),
        other => other.to_string(),
    }
}

fn drive<P: Program>(stream: &[J]) -> String {
    let mut p = P::default();
    let mut out = format!("0 (initial) {p}\n");
    for (k, j) in stream.iter().enumerate() {
        let k = k + 1;
        let e: P::Event = serde_json::from_value(j.clone()).expect("an event of the program");
        match p.step(e) {
            Ok(outs) => {
                out += &format!("{k} {} {p}\n", show_event(j));
                for o in outs { out += &format!("  emit {o}\n"); }
            }
            Err(r) => out += &format!("{k} {} kept: {r}\n", show_event(j)),
        }
    }
    out
}

/// The per-call cost of the hand-written version: JSON text to the typed
/// event, the step, and the state to text.
fn bench<P: Program>(stream: &[J], n: usize) -> String {
    let texts: Vec<String> = stream.iter().map(|e| e.to_string()).collect();
    let (mut dec, mut st, mut enc, mut calls) = (0u128, 0u128, 0u128, 0u128);
    for _ in 0..n {
        let mut p = P::default();
        for t in &texts {
            let t0 = Instant::now();
            let e: P::Event = serde_json::from_str(t).unwrap();
            let t1 = Instant::now();
            let _ = std::hint::black_box(p.step(e));
            let t2 = Instant::now();
            std::hint::black_box(p.to_string());
            let t3 = Instant::now();
            dec += (t1 - t0).as_nanos();
            st += (t2 - t1).as_nanos();
            enc += (t3 - t2).as_nanos();
            calls += 1;
        }
    }
    format!("{{\"calls\":{calls},\"decode_ns\":{},\"step_ns\":{},\"encode_ns\":{}}}\n", dec / calls, st / calls, enc / calls)
}

macro_rules! dispatch {
    ($f:ident, $name:expr, $($a:expr),*) => {
        match $name {
            "counter" => $f::<Counter>($($a),*),
            "bank" => $f::<Account>($($a),*),
            "multi_region" => $f::<Shop>($($a),*),
            "traffic_light" => $f::<Signal>($($a),*),
            "door" => $f::<Door>($($a),*),
            "inventory" => $f::<Inventory>($($a),*),
            "rcedb" => $f::<Db>($($a),*),
            "float_keys" => $f::<Tally>($($a),*),
            n => panic!("no program {n}"),
        }
    };
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let stream: Vec<J> = serde_json::from_str(&std::fs::read_to_string(&a[3]).unwrap()).unwrap();
    let out = match a[1].as_str() {
        "drive" => dispatch!(drive, a[2].as_str(), &stream),
        "bench" => dispatch!(bench, a[2].as_str(), &stream, a[4].parse().unwrap()),
        m => panic!("unknown mode {m}"),
    };
    print!("{out}");
}
