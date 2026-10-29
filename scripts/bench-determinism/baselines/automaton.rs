//! Hand-written baseline of programs/world/cellular_automaton.rce (rule 30
//! on a ring of 8 cells).
#[path = "common.rs"]
mod common;
use std::hint::black_box;

fn next(c: [u8; 8]) -> [u8; 8] {
    let mut o = [0u8; 8];
    for i in 0..8 {
        o[i] = (c[(i + 7) % 8] ^ (c[i] | c[(i + 1) % 8])) & 1;
    }
    o
}

fn pos(c: &[u8; 8]) -> String {
    let l: Vec<String> = c.iter().map(|x| x.to_string()).collect();
    format!("[[[{}]]]", l.join(","))
}

fn main() {
    let (evs, reps) = common::input();
    let n = evs.len();
    let (mut last, mut head) = ([0u8; 8], String::new());
    for _ in 0..reps {
        let mut ch = common::Chain::new();
        let mut cells = [0, 0, 0, 0, 1, 0, 0, 0];
        for _ in 0..black_box(n) {
            cells = next(black_box(cells));
            if cfg!(bench_log) { ch.add("{\"payload\":[],\"type\":0}", &pos(&cells)); }
        }
        last = cells;
        head = ch.prev;
    }
    if cfg!(bench_log) { println!("head {head}"); } else { println!("state {}", pos(&last)); }
}
