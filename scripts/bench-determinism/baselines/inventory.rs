//! Hand-written baseline of spec/examples/31_inventory.rce.
#[path = "common.rs"]
mod common;
use std::collections::HashMap;
use std::hint::black_box;

enum E { Receive(String, u32), Ship(String, u32) }
struct Shipped<'a> { sku: &'a str, qty: u32 }

fn step<'a>(stock: &mut HashMap<String, u32>, e: &'a E) -> Result<Option<Shipped<'a>>, &'a str> {
    match e {
        E::Receive(s, q) => { *stock.entry(s.clone()).or_insert(0) += q; Ok(None) }
        E::Ship(s, q) => {
            let have = stock.get(s).copied().unwrap_or(0);
            if have < *q { return Err(s); }
            stock.insert(s.clone(), have - q);
            Ok(Some(Shipped { sku: s, qty: *q }))
        }
    }
}

/// positional JSON of the state: a Map is its [key, value] pairs in
/// canonical key order
fn pos(stock: &HashMap<String, u32>) -> String {
    let mut ks: Vec<(&String, &u32)> = stock.iter().collect();
    ks.sort_by(|a, b| a.0.chars().cmp(b.0.chars()));
    let l: Vec<String> = ks.iter().map(|(k, v)| format!("[{},{v}]", common::js(k))).collect();
    format!("[[[{}]]]", l.join(","))
}

fn ev_pos(e: &E) -> String {
    match e {
        E::Receive(s, q) => format!("{{\"payload\":[{},{q}],\"type\":0}}", common::js(s)),
        E::Ship(s, q) => format!("{{\"payload\":[{},{q}],\"type\":1}}", common::js(s)),
    }
}

fn main() {
    let (evs, reps) = common::input();
    let evs: Vec<E> = evs.iter().map(|e| match e.name.as_str() {
        "Receive" => E::Receive(e.text("sku"), e.int("qty") as u32),
        _ => E::Ship(e.text("sku"), e.int("qty") as u32),
    }).collect();
    let (mut last, mut head) = (HashMap::new(), String::new());
    for _ in 0..reps {
        let mut ch = common::Chain::new();
        let mut stock: HashMap<String, u32> = HashMap::new();
        for e in black_box(&evs) {
            let r = step(&mut stock, e);
            black_box(r.ok().flatten().map(|s| (s.sku.len(), s.qty)));
            if cfg!(bench_log) { ch.add(&ev_pos(e), &pos(&stock)); }
        }
        last = stock;
        head = ch.prev;
    }
    if cfg!(bench_log) { println!("head {head}"); } else { println!("state {}", pos(&last)); }
}
