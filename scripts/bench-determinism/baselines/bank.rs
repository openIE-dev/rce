//! Hand-written baseline of spec/examples/04_bank_account.rce.
#[path = "common.rs"]
mod common;
use std::hint::black_box;

enum E { Deposit(i64), Withdraw(i64) }
struct Withdrawn { amount: i64 }

fn step(balance: &mut i64, e: &E) -> Result<Option<Withdrawn>, i64> {
    match *e {
        E::Deposit(a) if a > 0 => { *balance += a; Ok(None) }
        E::Deposit(_) => Ok(None),
        E::Withdraw(a) if a > *balance => Err(a - *balance),
        E::Withdraw(a) => { *balance -= a; Ok(Some(Withdrawn { amount: a })) }
    }
}

fn ev_pos(e: &E) -> String {
    match e { E::Deposit(a) => format!("{{\"payload\":[{a}],\"type\":0}}"), E::Withdraw(a) => format!("{{\"payload\":[{a}],\"type\":1}}") }
}

fn main() {
    let (evs, reps) = common::input();
    let evs: Vec<E> = evs.iter().map(|e| match e.name.as_str() {
        "Deposit" => E::Deposit(e.int("amount")),
        _ => E::Withdraw(e.int("amount")),
    }).collect();
    let (mut last, mut head) = (0, String::new());
    for _ in 0..reps {
        let mut ch = common::Chain::new();
        let mut balance = 0i64;
        for e in black_box(&evs) {
            let r = step(&mut balance, e);
            black_box(r.map(|o| o.map(|w| w.amount)).ok());
            if cfg!(bench_log) { ch.add(&ev_pos(e), &format!("[[{balance}]]")); }
        }
        last = balance;
        head = ch.prev;
    }
    if cfg!(bench_log) { println!("head {head}"); } else { println!("state [[{last}]]"); }
}
