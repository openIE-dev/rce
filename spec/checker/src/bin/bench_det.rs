//! `rce-bench-det`: a benchmark-only binary for the cost of the log D
//! (scripts/bench-determinism.py, critique 7). It is not part of `rce` and
//! adds no command or flag to it; it changes no semantics. Every mode
//! applies the same events to the same states with the same δ, so the
//! states are equal; the modes differ only in what each step also writes.
//!
//!   rce-bench-det filter FILE CANDIDATES N
//!       apply the candidate events (one per line, example syntax) in order
//!       and print the first N whose step applies (a kept step is skipped,
//!       so the printed stream has no kept step); the stream is a pure
//!       function of FILE and CANDIDATES
//!   rce-bench-det emit FILE STREAM
//!       the emitted Rust (10.5, as `rce emit rust`) of FILE with STREAM as
//!       its only example (the module's own examples are left out): FILE
//!       is admitted first; the stream's expectation is never checked, as
//!       for `rce trace streams`
//!   rce-bench-det run MODE FILE STREAM REPS
//!       apply the events of STREAM (one per line) REPS times from s0 and
//!       print `steps K check C` (C: a value derived from the last state,
//!       printed so no work is dead). MODE:
//!         delta   δ only: build the event, run δ (the floor; no log)
//!         encode  delta, plus the canonical JSON of the entry (readable
//!                 event, positional event and positional state, 6.1, T7)
//!                 and its stored line, without hashing
//!         chain   encode, plus the two SHA-256 of every entry: state_hash
//!                 and hash over {event, prev_hash, seq, state_hash}; the
//!                 entry chain of `rce trace` and of the host (D.6); C is
//!                 the head hash of the last run, equal to the `trace`
//!                 line of the emitted program under the word trace
//!         chainfast  chain with the same bytes and hashes, the SHA-256
//!                 hex written from a table instead of one `format!` per
//!                 byte (`d_sha`): a semantics-preserving speed-up, priced
//!         ckpt64  chain with the state hash computed only every 64th step
//!                 (other entries hash the event and prev_hash only): a
//!                 different D, measured to price the proposal HS-12
//!                 direction; not the D of the specification
//!         host    the host runtime (Appendix E) on an in-memory store:
//!                 one batch of STREAM per run (apply_input per event, then
//!                 settle), as `rce host run FILE D EVENTS` does,
//!                 without the file and its fdatasync
//!         host1   the host runtime with one batch (and its barrier
//!                 entries) per event, as `rce host serve` does per line
use rce_spec_check::ast::{Arg, Item};
use rce_spec_check::host::{Host, MemStore};
use rce_spec_check::interp::{D_GENESIS, Interp, d_sha, event_json, event_pos_json, state_pos_json, why};
use std::collections::{HashMap, HashSet};

fn events_of(src: &str, lines: &[String]) -> Result<Vec<(String, Vec<Arg>)>, String> {
    let probe = rce_spec_check::parse::parse(&format!("{src}\nexample {} -> 0\n", lines.join(", "))).map_err(|e| format!("events: {e}"))?;
    match probe.items.last() {
        Some(Item::Example(x)) => Ok(x.events.clone()),
        _ => Err("events: not an event list".into()),
    }
}

/// SHA-256 as lower-case hex, equal to `d_sha`, hex from a table.
fn fast_sha(s: &str) -> String {
    use sha2::{Digest, Sha256};
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let d = Sha256::digest(s.as_bytes());
    let mut o = Vec::with_capacity(64);
    for b in d.iter() {
        o.push(HEX[(b >> 4) as usize]);
        o.push(HEX[(b & 15) as usize]);
    }
    String::from_utf8(o).expect("hex")
}

fn read_lines(p: &str) -> Vec<String> {
    std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("{p}: {e}"))
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(String::from)
        .collect()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let src = std::fs::read_to_string(&a[if a[0] == "run" { 2 } else { 1 }]).expect("FILE");
    let d = rce_spec_check::cli::diagnostics(&src, &HashSet::new());
    if !d.is_empty() {
        eprintln!("requires change: {}", d.into_iter().map(|x| x.1).collect::<Vec<_>>().join("; "));
        std::process::exit(2);
    }
    let m = rce_spec_check::program(&src).expect("program");
    match a[0].as_str() {
        "filter" => {
            let cands = read_lines(&a[2]);
            let n: usize = a[3].parse().expect("N");
            let evs = events_of(&src, &cands).expect("candidates");
            let mut it = Interp::new(&m);
            let mut s = it.initial_state().map_err(why).expect("s0");
            let (mut kept, mut out) = (0usize, 0usize);
            for (k, (name, args)) in evs.iter().enumerate() {
                if out == n {
                    break;
                }
                it.scopes = vec![HashMap::new()];
                it.refuel();
                let Ok(Some(ev)) = it.construct(name, args) else {
                    kept += 1;
                    continue;
                };
                match it.delta(&s, &ev) {
                    Ok((s2, _)) => {
                        s = s2;
                        println!("{}", cands[k]);
                        out += 1;
                    }
                    Err(_) => kept += 1,
                }
            }
            eprintln!("filter: {out} applied, {kept} skipped");
            if out < n {
                std::process::exit(1);
            }
        }
        "emit" => {
            let lines = read_lines(&a[2]);
            let mut probe = rce_spec_check::parse::parse(&format!("{src}\nexample {} -> 0\n", lines.join(", "))).expect("stream");
            // the stream is the only example the emitted program runs
            let stream = probe.items.pop().expect("stream");
            probe.items.retain(|i| !matches!(i, Item::Example(_)));
            for i in probe.items.iter_mut() {
                if let Item::Intent(_, xs, _) = i {
                    xs.clear();
                }
            }
            probe.items.push(stream);
            print!("{}", rce_spec_check::emit::emit(&probe, rce_spec_check::emit::Lang::Rust).expect("emit"));
        }
        "run" => {
            let mode = a[1].as_str();
            let lines = read_lines(&a[3]);
            let reps: usize = a[4].parse().expect("REPS");
            let evs = events_of(&src, &lines).expect("stream");
            let mut steps = 0usize;
            let mut check = String::new();
            if mode == "host" || mode == "host1" {
                for _ in 0..reps {
                    let mut h = Host::start(&m, Box::new(MemStore::default())).expect("host");
                    for (name, args) in &evs {
                        if mode == "host" {
                            h.apply_input(name, args.clone()).expect("input");
                        } else {
                            h.admit_event(name, args.clone()).expect("input");
                        }
                        steps += 1;
                    }
                    h.settle().expect("settle");
                    check = format!("{}", h.queued());
                }
            } else {
                let mut it = Interp::new(&m);
                let s0 = it.initial_state().map_err(why).expect("s0");
                let mut store: Vec<String> = Vec::with_capacity(evs.len());
                for _ in 0..reps {
                    store.clear();
                    let mut s = s0.clone();
                    let mut prev = D_GENESIS.to_string();
                    let mut bytes = 0usize;
                    for (seq, (name, args)) in evs.iter().enumerate() {
                        it.scopes = vec![HashMap::new()];
                        it.refuel();
                        let ev = match it.construct(name, args) {
                            Ok(Some(v)) => v,
                            _ => panic!("event {name}"),
                        };
                        if let Ok((s2, _outs)) = it.delta(&s, &ev) {
                            s = s2;
                        }
                        steps += 1;
                        if mode == "delta" {
                            continue;
                        }
                        let ej = event_json(name, &ev, &m);
                        let evp = event_pos_json(name, &ev, &m);
                        let stp = state_pos_json(&s, &m);
                        let (sh, h) = match mode {
                            "encode" => (String::new(), String::new()),
                            "chainfast" => {
                                let sh = fast_sha(&stp);
                                let body = format!("{{\"event\":{evp},\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state_hash\":\"{sh}\"}}");
                                let h = fast_sha(&body);
                                (sh, h)
                            }
                            "ckpt64" => {
                                let sh = if seq % 64 == 63 { d_sha(&stp) } else { String::new() };
                                let body = format!("{{\"event\":{evp},\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state_hash\":\"{sh}\"}}");
                                let h = d_sha(&body);
                                (sh, h)
                            }
                            _ => {
                                let sh = d_sha(&stp);
                                let body = format!("{{\"event\":{evp},\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state_hash\":\"{sh}\"}}");
                                let h = d_sha(&body);
                                (sh, h)
                            }
                        };
                        let line = format!("{{\"event\":{ej},\"hash\":\"{h}\",\"prev_hash\":\"{prev}\",\"seq\":{seq},\"state_hash\":\"{sh}\"}}");
                        bytes += line.len() + stp.len();
                        store.push(line);
                        if mode != "encode" {
                            prev = h;
                        }
                    }
                    check = match mode {
                        "delta" => d_sha(&state_pos_json(&s, &m)),
                        "encode" => format!("{bytes}"),
                        _ => prev,
                    };
                }
            }
            println!("steps {steps} check {check}");
        }
        _ => {
            eprintln!("usage: rce-bench-det filter FILE CANDIDATES N | emit FILE STREAM | run MODE FILE STREAM REPS");
            std::process::exit(2);
        }
    }
}
