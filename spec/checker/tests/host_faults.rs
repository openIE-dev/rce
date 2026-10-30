//! Seeded fault injection at the host boundary (Appendix E). The pure step
//! δ is exact; this harness measures what happens around it. One seed
//! drives a simulated transport and a simulated host process for a program
//! run on the real `Host` (durable D files, the same calls `rce host run`
//! and `rce host serve` make), and injects these faults:
//! - drop: an input copy is lost before it reaches the admitter;
//! - dup: an input copy is delivered twice (a retrying client or proxy);
//! - reorder: copies are delayed by up to a few rounds, so they arrive out
//!   of send order;
//! - lose: the answer to an admitted input is lost, so the client retries;
//! - restart: the host process is killed and D reopened (HC-22): between
//!   rounds or in the middle of a batch before its barrier, with the D file
//!   as a kill -9 leaves it, with a torn last line, or cut back to the last
//!   barrier (a power cut, HS-2);
//! - skew: `host.WallTime` is read from a clock that steps backward and
//!   forward of true time.
//!
//! After each run, from D alone, it checks:
//! - (a) replay with no adapters re-derives the live state and its hash;
//! - (b) every entry is a step or a kept step (T8): a host entry and a kept
//!   entry leave the state hash unchanged, and each kept cause is one of
//!   HC-21's;
//! - (c) a restart never reuses ids (HC-22): each `host.Restarted` has
//!   below equal to its own seq, every net resource id after it is at least
//!   below and none is issued twice, and every HTTP request id is unique;
//! - (d) what the program does with the faults at the application level,
//!   against an oracle of the client's intent (each operation applied
//!   once, in order). These counts are faults the language does not
//!   prevent: a program without an idempotency key applies a duplicate
//!   twice, and D records it so that it can be replayed and found.
//!
//! The programs: `programs/money/{account,ledger}.rce` and
//! `programs/network/handshake.rce` as event streams, admitted as typed
//! lines (the input stream of `rce host run`/`serve`) and as sourced inputs
//! with (source, seq) (HC-15, HC-16); `programs/network/notes_http.rce` on
//! the HTTP event host (HC-24); `tests/host/notes_net.rce` on the net
//! adapter with Tick polling (HC-23); and `tests/host/faults/ratelimit.rce`,
//! a token bucket over `WallTime`.
//!
//! RCE_FAULT_SEEDS (default 3), RCE_FAULT_SEED (one seed), RCE_FAULT_MIXES
//! (comma list of `all`, `none`, or one class: default `all`),
//! RCE_FAULT_PROGRAMS (comma list, default every program), RCE_FAULT_THREADS
//! (default 2), RCE_FAULT_OUT (append one JSON line per run), RCE_FAULT_DIR
//! (where D files go), RCE_FAULT_KEEP (keep them).
use rce_spec_check::ast::{Arg, Item, Module};
use rce_spec_check::host::{Adapter, EffectResult, Entry, Host, MemStore, Sourced, replay_report};
use rce_spec_check::interp::V;
use rce_spec_check::parse::parse;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;

// ---------------------------------------------------------------- seeds

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64, as in tests/dst_chat.rs
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn chance(&mut self, pct: u64) -> bool {
        pct > 0 && self.below(100) < pct
    }
}

/// The fault classes, each a percentage per copy, answer or round.
#[derive(Clone, Copy, Debug, Default)]
struct Mix {
    drop: u64,
    dup: u64,
    /// the most rounds a copy is delayed (reorder)
    delay: u64,
    lose: u64,
    restart: u64,
    skew: u64,
}

fn mix(name: &str) -> Mix {
    let all = Mix { drop: 10, dup: 10, delay: 4, lose: 10, restart: 3, skew: 10 };
    match name {
        "all" => all,
        "none" => Mix::default(),
        "drop" => Mix { drop: all.drop, ..Default::default() },
        "dup" => Mix { dup: all.dup, ..Default::default() },
        "reorder" => Mix { delay: all.delay, ..Default::default() },
        "lose" => Mix { lose: all.lose, ..Default::default() },
        "restart" => Mix { restart: all.restart, ..Default::default() },
        "skew" => Mix { skew: all.skew, ..Default::default() },
        _ => panic!("unknown fault mix {name}"),
    }
}

/// Rounds between a send and its retransmission when no answer came.
const TIMEOUT: u64 = 6;
/// The longest a run may take; a run that does not finish is not live.
const MAX_ROUNDS: u64 = 3000;
/// True time per round.
const ROUND_NS: u64 = 100_000_000;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// One event of example syntax as a name and arguments.
fn event_of(src: &str, line: &str) -> (String, Vec<Arg>) {
    let probe = parse(&format!("{src}\nexample {line} -> 0\n")).unwrap_or_else(|e| panic!("{line}: {e}"));
    match probe.items.last() {
        Some(Item::Example(x)) => x.events[0].clone(),
        _ => panic!("{line}: not an event"),
    }
}

fn field(h: &Host, name: &str) -> V {
    h.state().iter().flat_map(|(_, fs)| fs.iter()).find(|(n, _, _)| n == name).map(|(_, v, _)| v.clone()).unwrap_or_else(|| panic!("no field {name}"))
}

fn as_u64(v: &V) -> u64 {
    match v {
        V::Int(i, _) => *i as u64,
        _ => panic!("{v:?}"),
    }
}

fn text_map(v: &V) -> BTreeMap<String, u64> {
    match v {
        V::Map(kv) => kv.iter().map(|(k, v)| (if let V::Text(t) = k { t.to_string() } else { format!("{k:?}") }, as_u64(v))).collect(),
        _ => panic!("{v:?}"),
    }
}

/// The note texts of a `notes: Map[u64, text]` field, with their counts.
fn note_texts(v: &V) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    if let V::Map(kv) = v {
        for (_, t) in kv {
            if let V::Text(t) = t {
                *out.entry(t.to_string()).or_default() += 1;
            }
        }
    }
    out
}

// ---------------------------------------------------------------- a run

/// What one run did, and what D says about it.
#[derive(Default)]
struct Run {
    rounds: u64,
    live: bool,
    faults: BTreeMap<&'static str, u64>,
    /// (a), (b), (c) violations: each is a host or contract fault
    a: Vec<String>,
    b: Vec<String>,
    c: Vec<String>,
    /// (d): faults the program mishandles at the application level
    d: BTreeMap<&'static str, u64>,
    d_note: Vec<String>,
    /// what the host contract leaves to the client (not a program fault)
    gap: BTreeMap<&'static str, u64>,
    /// (c): net ids checked, and how many of them were issued after a restart
    ids: u64,
    ids_after_restart: u64,
    kept: BTreeMap<String, u64>,
    entries: u64,
    host_dups: u64,
    d_text: String,
}

impl Run {
    fn fault(&mut self, k: &'static str) {
        *self.faults.entry(k).or_default() += 1;
    }
    fn mishandled(&mut self, k: &'static str, note: String) {
        *self.d.entry(k).or_default() += 1;
        if self.d_note.len() < 4 {
            self.d_note.push(note);
        }
    }
}

/// The host process: a `Host` on a D file, killed and reopened.
struct Life<'m> {
    m: &'m Module,
    path: PathBuf,
    h: Option<Host<'m>>,
}

impl<'m> Life<'m> {
    fn h(&mut self) -> &mut Host<'m> {
        self.h.as_mut().unwrap()
    }
    /// Kill the process. D stays as a kill -9 leaves it (every appended
    /// line is in the file), with a torn last line, or cut back to its last
    /// barrier (a power cut: what was not durable is gone). Returns the
    /// highest seq left in D.
    fn kill(&mut self, rng: &mut Rng, run: &mut Run) -> Option<u64> {
        let h = self.h.take().unwrap();
        let durable = h.durable;
        drop(h);
        run.fault("restart");
        match rng.below(5) {
            0 => {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new().append(true).open(&self.path).unwrap();
                f.write_all(b"{\"event\":{\"payload\":{\"torn").unwrap();
                run.fault("restart_torn");
            }
            1 => {
                let t = std::fs::read_to_string(&self.path).unwrap();
                let keep: String = t.lines().filter(|l| Entry::parse_line(l).is_some_and(|e| durable.is_some_and(|d| e.seq <= d))).map(|l| format!("{l}\n")).collect();
                if keep.len() < t.len() {
                    run.fault("restart_power_cut");
                }
                std::fs::write(&self.path, keep).unwrap();
            }
            2 => {
                // a power cut after part of what was not durable reached the
                // disk: the lines up to some point past the last barrier
                let t = std::fs::read_to_string(&self.path).unwrap();
                let lines: Vec<&str> = t.lines().collect();
                let from = lines.iter().filter(|l| Entry::parse_line(l).is_some_and(|e| durable.is_some_and(|d| e.seq <= d))).count();
                let k = from + rng.below((lines.len() - from) as u64 + 1) as usize;
                if k < lines.len() {
                    run.fault("restart_partial_cut");
                }
                std::fs::write(&self.path, lines[..k].iter().map(|l| format!("{l}\n")).collect::<String>()).unwrap();
            }
            _ => {}
        }
        let t = std::fs::read_to_string(&self.path).unwrap();
        t.lines().filter_map(Entry::parse_line).map(|e| e.seq).next_back()
    }
    fn open(&mut self, adapters: Vec<Box<dyn Adapter + 'm>>) {
        self.h = Some(Host::open_file_with(self.m, &self.path, adapters).unwrap_or_else(|e| panic!("reopen {}: {e}", self.path.display())));
    }
}

/// The wall clock the admitter reads (HC-2): true time plus a skew that
/// steps when the skew class fires.
struct Clock {
    skew_ns: i64,
}

impl Clock {
    fn read(&mut self, round: u64, mix: &Mix, rng: &mut Rng, run: &mut Run) -> u64 {
        if rng.chance(mix.skew) {
            run.fault("skew");
            // an NTP step: back up to 3 s, forward up to 3 s, or back to true time
            self.skew_ns = match rng.below(3) {
                0 => -(rng.below(3_000_000_000) as i64),
                1 => rng.below(3_000_000_000) as i64,
                _ => 0,
            };
        }
        // a clock does not read before the epoch
        ((round * ROUND_NS + 1_000_000_000_000) as i64 + self.skew_ns).max(0) as u64
    }
}

/// (a), (b), (c) on the D a run left, against the live state.
fn check_d(m: &Module, text: &str, live_hash: &str, live_state: &str, run: &mut Run) {
    const CAUSES: [&str; 6] = ["fail", "fuel", "size", "value", "depth", "trap"];
    let entries: Vec<Entry> = text.lines().map(|l| Entry::parse_line(l).unwrap_or_else(|| panic!("a line of D does not parse: {l}"))).collect();
    run.entries = entries.len() as u64;
    // (a) replay with every adapter disconnected
    let rep = match replay_report(m, text) {
        Ok(r) => r,
        Err(e) => {
            run.a.push(format!("replay failed: {e}"));
            return;
        }
    };
    if rep.state_hash != live_hash {
        run.a.push(format!("replay hash {} differs from live {live_hash}", rep.state_hash));
    }
    if format!("{:?}", rep.state) != live_state {
        run.a.push("replayed state differs from the live state".into());
    }
    // (b) every entry is a step or a kept step
    let kept: BTreeMap<u64, &str> = rep.kept_outcomes.iter().map(|(q, c, _)| (*q, *c)).collect();
    for c in kept.values() {
        *run.kept.entry(c.to_string()).or_default() += 1;
        if !CAUSES.contains(c) {
            run.b.push(format!("kept cause {c} is not one of HC-21"));
        }
    }
    for w in entries.windows(2) {
        let (p, e) = (&w[0], &w[1]);
        if (e.is_host() || kept.contains_key(&e.seq)) && e.state_hash != p.state_hash {
            run.b.push(format!("entry {} ({}) changes the state hash although it is {}", e.seq, e.kind(), if e.is_host() { "a host entry" } else { "kept" }));
        }
    }
    // (c) restarts never reuse ids
    let mut floor: Option<u64> = None;
    let mut net_ids = BTreeSet::new();
    let mut http_ids = BTreeSet::new();
    for e in &entries {
        let v: Value = serde_json::from_str(&e.event).unwrap();
        let p = &v["payload"];
        match e.kind().as_str() {
            "host.Restarted" => {
                if p["below"].as_u64() != Some(e.seq) {
                    run.c.push(format!("host.Restarted at {} has below {}", e.seq, p["below"]));
                }
                floor = Some(e.seq);
            }
            "NetListenDone" | "NetAcceptDone" | "NetConnectDone" if p["status"] == "delivered" => {
                let id = p["data"].as_u64().or_else(|| p["data"]["some"].as_u64());
                match id {
                    Some(id) => {
                        run.ids += 1;
                        run.ids_after_restart += u64::from(floor.is_some());
                        if floor.is_some_and(|f| id < f) {
                            run.c.push(format!("entry {}: id {id} issued after the restart at {} is below it", e.seq, floor.unwrap()));
                        }
                        if !net_ids.insert(id) {
                            run.c.push(format!("entry {}: id {id} issued twice", e.seq));
                        }
                    }
                    None => run.c.push(format!("entry {}: no id in {}", e.seq, p["data"])),
                }
            }
            "HttpRequest" | "HttpMalformed" => {
                let id = p["id"].as_u64().unwrap_or(u64::MAX);
                if !http_ids.insert(id) {
                    run.c.push(format!("entry {}: HTTP request id {id} used twice", e.seq));
                }
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------- event streams

/// A program driven by one client's sequence of events.
struct Stream {
    name: &'static str,
    file: &'static str,
    ops: fn(&mut Rng) -> Vec<String>,
    /// the program reads `WallTime`: the admitter delivers it every round
    clock: bool,
    /// operations in flight at once
    window: usize,
}

fn gen_account(r: &mut Rng) -> Vec<String> {
    (0..30).map(|_| if r.chance(55) { format!("Deposit(cents: {})", 1 + r.below(1000)) } else { format!("Withdraw(cents: {})", 1 + r.below(1500)) }).collect()
}

fn gen_ledger(r: &mut Rng) -> Vec<String> {
    const A: [&str; 4] = ["cash", "rent", "food", "bank"];
    (0..30)
        .map(|_| {
            let (s, d) = (A[r.below(4) as usize], A[r.below(4) as usize]);
            format!("Transfer(src: \"{s}\", dst: \"{d}\", cents: {})", r.below(900) as i64 - 50)
        })
        .collect()
}

fn gen_handshake(r: &mut Rng) -> Vec<String> {
    let mut v = vec![];
    for _ in 0..3 {
        v.extend(["Syn", "SynAck", "Ack"].map(String::from));
        for _ in 0..(1 + r.below(5)) {
            v.push(format!("Data(n: {})", 1 + r.below(100)));
        }
        v.push("Fin".into());
    }
    v
}

fn gen_ratelimit(r: &mut Rng) -> Vec<String> {
    (0..40).map(|_| format!("Request(client: \"{}\")", if r.chance(50) { "a" } else { "b" })).collect()
}

const STREAMS: [Stream; 4] = [
    Stream { name: "account", file: "../../programs/money/account.rce", ops: gen_account, clock: false, window: 3 },
    Stream { name: "ledger", file: "../../programs/money/ledger.rce", ops: gen_ledger, clock: false, window: 3 },
    Stream { name: "handshake", file: "../../programs/network/handshake.rce", ops: gen_handshake, clock: false, window: 3 },
    Stream { name: "ratelimit", file: "tests/host/faults/ratelimit.rce", ops: gen_ratelimit, clock: true, window: 1 },
];

/// How the client's inputs reach the admitter.
#[derive(Clone, Copy, PartialEq)]
enum Path_ {
    /// typed lines, as on the input stream of `rce host run` and `serve`
    /// (HC-10b): no source identity, so nothing can dedupe them
    Typed,
    /// (source, seq, tick) inputs in admission windows (HC-15 to HC-17)
    Sourced,
}

fn outcome_text(o: &Result<Vec<String>, String>) -> String {
    match o {
        Ok(_) => "applied".into(),
        Err(r) => format!("kept: {r}"),
    }
}

/// The answer a sourced client gets for a copy the admitter found in D.
const DUP: &str = "duplicate";

struct Copy_ {
    at: u64,
    n: u64,
    op: usize,
    sent: u64,
}

fn run_stream(p: &Stream, path: Path_, mx: &Mix, seed: u64, dir: &Path) -> Run {
    let src = read(p.file);
    let m = rce_spec_check::program(&src).unwrap();
    let mut run = Run::default();
    let ops = (p.ops)(&mut Rng(seed ^ 0xa5a5_0001));
    let evs: Vec<(String, Vec<Arg>)> = ops.iter().map(|l| event_of(&src, l)).collect();
    // the oracle: each operation once, in the client's order
    let mut reference = Host::start(&m, Box::new(MemStore::default())).unwrap();
    let mut want: Vec<String> = vec![];
    for (n, a) in &evs {
        let ad = reference.admit_event(n, a.clone()).unwrap();
        want.push(outcome_text(&ad[0].outcome));
    }
    let mut rng = Rng(seed ^ 0x7777_0002);
    let mut clock = Clock { skew_ns: 0 };
    let log = dir.join(format!("{}.d", p.name));
    let _ = std::fs::remove_file(&log);
    let mut life = Life { m: &m, path: log.clone(), h: None };
    life.open(vec![]);
    // the client
    let mut next = 0usize;
    let mut outstanding: BTreeMap<usize, u64> = BTreeMap::new();
    let mut done: Vec<Option<String>> = vec![None; ops.len()];
    // (op, seq) of every admission of an operation that is still in D
    let mut applied: Vec<(usize, u64, bool, u64)> = vec![];
    let mut wire: Vec<Copy_> = vec![];
    let mut copies = 0u64;
    let mut round = 0;
    while round < MAX_ROUNDS {
        round += 1;
        if p.clock || rng.chance(mx.skew) {
            let t = clock.read(round, mx, &mut rng, &mut run);
            life.h().wall_time(t).unwrap();
        }
        // send new operations, retransmit unanswered ones
        while outstanding.len() < p.window && next < ops.len() {
            outstanding.insert(next, 0);
            next += 1;
        }
        for (&op, sent) in outstanding.iter_mut() {
            if *sent != 0 && round - *sent < TIMEOUT {
                continue;
            }
            *sent = round;
            let k = if rng.chance(mx.dup) {
                run.fault("dup");
                2
            } else {
                1
            };
            for _ in 0..k {
                if rng.chance(mx.drop) {
                    run.fault("drop");
                    continue;
                }
                let d = rng.below(mx.delay + 1);
                if d > 0 {
                    run.fault("reorder");
                }
                copies += 1;
                wire.push(Copy_ { at: round + d, n: copies, op, sent: round });
            }
        }
        wire.sort_by_key(|c| (c.at, c.n));
        let k = wire.iter().take_while(|c| c.at <= round).count();
        let arrived: Vec<Copy_> = wire.drain(..k).collect();
        // a kill in this round, at a position among the arrivals
        let kill_at = rng.chance(mx.restart).then(|| rng.below(arrived.len() as u64 + 1) as usize);
        let mut answers: Vec<(usize, String)> = vec![];
        let mut killed = false;
        match path {
            Path_::Typed => {
                let mut batch: Vec<(usize, String, u64, bool)> = vec![];
                for (i, c) in arrived.iter().enumerate() {
                    if kill_at == Some(i) {
                        killed = true;
                        break;
                    }
                    let (n, a) = &evs[c.op];
                    let ad = life.h().apply_input(n, a.clone()).unwrap();
                    batch.push((c.op, outcome_text(&ad[0].outcome), ad[0].seq, ad[0].outcome.is_ok()));
                }
                if !killed {
                    life.h().settle().unwrap();
                }
                for (op, o, seq, ok) in batch {
                    applied.push((op, seq, ok, round));
                    if !killed {
                        answers.push((op, o));
                    }
                }
            }
            Path_::Sourced => {
                for (i, c) in arrived.iter().enumerate() {
                    if kill_at == Some(i) {
                        killed = true;
                        break;
                    }
                    let (n, a) = &evs[c.op];
                    let x = Sourced::new("client", c.op as u64 + 1, c.sent, n, a.clone());
                    life.h().offer(x).unwrap();
                }
                if !killed {
                    let w = life.h().close_window(round).unwrap();
                    run.host_dups += w.dups.len() as u64;
                    // HC-16: a duplicate is acked, but its window holds no
                    // outcome: the answer of the first copy is not repeated
                    for (_, q) in &w.dups {
                        answers.push((*q as usize - 1, DUP.into()));
                    }
                    assert_eq!(w.acks.len(), w.admitted.len(), "seed {seed}: a stream input requested an effect");
                    for ((_, q), ad) in w.acks.iter().zip(w.admitted.iter()) {
                        let op = *q as usize - 1;
                        applied.push((op, ad.seq, ad.outcome.is_ok(), round));
                        answers.push((op, outcome_text(&ad.outcome)));
                    }
                }
            }
        }
        if kill_at == Some(arrived.len()) && !killed {
            killed = true;
        }
        if killed {
            let top = life.kill(&mut rng, &mut run);
            applied.retain(|(_, q, _, _)| top.is_some_and(|t| *q <= t));
            life.open(vec![]);
        }
        // the answers reach the client, or are lost
        for (op, o) in answers {
            if rng.chance(mx.lose) {
                run.fault("lose");
                continue;
            }
            if done[op].is_none() {
                done[op] = Some(o);
                outstanding.remove(&op);
            }
        }
        if next == ops.len() && outstanding.is_empty() && wire.is_empty() {
            run.live = true;
            break;
        }
    }
    run.rounds = round;
    let h = life.h.take().unwrap();
    let live_hash = h.state_hash().to_string();
    let live_state = format!("{:?}", h.state());
    let final_state = h.state().to_vec();
    let ref_hash = reference.state_hash().to_string();
    let allowed = p.clock.then(|| {
        let v = final_state.iter().flat_map(|(_, fs)| fs.iter()).find(|(n, _, _)| n == "allowed").map(|(_, v, _)| v.clone()).unwrap();
        text_map(&v)
    });
    drop(h);
    let text = std::fs::read_to_string(&log).unwrap();
    check_d(&m, &text, &live_hash, &live_state, &mut run);
    // (d) against the client's intent
    let mut count: BTreeMap<usize, u64> = BTreeMap::new();
    for (op, _, ok, _) in &applied {
        if *ok {
            *count.entry(*op).or_default() += 1;
        }
    }
    for (op, n) in &count {
        if *n > 1 {
            run.mishandled("duplicate_applied", format!("op {op} `{}` applied {n} times", ops[*op]));
        }
    }
    if p.clock {
        // a token bucket allows at most BURST, then one per second of true
        // time: from the first request of a client in D to its last
        let mut span: BTreeMap<String, (u64, u64)> = BTreeMap::new();
        for (op, _, _, r) in &applied {
            let c = ops[*op].split('"').nth(1).unwrap().to_string();
            let e = span.entry(c).or_insert((*r, *r));
            e.0 = e.0.min(*r);
            e.1 = e.1.max(*r);
        }
        for (c, n) in allowed.unwrap_or_default() {
            let (a, b) = span.get(&c).copied().unwrap_or((0, 0));
            let secs = (b - a) * ROUND_NS / 1_000_000_000;
            if n > 3 + secs {
                run.mishandled("over_limit", format!("client {c} allowed {n}, more than 3 + {secs} s of true time"));
            }
        }
        for (q, c, t) in &replay_report(&m, &text).map(|r| r.kept_outcomes).unwrap_or_default() {
            if *c == "trap" {
                run.mishandled("kept_on_clock_step", format!("seq {q}: {t}"));
            }
        }
    } else {
        for (op, o) in done.iter().enumerate() {
            if o.as_deref() == Some(DUP) {
                *run.gap.entry("duplicate_acked_without_outcome").or_default() += 1;
            } else if let Some(o) = o
                && *o != want[op]
            {
                run.mishandled("answer_differs", format!("op {op} `{}` answered `{o}`, in order `{}`", ops[op], want[op]));
            }
        }
        if live_hash != ref_hash {
            run.mishandled("state_differs", "the final state is not the fold of the client's operations in order".into());
        }
    }
    run.d_text = text;
    run
}

// ---------------------------------------------------------------- HTTP clients

/// One client operation on the notes service.
#[derive(Clone, Debug)]
enum NoteOp {
    Post(String),
    /// delete the note made by the client's post at this index
    Delete(usize),
    List,
}

fn gen_notes(r: &mut Rng, clients: usize, per: usize) -> Vec<Vec<NoteOp>> {
    (0..clients)
        .map(|c| {
            let mut v = vec![];
            for k in 0..per {
                let gone: Vec<usize> = v.iter().filter_map(|o| if let NoteOp::Delete(j) = o { Some(*j) } else { None }).collect();
                let posts: Vec<usize> = v.iter().enumerate().filter(|(i, o)| matches!(o, NoteOp::Post(_)) && !gone.contains(i)).map(|(i, _)| i).collect();
                let x = r.below(10);
                v.push(if x < 6 || posts.is_empty() {
                    NoteOp::Post(format!("c{c}n{k}"))
                } else if x < 8 {
                    NoteOp::Delete(posts[r.below(posts.len() as u64) as usize])
                } else {
                    NoteOp::List
                });
            }
            v
        })
        .collect()
}

/// A client of the notes service: one request at a time, retried on timeout.
struct Client {
    ops: Vec<NoteOp>,
    at: usize,
    sent: u64,
    /// the note id each post was answered with
    ids: BTreeMap<usize, u64>,
    /// the final status of each operation
    status: Vec<Option<u64>>,
}

impl Client {
    fn busy(&self) -> bool {
        self.at < self.ops.len()
    }
    /// The operation's request, or None when a delete names a note whose
    /// post was never answered with an id.
    fn request(&self) -> Option<(String, String, Vec<u8>)> {
        match &self.ops[self.at] {
            NoteOp::Post(b) => Some(("POST".into(), "/notes".into(), b.as_bytes().to_vec())),
            NoteOp::Delete(i) => self.ids.get(i).map(|id| ("DELETE".into(), format!("/notes/{id}"), vec![])),
            NoteOp::List => Some(("GET".into(), "/notes".into(), vec![])),
        }
    }
    fn answer(&mut self, op: usize, status: u64, body: &str) {
        if op != self.at {
            return; // an answer to an earlier attempt of a finished operation
        }
        if let NoteOp::Post(_) = self.ops[op]
            && status == 201
        {
            let id = body.split("\"id\":").nth(1).and_then(|s| s.split(',').next()).and_then(|s| s.parse().ok());
            if let Some(id) = id {
                self.ids.insert(op, id);
            }
        }
        self.status[op] = Some(status);
        self.at += 1;
        self.sent = 0;
    }
}

/// (d) for the notes services: the client's intent is each post makes one
/// note and each delete removes it.
fn judge_notes(clients: &[Client], notes: &BTreeMap<String, u64>, run: &mut Run) {
    let mut want: BTreeMap<String, u64> = BTreeMap::new();
    for c in clients {
        for (i, o) in c.ops.iter().enumerate() {
            match o {
                NoteOp::Post(b) => *want.entry(b.clone()).or_default() += 1,
                NoteOp::Delete(j) => {
                    if c.ids.contains_key(j)
                        && let NoteOp::Post(b) = &c.ops[*j]
                    {
                        *want.entry(b.clone()).or_default() -= 1;
                    }
                    if c.status[i] == Some(404) {
                        run.mishandled("delete_answered_404", format!("delete of {:?} answered 404: an earlier attempt deleted it", c.ops[*j]));
                    }
                }
                NoteOp::List => {}
            }
        }
    }
    for (b, n) in &want {
        let got = notes.get(b).copied().unwrap_or(0);
        if got > *n {
            run.mishandled("duplicate_note", format!("note {b:?} held {got} times, intended {n}"));
        } else if got < *n {
            run.mishandled("lost_note", format!("note {b:?} held {got} times, intended {n}"));
        }
    }
}

// ---------------------------------------------------------------- HTTP event host

#[derive(Default)]
struct HttpWorld {
    /// requests of this process the program may answer
    open: BTreeSet<u64>,
    out: Vec<(u64, u64, String)>,
}

/// `HttpRespond` performed against the simulated connections (HC-26): a
/// req that is not open fails with `no request N`.
struct Responder(Rc<RefCell<HttpWorld>>);
impl Adapter for Responder {
    fn kind(&self) -> &'static str {
        "HttpRespond"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let mut w = self.0.borrow_mut();
        let req = r["req"].as_u64().unwrap_or(0);
        if !w.open.remove(&req) {
            return EffectResult::failed(format!("no request {req}"));
        }
        let body = r["body"].as_str().unwrap_or("").to_string();
        let n = body.len();
        w.out.push((req, r["status"].as_u64().unwrap_or(0), body));
        EffectResult::ok(json!(n))
    }
}

struct Attempt {
    at: u64,
    n: u64,
    client: usize,
    op: usize,
    req: (String, String, Vec<u8>),
}

fn run_notes_http(mx: &Mix, seed: u64, dir: &Path) -> Run {
    let src = read("../../programs/network/notes_http.rce");
    let m = rce_spec_check::program(&src).unwrap();
    let mut run = Run::default();
    let mut clients: Vec<Client> = gen_notes(&mut Rng(seed ^ 0xa5a5_0003), 3, 8).into_iter().map(|ops| Client { status: vec![None; ops.len()], ops, at: 0, sent: 0, ids: BTreeMap::new() }).collect();
    let mut rng = Rng(seed ^ 0x7777_0004);
    let mut clock = Clock { skew_ns: 0 };
    let world = Rc::new(RefCell::new(HttpWorld::default()));
    let log = dir.join("notes_http.d");
    let _ = std::fs::remove_file(&log);
    let mut life = Life { m: &m, path: log.clone(), h: None };
    life.open(vec![Box::new(Responder(world.clone()))]);
    let mut wire: Vec<Attempt> = vec![];
    let mut copies = 0;
    let mut round = 0;
    while round < MAX_ROUNDS {
        round += 1;
        if rng.chance(mx.skew) {
            let t = clock.read(round, mx, &mut rng, &mut run);
            life.h().wall_time(t).unwrap();
        }
        for (ci, c) in clients.iter_mut().enumerate() {
            while c.busy() && c.request().is_none() {
                // a delete of a note the client never learned the id of
                c.status[c.at] = None;
                c.at += 1;
            }
            if !c.busy() || (c.sent != 0 && round - c.sent < TIMEOUT) {
                continue;
            }
            c.sent = round;
            let req = c.request().unwrap();
            let k = if rng.chance(mx.dup) {
                run.fault("dup");
                2
            } else {
                1
            };
            for _ in 0..k {
                if rng.chance(mx.drop) {
                    // the connection closes before the request is complete:
                    // nothing enters D (HC-24)
                    run.fault("drop");
                    continue;
                }
                let d = rng.below(mx.delay + 1);
                if d > 0 {
                    run.fault("reorder");
                }
                copies += 1;
                wire.push(Attempt { at: round + d, n: copies, client: ci, op: c.at, req: req.clone() });
            }
        }
        wire.sort_by_key(|a| (a.at, a.n));
        let k = wire.iter().take_while(|a| a.at <= round).count();
        let arrived: Vec<Attempt> = wire.drain(..k).collect();
        let kill_at = rng.chance(mx.restart).then(|| rng.below(arrived.len() as u64 + 1) as usize);
        let mut ids: BTreeMap<u64, (usize, usize)> = BTreeMap::new();
        let mut host_answers: Vec<(usize, usize, u64)> = vec![];
        let mut killed = false;
        for (i, a) in arrived.iter().enumerate() {
            if kill_at == Some(i) {
                killed = true;
                break;
            }
            let mut p = serde_json::Map::new();
            p.insert("body".into(), Value::from(a.req.2.clone()));
            p.insert("method".into(), a.req.0.clone().into());
            p.insert("names".into(), json!(["Host"]));
            p.insert("target".into(), a.req.1.clone().into());
            p.insert("values".into(), json!(["notes.test"]));
            p.insert("version".into(), "HTTP/1.1".into());
            let ad = life.h().http_input("HttpRequest", p).unwrap();
            match ad.answer {
                Some(s) => host_answers.push((a.client, a.op, s)),
                None => {
                    world.borrow_mut().open.insert(ad.id);
                    ids.insert(ad.id, (a.client, a.op));
                }
            }
        }
        if !killed {
            life.h().settle().unwrap();
        }
        if kill_at == Some(arrived.len()) {
            killed = true;
        }
        let out: Vec<(u64, u64, String)> = std::mem::take(&mut world.borrow_mut().out);
        if killed {
            // the connections belong to the process that is gone
            world.borrow_mut().open.clear();
            life.kill(&mut rng, &mut run);
            life.open(vec![Box::new(Responder(world.clone()))]);
        }
        let mut answers: Vec<(usize, usize, u64, String)> = host_answers.into_iter().map(|(c, o, s)| (c, o, s, String::new())).collect();
        for (req, status, body) in out {
            if let Some(&(c, o)) = ids.get(&req) {
                answers.push((c, o, status, body));
            }
        }
        for (c, o, s, b) in answers {
            if rng.chance(mx.lose) {
                run.fault("lose");
                continue;
            }
            clients[c].answer(o, s, &b);
        }
        if clients.iter().all(|c| !c.busy()) && wire.is_empty() {
            run.live = true;
            break;
        }
    }
    run.rounds = round;
    let h = life.h.take().unwrap();
    let (live_hash, live_state) = (h.state_hash().to_string(), format!("{:?}", h.state()));
    let notes = note_texts(&field(&h, "notes"));
    drop(h);
    let text = std::fs::read_to_string(&log).unwrap();
    check_d(&m, &text, &live_hash, &live_state, &mut run);
    judge_notes(&clients, &notes, &mut run);
    run.d_text = text;
    run
}

// ---------------------------------------------------------------- net adapter

/// One simulated TCP connection from a client.
struct Conn {
    client: usize,
    op: usize,
    /// (round it arrives, bytes), in order
    chunks: VecDeque<(u64, Vec<u8>)>,
    /// the client reset the connection after the chunks it sent
    reset: bool,
    sent: Vec<u8>,
}

#[derive(Default)]
struct NetWorld {
    now: u64,
    /// connections not yet accepted, with the round they arrive
    backlog: Vec<(u64, u64, Conn)>,
    /// accepted connections, by world id
    conns: BTreeMap<u64, Conn>,
    /// closed connections whose response the client may read
    closed: Vec<Conn>,
    listening: bool,
    next_conn: u64,
}

/// The sockets of one host process.
#[derive(Default)]
struct Proc {
    next: u64,
    socks: BTreeMap<u64, Option<u64>>, // None: the listener; Some(world id): a connection
}

struct SimNet {
    kind: &'static str,
    p: Rc<RefCell<Proc>>,
    w: Rc<RefCell<NetWorld>>,
}

impl Adapter for SimNet {
    fn kind(&self) -> &'static str {
        self.kind
    }
    /// HC-22, as the reference net adapter does it: ids continue from floor.
    fn restart(&mut self, floor: u64) {
        let mut p = self.p.borrow_mut();
        p.next = p.next.max(floor.saturating_sub(1));
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let mut p = self.p.borrow_mut();
        let mut w = self.w.borrow_mut();
        let id = r["id"].as_u64().unwrap_or(0);
        match self.kind {
            "NetListen" => {
                if w.listening {
                    return EffectResult::failed("address in use");
                }
                w.listening = true;
                p.next += 1;
                let n = p.next;
                p.socks.insert(n, None);
                EffectResult::ok(json!(n))
            }
            "NetAccept" => {
                if p.socks.get(&id) != Some(&None) {
                    return EffectResult::failed(format!("no listener {id}"));
                }
                let now = w.now;
                let Some(i) = w.backlog.iter().position(|(at, _, _)| *at <= now) else {
                    return EffectResult::failed("none");
                };
                let (_, wid, c) = w.backlog.remove(i);
                w.conns.insert(wid, c);
                p.next += 1;
                let n = p.next;
                p.socks.insert(n, Some(wid));
                EffectResult::ok(json!(n))
            }
            "NetRecv" => {
                let Some(Some(wid)) = p.socks.get(&id).copied() else {
                    return EffectResult::failed(format!("no connection {id}"));
                };
                let now = w.now;
                let c = w.conns.get_mut(&wid).unwrap();
                let mut got = vec![];
                while c.chunks.front().is_some_and(|(at, _)| *at <= now) {
                    got.extend(c.chunks.pop_front().unwrap().1);
                }
                if got.is_empty() && c.chunks.is_empty() {
                    return EffectResult::failed(if c.reset { "connection reset" } else { "eof" });
                }
                EffectResult::ok(json!(got))
            }
            "NetSend" => {
                let Some(Some(wid)) = p.socks.get(&id).copied() else {
                    return EffectResult::failed(format!("no connection {id}"));
                };
                let d = r["data"].as_str().unwrap_or("").as_bytes().to_vec();
                let n = d.len();
                w.conns.get_mut(&wid).unwrap().sent.extend(d);
                EffectResult::ok(json!(n))
            }
            "NetClose" => match p.socks.remove(&id) {
                Some(Some(wid)) => {
                    let c = w.conns.remove(&wid).unwrap();
                    w.closed.push(c);
                    EffectResult::ok(json!(0))
                }
                Some(None) => {
                    w.listening = false;
                    EffectResult::ok(json!(0))
                }
                None => EffectResult::failed(format!("no connection {id}")),
            },
            k => EffectResult::failed(format!("{k} is not simulated")),
        }
    }
}

fn sim_net<'m>(w: &Rc<RefCell<NetWorld>>) -> Vec<Box<dyn Adapter + 'm>> {
    let p = Rc::new(RefCell::new(Proc::default()));
    ["NetListen", "NetAccept", "NetRecv", "NetSend", "NetClose"].iter().map(|k| Box::new(SimNet { kind: k, p: p.clone(), w: w.clone() }) as Box<dyn Adapter + 'm>).collect()
}

fn http_bytes(method: &str, target: &str, body: &[u8]) -> Vec<u8> {
    let mut v = format!("{method} {target} HTTP/1.1\r\nHost: notes.test\r\n").into_bytes();
    if !body.is_empty() {
        v.extend(format!("Content-Length: {}\r\n", body.len()).into_bytes());
    }
    v.extend(b"\r\n");
    v.extend(body);
    v
}

fn run_notes_net(mx: &Mix, seed: u64, dir: &Path) -> Run {
    let src = read("tests/host/notes_net.rce");
    let m = rce_spec_check::program(&src).unwrap();
    let mut run = Run::default();
    let mut clients: Vec<Client> = gen_notes(&mut Rng(seed ^ 0xa5a5_0005), 2, 6).into_iter().map(|ops| Client { status: vec![None; ops.len()], ops, at: 0, sent: 0, ids: BTreeMap::new() }).collect();
    let mut rng = Rng(seed ^ 0x7777_0006);
    let mut clock = Clock { skew_ns: 0 };
    let world = Rc::new(RefCell::new(NetWorld::default()));
    let log = dir.join("notes_net.d");
    let _ = std::fs::remove_file(&log);
    let mut life = Life { m: &m, path: log.clone(), h: None };
    life.open(sim_net(&world));
    life.h().admit(&src, "Start(addr: \"127.0.0.1:8080\")").unwrap();
    let mut round = 0;
    while round < MAX_ROUNDS {
        round += 1;
        world.borrow_mut().now = round;
        if rng.chance(mx.skew) {
            let t = clock.read(round, mx, &mut rng, &mut run);
            life.h().wall_time(t).unwrap();
        }
        for (ci, c) in clients.iter_mut().enumerate() {
            while c.busy() && c.request().is_none() {
                c.at += 1;
            }
            if !c.busy() || (c.sent != 0 && round - c.sent < TIMEOUT * 2) {
                continue;
            }
            c.sent = round;
            let (me, t, b) = c.request().unwrap();
            let bytes = http_bytes(&me, &t, &b);
            let k = if rng.chance(mx.dup) {
                run.fault("dup");
                2
            } else {
                1
            };
            for _ in 0..k {
                let d = rng.below(mx.delay + 1);
                if d > 0 {
                    run.fault("reorder");
                }
                // the bytes in one to three segments
                let mut cuts: Vec<usize> = (0..rng.below(3)).map(|_| 1 + rng.below(bytes.len() as u64 - 1) as usize).collect();
                cuts.sort();
                cuts.dedup();
                let mut chunks = VecDeque::new();
                let mut from = 0;
                for (i, &x) in cuts.iter().chain(std::iter::once(&bytes.len())).enumerate() {
                    chunks.push_back((round + d + i as u64, bytes[from..x].to_vec()));
                    from = x;
                }
                let mut reset = false;
                if rng.chance(mx.drop) {
                    // the client resets the connection part way through
                    run.fault("drop");
                    let keep = rng.below(chunks.len() as u64) as usize;
                    chunks.truncate(keep);
                    reset = true;
                }
                let mut w = world.borrow_mut();
                w.next_conn += 1;
                let wid = w.next_conn;
                w.backlog.push((round + d, wid, Conn { client: ci, op: c.at, chunks, reset, sent: vec![] }));
            }
        }
        let killed = rng.chance(mx.restart);
        if killed {
            // the sockets of the process that is gone: every connection resets
            {
                let mut w = world.borrow_mut();
                w.backlog.clear();
                w.conns.clear();
                w.closed.clear();
                w.listening = false;
            }
            life.kill(&mut rng, &mut run);
            life.open(sim_net(&world));
        }
        let n = life.h().last_tick() + 1;
        life.h().tick(n).unwrap();
        let closed: Vec<Conn> = std::mem::take(&mut world.borrow_mut().closed);
        for c in closed {
            if c.sent.is_empty() {
                continue;
            }
            if rng.chance(mx.lose) {
                run.fault("lose");
                continue;
            }
            let t = String::from_utf8_lossy(&c.sent).to_string();
            let status: u64 = t.get(9..12).and_then(|s| s.parse().ok()).unwrap_or(0);
            let body = t.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
            clients[c.client].answer(c.op, status, &body);
        }
        let w = world.borrow();
        if clients.iter().all(|c| !c.busy()) && w.backlog.is_empty() && w.conns.is_empty() {
            run.live = true;
            break;
        }
    }
    run.rounds = round;
    let h = life.h.take().unwrap();
    let (live_hash, live_state) = (h.state_hash().to_string(), format!("{:?}", h.state()));
    let notes = note_texts(&field(&h, "notes"));
    drop(h);
    let text = std::fs::read_to_string(&log).unwrap();
    check_d(&m, &text, &live_hash, &live_state, &mut run);
    judge_notes(&clients, &notes, &mut run);
    run.d_text = text;
    run
}

// ---------------------------------------------------------------- driver

const PROGRAMS: [&str; 9] = ["account/typed", "account/sourced", "ledger/typed", "ledger/sourced", "handshake/typed", "handshake/sourced", "ratelimit/typed", "notes_http/http", "notes_net/net"];

fn run_one(program: &str, mix_name: &str, seed: u64) -> Run {
    let mx = mix(mix_name);
    let base = std::env::var("RCE_FAULT_DIR").map(PathBuf::from).unwrap_or_else(|_| std::env::temp_dir());
    let dir = base.join(format!("rce-faults-{}-{}-{mix_name}-{seed}", std::process::id(), program.replace('/', "-")));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let (name, path) = program.split_once('/').unwrap();
    let run = match (name, path) {
        ("notes_http", _) => run_notes_http(&mx, seed, &dir),
        ("notes_net", _) => run_notes_net(&mx, seed, &dir),
        (s, p) => {
            let st = STREAMS.iter().find(|x| x.name == s).unwrap_or_else(|| panic!("no program {s}"));
            run_stream(st, if p == "sourced" { Path_::Sourced } else { Path_::Typed }, &mx, seed, &dir)
        }
    };
    if std::env::var("RCE_FAULT_KEEP").is_err() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    run
}

fn record(program: &str, mix_name: &str, seed: u64, r: &Run, rerun_same: Option<bool>) -> Value {
    json!({
        "program": program, "mix": mix_name, "seed": seed, "rounds": r.rounds, "live": r.live,
        "entries": r.entries, "faults": r.faults, "host_dups_removed": r.host_dups,
        "a": r.a, "b": r.b, "c": r.c, "kept": r.kept, "d": r.d, "d_note": r.d_note, "gap": r.gap, "ids": r.ids, "ids_after_restart": r.ids_after_restart,
        "rerun_same_d": rerun_same,
    })
}

fn list(var: &str, default: &[&str]) -> Vec<String> {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => v.split(',').map(String::from).collect(),
        _ => default.iter().map(|s| s.to_string()).collect(),
    }
}

fn seeds() -> Vec<u64> {
    if let Ok(one) = std::env::var("RCE_FAULT_SEED") {
        return vec![one.parse().unwrap()];
    }
    let n: u64 = std::env::var("RCE_FAULT_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    (1..=n).collect()
}

/// Every run over the seeds: (a), (b) and (c) hold for every seed and fault
/// mix, sourced inputs are applied at most once (HC-16), and every tenth
/// seed reruns to a byte-identical D. Application-level counts (d) are
/// written to RCE_FAULT_OUT, not asserted: they are the programs' faults.
#[test]
fn host_contract_holds_under_seeded_faults() {
    let programs = list("RCE_FAULT_PROGRAMS", &PROGRAMS);
    let mixes = list("RCE_FAULT_MIXES", &["all"]);
    let ss = seeds();
    let threads: usize = std::env::var("RCE_FAULT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(2);
    let mut jobs: Vec<(String, String, u64)> = vec![];
    for p in &programs {
        for mx in &mixes {
            for s in &ss {
                jobs.push((p.clone(), mx.clone(), *s));
            }
        }
    }
    let jobs = std::sync::Mutex::new(jobs.into_iter());
    let out = std::env::var("RCE_FAULT_OUT").ok();
    let sink = std::sync::Mutex::new(out.map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).unwrap()));
    let failures = std::sync::Mutex::new(Vec::<String>::new());
    std::thread::scope(|sc| {
        for _ in 0..threads {
            sc.spawn(|| {
                loop {
                    let Some((p, mx, seed)) = jobs.lock().unwrap().next() else { break };
                    let r = run_one(&p, &mx, seed);
                    let rerun = (seed % 10 == 0).then(|| run_one(&p, &mx, seed).d_text == r.d_text);
                    let mut bad = vec![];
                    bad.extend(r.a.iter().map(|x| format!("(a) {x}")));
                    bad.extend(r.b.iter().map(|x| format!("(b) {x}")));
                    bad.extend(r.c.iter().map(|x| format!("(c) {x}")));
                    if rerun == Some(false) {
                        bad.push("(a) a rerun of the seed wrote a different D".into());
                    }
                    if p.ends_with("/sourced") && r.d.contains_key("duplicate_applied") {
                        bad.push(format!("HC-16: a sourced input was applied twice: {:?}", r.d_note));
                    }
                    if !bad.is_empty() {
                        failures.lock().unwrap().push(format!("{p} mix {mx} seed {seed}: {}", bad.join("; ")));
                    }
                    if let Some(f) = sink.lock().unwrap().as_mut() {
                        use std::io::Write;
                        writeln!(f, "{}", record(&p, &mx, seed, &r, rerun)).unwrap();
                    }
                }
            });
        }
    });
    let f = failures.into_inner().unwrap();
    assert!(f.is_empty(), "{} runs broke the host contract:\n{}", f.len(), f.join("\n"));
}

/// The harness is deterministic: one seed, two runs, the same D, for each
/// kind of program.
#[test]
fn a_fault_seed_reruns_to_the_same_d() {
    for p in ["account/typed", "notes_http/http", "notes_net/net"] {
        let a = run_one(p, "all", 7);
        let b = run_one(p, "all", 7);
        assert!(a.d_text == b.d_text, "{p} seed 7: two runs wrote different D files");
        assert!(a.live, "{p} seed 7: the run did not finish");
    }
}

/// With no faults, no program mishandles anything: the oracle is the
/// program's own fault-free behaviour.
#[test]
fn no_faults_means_no_mishandling() {
    for p in PROGRAMS {
        for seed in 1..=3 {
            let r = run_one(p, "none", seed);
            assert!(r.live, "{p} seed {seed}: not live");
            assert!(r.d.is_empty(), "{p} seed {seed}: {:?} {:?}", r.d, r.d_note);
            assert!(r.faults.is_empty(), "{p} seed {seed}: faults injected under `none`: {:?}", r.faults);
        }
    }
}
