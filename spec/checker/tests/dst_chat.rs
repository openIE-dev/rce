//! W6 E2: protocol-aware deterministic simulation of P4 channels and W4
//! chat, after TigerBeetle's protocol-aware DST (C. Bhandari, 20 August
//! 2026). Three chat replicas run on the real host (durable D files) over a
//! simulated network and clock driven by one seed: drops, duplicates,
//! reorders (random delays), partitions and kill -9 with a torn tail.
//! The simulator reads each replica's protocol state, not only output, and
//! after the faults heal asserts, per replica:
//! - liveness: every reliable op ends Delivered (no op is left in `iwire`
//!   or `retry`), within a bound of rounds;
//! - every replica (not a majority) holds the same replicated state:
//!   messages, reactions, edits and deletes of shared conversations, and
//!   the DMs of each pair on both of its members;
//! - a restarted replica recovers from its own D alone: reopening sends
//!   nothing to peers, and every D replays with no adapters to the live hash;
//! - acks and retries stop: once converged, more rounds send no reliable op;
//! - each pair's Double Ratchet exists on both sides and nothing was rejected.
//! A seed reruns to identical D files. Failing seeds go in REGRESSION.
use rce_spec_check::host::{Adapter, EffectResult, Host, Status, replay};
use rce_spec_check::interp::V;
use rce_spec_check::ast::Module;
use rce_spec_check::parse::parse;
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// Seeds that once failed; each must keep passing.
/// 1: DM keyed by different clocks on sender and receiver; 2, 3: an edit
/// or delete that arrived before its message was lost, and two deletes of
/// one message kept different clocks.
const REGRESSION: &[u64] = &[1, 2, 3];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
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
        self.below(100) < pct
    }
}

#[derive(Default)]
struct Faults {
    drop_pct: u64,
    dup_pct: u64,
    max_delay: u64,
}

struct Msg {
    at: u64,
    seq: u64,
    from: String,
    id: u64,
    body: String,
}

struct Net {
    rng: Rng,
    now: u64,
    seq: u64,
    f: Faults,
    cut: BTreeSet<(String, String)>,
    down: BTreeSet<String>,
    inbox: BTreeMap<String, Vec<Msg>>,
    /// the adapter's receive-side dedup set: memory, lost on kill -9
    seen: BTreeMap<String, HashSet<(String, u64)>>,
    /// reliable sends performed, per sender
    sends: BTreeMap<String, u64>,
}

const ATTEMPTS: u64 = 6; // CHAN_DEADLINE_MS / resend interval of the real adapter

impl Net {
    fn link_up(&self, a: &str, b: &str) -> bool {
        !self.down.contains(b) && !self.down.contains(a) && !self.cut.contains(&(a.into(), b.into()))
    }
    fn deliver(&mut self, from: &str, to: &str, id: u64, body: &str) {
        let at = self.now + self.rng.below(self.f.max_delay + 1);
        self.seq += 1;
        let m = Msg { at, seq: self.seq, from: from.into(), id, body: body.into() };
        self.inbox.entry(to.into()).or_default().push(m);
    }
    fn send(&mut self, me: &str, r: &Value) -> EffectResult {
        let peer = r["peer"].as_str().unwrap_or("").to_string();
        let id = r["id"].as_u64().unwrap_or(0);
        let body = r["body"].as_str().unwrap_or("").to_string();
        let reliable = r["reliable"].as_bool().unwrap_or(true);
        if !reliable {
            if self.link_up(me, &peer) && !self.rng.chance(self.f.drop_pct) {
                self.deliver(me, &peer, id, &body);
            }
            return EffectResult::ok(json!(id));
        }
        *self.sends.entry(me.into()).or_default() += 1;
        let mut reached = false;
        for _ in 0..ATTEMPTS {
            if !self.link_up(me, &peer) {
                continue;
            }
            reached = true;
            if self.rng.chance(self.f.drop_pct) {
                continue;
            }
            self.deliver(me, &peer, id, &body);
            if self.rng.chance(self.f.dup_pct) {
                self.deliver(me, &peer, id, &body);
            }
            // the ack crosses back
            if self.link_up(&peer, me) && !self.rng.chance(self.f.drop_pct) {
                return EffectResult::ok(json!(id));
            }
        }
        let status = if reached { Status::TimedOut } else { Status::Unreachable };
        EffectResult { status, data: None, error: format!("no ack from {peer} for {id}") }
    }
    fn recv(&mut self, me: &str, max: usize) -> EffectResult {
        let now = self.now;
        let q = self.inbox.entry(me.into()).or_default();
        q.sort_by_key(|m| (m.at, m.seq));
        let n = q.iter().take_while(|m| m.at <= now).count().min(max);
        let got: Vec<Msg> = q.drain(..n).collect();
        let seen = self.seen.entry(me.into()).or_default();
        let lines: Vec<String> = got.into_iter().filter(|m| seen.insert((m.from.clone(), m.id))).map(|m| format!("{}\t{}\t{}", m.from, m.id, m.body)).collect();
        EffectResult::ok(json!(lines))
    }
}

struct SimChan {
    kind: &'static str,
    me: String,
    net: Rc<RefCell<Net>>,
}
impl Adapter for SimChan {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let mut n = self.net.borrow_mut();
        if self.kind == "ChanSend" { n.send(&self.me, r) } else { n.recv(&self.me, r["max"].as_u64().unwrap_or(256) as usize) }
    }
}

struct SimRandom(Rng);
impl Adapter for SimRandom {
    fn kind(&self) -> &'static str {
        "Random"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let n = r["n"].as_u64().unwrap_or(32) as usize;
        EffectResult::ok(json!((0..n).map(|_| format!("{:02x}", self.0.next() as u8)).collect::<String>()))
    }
}

const NAMES: [&str; 3] = ["ann", "bob", "cy"];
fn addr(n: &str) -> String {
    format!("sim:{n}")
}

struct Sim<'m> {
    m: &'m Module,
    src: String,
    dir: PathBuf,
    net: Rc<RefCell<Net>>,
    hosts: Vec<Option<Host<'m>>>,
    restarts: u64,
    seed: u64,
}

impl<'m> Sim<'m> {
    fn adapters(&mut self, i: usize) -> Vec<Box<dyn Adapter + 'm>> {
        self.restarts += 1;
        let me = addr(NAMES[i]);
        vec![
            Box::new(SimChan { kind: "ChanSend", me: me.clone(), net: self.net.clone() }),
            Box::new(SimChan { kind: "ChanRecv", me, net: self.net.clone() }),
            Box::new(SimRandom(Rng(self.seed.wrapping_mul(1000).wrapping_add(self.restarts * 17 + i as u64)))),
        ]
    }
    fn path(&self, i: usize) -> PathBuf {
        self.dir.join(format!("{}.d", NAMES[i]))
    }
    fn open(&mut self, i: usize) -> Host<'m> {
        let ads = self.adapters(i);
        Host::open_file_with(self.m, &self.path(i), ads).unwrap()
    }
    fn admit(&mut self, i: usize, ev: &str) {
        let src = self.src.clone();
        if let Some(h) = self.hosts[i].as_mut() {
            h.admit(&src, ev).unwrap_or_else(|e| panic!("seed {}: {} did not admit {ev}: {e}", self.seed, NAMES[i]));
        }
    }
    /// kill -9: the process is gone, D stays, maybe with a torn last line.
    fn kill_and_restart(&mut self, i: usize, torn: bool) {
        self.hosts[i] = None;
        let me = addr(NAMES[i]);
        self.net.borrow_mut().seen.remove(&me);
        if torn {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(self.path(i)).unwrap();
            f.write_all(b"{\"event\":{\"payload\":{\"torn").unwrap();
        }
        let before = self.net.borrow().sends.get(&me).copied().unwrap_or(0);
        let h = self.open(i);
        let after = self.net.borrow().sends.get(&me).copied().unwrap_or(0);
        assert_eq!(before, after, "seed {}: {} contacted peers while recovering from D", self.seed, NAMES[i]);
        self.hosts[i] = Some(h);
    }
}

fn field<'a>(h: &'a Host, name: &str) -> &'a V {
    &h.state()[0].1.iter().find(|(n, _, _)| n == name).unwrap().1
}
fn map(v: &V) -> BTreeMap<String, String> {
    match v {
        V::Map(kv) => kv.iter().map(|(k, v)| (format!("{k:?}"), format!("{v:?}"))).collect(),
        _ => BTreeMap::new(),
    }
}
fn int(v: &V) -> i128 {
    match v {
        V::Int(i, _) => *i,
        _ => -1,
    }
}

/// The replicated view of replica `h` as seen by the simulator: shared
/// conversations everywhere, and DMs of the pair (a, b) on a and b.
fn view(h: &Host, dm_pair: Option<&str>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for f in ["messages", "reactions", "edited", "deleted"] {
        for (k, v) in map(field(h, f)) {
            let is_dm = k.contains("dm:");
            let keep = match dm_pair {
                None => !is_dm,
                Some(p) => is_dm && k.contains(p),
            };
            if keep {
                out.insert(format!("{f} {k}"), v);
            }
        }
    }
    out
}

/// Run one seed; returns the final D files (for the determinism check).
fn run_seed(seed: u64) -> Vec<String> {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/chat.rce")).unwrap();
    let m = parse(&src).unwrap();
    let dir = std::env::temp_dir().join(format!("rce-dst-{}-{seed}-{:?}", std::process::id(), std::thread::current().id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let net = Rc::new(RefCell::new(Net {
        rng: Rng(seed),
        now: 0,
        seq: 0,
        f: Faults { drop_pct: 20, dup_pct: 10, max_delay: 300 },
        cut: BTreeSet::new(),
        down: BTreeSet::new(),
        inbox: BTreeMap::new(),
        seen: BTreeMap::new(),
        sends: BTreeMap::new(),
    }));
    let mut sim = Sim { m: &m, src: src.clone(), dir: dir.clone(), net: net.clone(), hosts: vec![], restarts: 0, seed };
    for i in 0..3 {
        let h = sim.open(i);
        sim.hosts.push(Some(h));
        let peers: Vec<String> = (0..3).filter(|&j| j != i).map(|j| format!("\"{}\"", addr(NAMES[j]))).collect();
        sim.admit(i, &format!("Setup(me: \"{}\", peers: [{}])", NAMES[i], peers.join(", ")));
    }
    let mut rng = Rng(seed ^ 0x5eed);
    let mut posts: Vec<(usize, u64)> = vec![]; // (author, clock) of own posts
    // faults
    for round in 0..160u64 {
        net.borrow_mut().now += 50;
        if rng.chance(8) {
            let (a, b) = (rng.below(3) as usize, rng.below(3) as usize);
            if a != b {
                let (x, y) = (addr(NAMES[a]), addr(NAMES[b]));
                let mut n = net.borrow_mut();
                if !n.cut.remove(&(x.clone(), y.clone())) {
                    n.cut.insert((x.clone(), y.clone()));
                    n.cut.insert((y, x));
                }
            }
        }
        if round > 20 && rng.chance(2) {
            let i = rng.below(3) as usize;
            let torn = rng.chance(50);
            sim.kill_and_restart(i, torn);
        }
        if round > 30 && rng.chance(25) {
            let i = rng.below(3) as usize;
            let clock = int(field(sim.hosts[i].as_ref().unwrap(), "clock")) as u64 + 1;
            match rng.below(5) {
                0 | 1 => {
                    sim.admit(i, &format!("Do(op: \"post\", conv: \"c\", topic: \"t\", body: \"m{round}\")"));
                    posts.push((i, clock));
                }
                2 if !posts.is_empty() => {
                    let (a, c) = posts[rng.below(posts.len() as u64) as usize];
                    let key = format!("c\\u{{1f}}t\\u{{1f}}{c:012}\\u{{1f}}{}", NAMES[a]);
                    sim.admit(i, &format!("Do(op: \"react\", conv: \"{key}\", topic: \"+1\", body: \"\")"));
                }
                3 if !posts.is_empty() => {
                    let (a, c) = posts[rng.below(posts.len() as u64) as usize];
                    let key = format!("c\\u{{1f}}t\\u{{1f}}{c:012}\\u{{1f}}{}", NAMES[a]);
                    let op = if rng.chance(70) { "edit" } else { "delete" };
                    sim.admit(a, &format!("Do(op: \"{op}\", conv: \"{key}\", topic: \"\", body: \"e{round}\")"));
                }
                _ => {
                    let j = (i + 1 + rng.below(2) as usize) % 3;
                    sim.admit(i, &format!("Do(op: \"dm\", conv: \"{}\", topic: \"\", body: \"d{round} ☕\")", NAMES[j]));
                }
            }
        }
        for i in 0..3 {
            sim.admit(i, "Poll");
        }
    }
    // heal
    {
        let mut n = net.borrow_mut();
        n.cut.clear();
        n.down.clear();
        n.f = Faults { drop_pct: 0, dup_pct: 0, max_delay: 0 };
    }
    let quiet = |s: &Sim| {
        s.hosts.iter().all(|h| {
            let h = h.as_ref().unwrap();
            map(field(h, "iwire")).is_empty() && map(field(h, "retry")).is_empty() && int(field(h, "jhead")) == int(field(h, "jtail"))
        })
    };
    let views = |s: &Sim| -> Vec<BTreeMap<String, String>> { s.hosts.iter().map(|h| view(h.as_ref().unwrap(), None)).collect() };
    let mut healed_at = None;
    for r in 0..1500 {
        net.borrow_mut().now += 50;
        for i in 0..3 {
            sim.admit(i, "Poll");
        }
        if quiet(&sim) && net.borrow().inbox.values().all(|q| q.is_empty()) {
            let v = views(&sim);
            if v[0] == v[1] && v[1] == v[2] {
                healed_at = Some(r);
                break;
            }
        }
    }
    let hs: Vec<&Host> = sim.hosts.iter().map(|h| h.as_ref().unwrap()).collect();
    for (i, h) in hs.iter().enumerate() {
        assert!(map(field(h, "iwire")).is_empty(), "seed {seed}: {} still has unconfirmed reliable ops: {:?}", NAMES[i], map(field(h, "iwire")).keys().collect::<Vec<_>>());
        assert_eq!(int(field(h, "rejected")), 0, "seed {seed}: {} rejected a message", NAMES[i]);
        let sessions = map(field(h, "sessions"));
        for j in 0..3 {
            if j != i {
                assert!(sessions.contains_key(&format!("{:?}", V::Text(NAMES[j].into()))), "seed {seed}: {} has no ratchet with {}", NAMES[i], NAMES[j]);
            }
        }
    }
    let healed_at = healed_at.unwrap_or_else(|| {
        let v = views(&sim);
        let keys: BTreeSet<&String> = v.iter().flat_map(|m| m.keys()).collect();
        let diff: Vec<_> = keys.into_iter().filter(|k| !(v[0].get(*k) == v[1].get(*k) && v[1].get(*k) == v[2].get(*k))).take(6).map(|k| (k.clone(), v[0].get(k).cloned(), v[1].get(k).cloned(), v[2].get(k).cloned())).collect();
        panic!("seed {seed}: replicas did not converge within 1500 rounds after healing; first differences: {diff:?}")
    });
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        let p = format!("dm:{},{}", NAMES[a], NAMES[b]);
        assert_eq!(view(hs[a], Some(&p)), view(hs[b], Some(&p)), "seed {seed}: DMs of {p} differ between its members");
    }
    // acks and retries stop once confirmed
    let before: u64 = net.borrow().sends.values().sum();
    for _ in 0..80 {
        net.borrow_mut().now += 50;
        for i in 0..3 {
            sim.admit(i, "Poll");
        }
    }
    let after: u64 = net.borrow().sends.values().sum();
    assert_eq!(before, after, "seed {seed}: reliable sends continued after every op was confirmed");
    // each D alone replays to the live state hash
    let lives: Vec<String> = sim.hosts.iter().map(|h| h.as_ref().unwrap().state_hash().to_string()).collect();
    sim.hosts.clear();
    let mut ds = vec![];
    for i in 0..3 {
        let text = std::fs::read_to_string(sim.path(i)).unwrap();
        assert_eq!(replay(&m, &text).unwrap().1, lives[i], "seed {seed}: {} D does not replay to its live hash", NAMES[i]);
        ds.push(text);
    }
    eprintln!("seed {seed}: converged {healed_at} rounds after healing; {} restarts", sim.restarts.saturating_sub(3));
    let _ = std::fs::remove_dir_all(&dir);
    ds
}

fn seeds() -> Vec<u64> {
    let n: u64 = std::env::var("RCE_DST_SEEDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    if let Ok(one) = std::env::var("RCE_DST_SEED") {
        return vec![one.parse().unwrap()];
    }
    let mut v: Vec<u64> = (1..=n).collect();
    v.extend_from_slice(REGRESSION);
    v.sort();
    v.dedup();
    v
}

#[test]
fn chat_over_faulty_channels_is_live_per_replica() {
    let ss = seeds();
    std::thread::scope(|s| {
        for h in ss.iter().map(|&seed| s.spawn(move || run_seed(seed))).collect::<Vec<_>>() {
            h.join().unwrap();
        }
    });
}

#[test]
fn a_seed_replays_exactly() {
    let a = run_seed(7);
    let b = run_seed(7);
    let strip = |d: &Vec<String>| d.iter().map(|t| t.replace(&std::env::temp_dir().to_string_lossy().to_string(), "")).collect::<Vec<_>>();
    assert!(strip(&a) == strip(&b), "seed 7: two runs wrote different D files");
}
