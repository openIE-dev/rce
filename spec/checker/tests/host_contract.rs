//! Appendix E, the host contract: conformance tests for the rules that a
//! spec-level check can decide without a host runtime. The small
//! reference functions here (verify on open, a host-entry fold, an
//! admitter, placement) are the normative text of E.3 to E.6 written as
//! code; a host runtime must agree with them. Lean: RCE/Host.lean (T37,
//! T38, T39).
use rce_spec_check::cli::{self, trace};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

fn spec() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}
fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}
/// 6.1: the hash of an entry is the SHA-256 of the canonical JSON of
/// {event, prev_hash, seq, state_hash} (serde_json sorts keys; floats and text in the canonical writer of 6.1.1),
/// where `event` is the hashed, positional form of the event (T7). An entry
/// of this model carries the readable event under `event` and, for a
/// program event, the hashed form under `hashed`; a host entry is hashed
/// as stored.
fn entry_hash(e: &Value) -> String {
    let hashed = e.get("hashed").filter(|h| !h.is_null()).unwrap_or(&e["event"]);
    sha(&rce_spec_check::interp::canon_json(&json!({"event": hashed, "prev_hash": e["prev_hash"], "seq": e["seq"], "state_hash": e["state_hash"]})))
}
fn entry(seq: usize, event: Value, prev: &str, state_hash: &str) -> Value {
    entry_hashed(seq, event, None, prev, state_hash)
}
/// An entry of a program event: `hashed` is its positional form (T7).
fn entry_hashed(seq: usize, event: Value, hashed: Option<Value>, prev: &str, state_hash: &str) -> Value {
    let mut e = json!({"seq": seq, "event": event, "prev_hash": prev, "state_hash": state_hash});
    if let Some(h) = hashed {
        e["hashed"] = h;
    }
    let h = entry_hash(&e);
    e["hash"] = Value::String(h);
    e
}

/// HS-4: the length of the longest prefix of D (one JSON entry per line)
/// that links from the genesis hash. A line that is not JSON (a torn
/// write) ends the prefix.
fn verify_open(lines: &[String]) -> usize {
    let mut prev = GENESIS.to_string();
    for (i, l) in lines.iter().enumerate() {
        let Ok(e) = serde_json::from_str::<Value>(l) else { return i };
        if e["seq"] != i || e["prev_hash"] != prev.as_str() || e["hash"] != entry_hash(&e).as_str() {
            return i;
        }
        prev = e["hash"].as_str().unwrap().to_string();
    }
    lines.len()
}

/// Every example trace: (file, steps as canonical entries).
fn logs() -> Vec<(String, Vec<Value>)> {
    let mut fs: Vec<PathBuf> = std::fs::read_dir(spec().join("examples"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    fs.sort();
    let mut out = vec![];
    for f in fs {
        let src = std::fs::read_to_string(&f).unwrap();
        let Ok(t) = trace(&f.to_string_lossy(), &src, None) else { continue };
        let t: Value = serde_json::from_str(&t).unwrap();
        for tr in t["traces"].as_array().unwrap() {
            let steps: Vec<Value> = tr["steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| json!({"seq": s["seq"], "event": s["event"], "hashed": s["hashed"]["event"], "prev_hash": s["prev_hash"], "state_hash": s["state_hash"], "hash": s["hash"]}))
                .collect();
            if !steps.is_empty() {
                out.push((f.file_name().unwrap().to_string_lossy().into_owned(), steps));
            }
        }
    }
    out
}

#[test]
fn verify_on_open_cuts_bad_suffix() {
    let mut checked = 0;
    for (f, d) in logs() {
        let lines: Vec<String> = d.iter().map(|e| e.to_string()).collect();
        let n = lines.len();
        assert_eq!(verify_open(&lines), n, "{f}: an intact log is kept whole (T38)");
        // Torn tail: the last write stopped half way.
        let mut torn = lines.clone();
        let l = torn[n - 1].len();
        torn[n - 1].truncate(l / 2);
        assert_eq!(verify_open(&torn), n - 1, "{f}: torn tail");
        // A flipped byte in entry k (silent corruption).
        for k in 0..n {
            let mut bad = lines.clone();
            let mut e: Value = serde_json::from_str(&bad[k]).unwrap();
            let s = e["state_hash"].as_str().unwrap();
            let flip = if s.starts_with('0') { "1" } else { "0" };
            e["state_hash"] = Value::String(format!("{flip}{}", &s[1..]));
            bad[k] = e.to_string();
            assert_eq!(verify_open(&bad), k, "{f}: flipped entry {k}");
        }
        // A misdirected write: two entries swapped.
        if n >= 2 {
            let mut sw = lines.clone();
            sw.swap(n - 2, n - 1);
            assert_eq!(verify_open(&sw), n - 2, "{f}: swapped pair");
        }
        checked += 1;
    }
    assert!(checked >= 20, "only {checked} logs checked");
}

/// Payload JSON back to RCE event syntax, for scalar payloads.
fn lit(v: &Value) -> Option<String> {
    Some(match v {
        Value::Null => "()".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) if n.is_i64() || n.is_u64() => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap(),
        _ => return None,
    })
}
fn event_text(e: &Value) -> Option<String> {
    let p = e["payload"].as_object()?;
    if p.is_empty() {
        return Some(e["type"].as_str()?.to_string());
    }
    let fs: Option<Vec<String>> = p.iter().map(|(k, v)| lit(v).map(|x| format!("{k}: {x}"))).collect();
    Some(format!("{}({})", e["type"].as_str()?, fs?.join(", ")))
}

/// A host entry; δ is the identity on it, so its state hash is the
/// previous entry's (E.1).
fn host(kind: &str, payload: Value) -> Value {
    json!({"type": format!("host.{kind}"), "payload": payload})
}
fn is_host(e: &Value) -> bool {
    e["event"]["type"].as_str().is_some_and(|t| t.starts_with("host."))
}
/// The adapter side of a live run: interleave host entries into a
/// program log (storage and fabric entries after the first step, a
/// `host.Synced` after every step) and rebuild the chain.
fn interleave(d: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = vec![];
    let mut prev = GENESIS.to_string();
    for (i, e) in d.iter().enumerate() {
        let st = e["state_hash"].as_str().unwrap().to_string();
        let mut evs = vec![(e["event"].clone(), Some(e["hashed"].clone()))];
        if i == 0 {
            evs.push((host("StorageCaps", json!({"barrier": "volatile"})), None));
            evs.push((host("FabricCaps", json!({"counters": [], "cpu": ["sse2"], "gpus": [], "memory": [], "npus": []})), None));
        }
        evs.push((host("Synced", json!({"upto": out.len()})), None));
        for (ev, hashed) in evs {
            let x = entry_hashed(out.len(), ev, hashed, &prev, &st);
            prev = x["hash"].as_str().unwrap().to_string();
            out.push(x);
        }
    }
    out
}

#[test]
fn foreign_and_host_entries_replay() {
    let mut replayed = 0;
    for (f, d) in logs() {
        let path = spec().join("examples").join(&f);
        let src = std::fs::read_to_string(&path).unwrap();
        let evs: Option<Vec<String>> = d.iter().map(|e| event_text(&e["event"])).collect();
        let Some(evs) = evs else { continue };
        let live = interleave(&d);
        let lines: Vec<String> = live.iter().map(|e| e.to_string()).collect();
        assert_eq!(verify_open(&lines), live.len(), "{f}: the interleaved log is a chain");
        // Replay with adapters disconnected (HC-8): drop host entries (δ is
        // the identity on them) and re-run the program events through the
        // reference; every state hash must equal the live log's.
        let prog: Vec<&Value> = live.iter().filter(|e| !is_host(e)).collect();
        let Ok(t) = trace(&path.to_string_lossy(), &src, Some(&evs.join(", "))) else { continue };
        let t: Value = serde_json::from_str(&t).unwrap();
        let steps = t["traces"][0]["steps"].as_array().unwrap();
        assert_eq!(steps.len(), prog.len(), "{f}: replay admits every logged event");
        for (a, b) in steps.iter().zip(&prog) {
            assert_eq!(a["state_hash"], b["state_hash"], "{f}: replay state differs (T37)");
            assert_eq!(a["event"], b["event"]);
        }
        // Host entries leave the state unchanged: each carries its
        // predecessor's state hash.
        for w in live.windows(2) {
            if is_host(&w[1]) {
                assert_eq!(w[1]["state_hash"], w[0]["state_hash"], "{f}: host entry changed state");
            }
        }
        replayed += 1;
    }
    assert!(replayed >= 20, "only {replayed} logs replayed");
}

/// HC-12: placement from the host entries of D only. Each op class goes
/// to the fabric with the least overhead + work / throughput among the
/// CPU and the fabrics with a matching `host.Exact` entry.
fn place(d: &[Value], class: &str, work: f64) -> String {
    let mut best = ("cpu".to_string(), 16.0 + work / 1.0);
    for e in d.iter().filter(|e| e["event"]["type"] == "host.Calibrated") {
        let p = &e["event"]["payload"];
        if p["class"] != class {
            continue;
        }
        let fab = p["fabric"].as_str().unwrap();
        let exact = d.iter().any(|x| {
            x["event"]["type"] == "host.Exact"
                && x["event"]["payload"]["fabric"] == fab
                && x["event"]["payload"]["class"] == class
                && x["event"]["payload"]["probe"] == x["event"]["payload"]["reference"]
        });
        let t = p["overhead"].as_f64().unwrap() + work / p["throughput"].as_f64().unwrap();
        if exact && t < best.1 {
            best = (fab.to_string(), t);
        }
    }
    best.0
}

#[test]
fn placement_is_a_function_of_host_entries() {
    let cal = |fab: &str, o: f64, th: f64| host("Calibrated", json!({"fabric": fab, "class": "matmul", "overhead": o, "throughput": th}));
    let exact = |fab: &str, ok: bool| host("Exact", json!({"fabric": fab, "class": "matmul", "probe": "ab", "reference": if ok {"ab"} else {"cd"}}));
    let mk = |evs: Vec<Value>| -> Vec<Value> {
        let mut prev = GENESIS.to_string();
        evs.into_iter()
            .enumerate()
            .map(|(i, ev)| {
                let e = entry(i, ev, &prev, GENESIS);
                prev = e["hash"].as_str().unwrap().to_string();
                e
            })
            .collect()
    };
    let d = mk(vec![cal("gpu0", 290.0, 400.0), exact("gpu0", true), cal("npu0", 500.0, 700.0), exact("npu0", false)]);
    assert_eq!(place(&d, "matmul", 10.0), "cpu", "small work stays on the CPU");
    assert_eq!(place(&d, "matmul", 1e6), "gpu0", "large work goes to the exact GPU");
    // The NPU is faster but not proven exact: never chosen for state (HC-13).
    assert_ne!(place(&d, "matmul", 1e9), "npu0");
    // Replay reads the same entries, so the same placement; program
    // events in between change nothing.
    let mut with_prog = d.clone();
    with_prog.insert(1, json!({"event": {"type": "Inc", "payload": {}}}));
    for w in [10.0, 1e6, 1e9] {
        assert_eq!(place(&d, "matmul", w), place(&with_prog, "matmul", w));
    }
}

/// E.2 HC-1, HC-9: the reference admitter. Inputs pending at one
/// admission are ordered by priority (highest first, T25), then arrival,
/// and appended to D in that order.
fn admit(pending: &[(u8, usize, &str)]) -> Vec<Value> {
    let mut p = pending.to_vec();
    p.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    let mut prev = GENESIS.to_string();
    p.iter()
        .enumerate()
        .map(|(i, (pr, _, name))| {
            let e = entry(i, json!({"type": name, "payload": {"priority": pr}}), &prev, GENESIS);
            prev = e["hash"].as_str().unwrap().to_string();
            e
        })
        .collect()
}

#[test]
fn admitter_order_is_d_order() {
    let d = admit(&[(0, 0, "Key"), (200, 1, "Brake"), (0, 2, "Tick"), (200, 3, "Alarm")]);
    let order: Vec<&str> = d.iter().map(|e| e["event"]["type"].as_str().unwrap()).collect();
    assert_eq!(order, ["Brake", "Alarm", "Key", "Tick"]);
    let lines: Vec<String> = d.iter().map(|e| e.to_string()).collect();
    assert_eq!(verify_open(&lines), 4);
    // The same pending set in another arrival interleaving at equal
    // priority keeps arrival order: the order is a function of the input.
    let again = admit(&[(200, 1, "Brake"), (0, 0, "Key"), (200, 3, "Alarm"), (0, 2, "Tick")]);
    assert_eq!(d, again);
}

/// HS-3: an admitter whose barrier fails at entry k stops; nothing after
/// it is appended, and no retry is made.
struct Admitter {
    log: Vec<String>,
    halted: bool,
    barrier_calls: usize,
}
impl Admitter {
    fn append(&mut self, line: String, sync: &mut dyn FnMut(usize) -> bool) -> bool {
        if self.halted {
            return false;
        }
        self.barrier_calls += 1;
        if !sync(self.log.len()) {
            self.halted = true;
            return false;
        }
        self.log.push(line);
        true
    }
}

#[test]
fn fatal_barrier_stops_admission() {
    for k in 0..5 {
        let mut a = Admitter { log: vec![], halted: false, barrier_calls: 0 };
        let mut sync = |i: usize| i != k;
        for i in 0..8 {
            a.append(format!("e{i}"), &mut sync);
        }
        assert_eq!(a.log.len(), k, "entries after a failed barrier are never appended (T39)");
        assert!(a.halted);
        assert_eq!(a.barrier_calls, k + 1, "a failed barrier is never retried");
    }
}

fn run(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let c = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (c, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

#[test]
fn capability_examples_check() {
    // HC-5: authority is a capability carried by a logged event (5.16).
    let f = spec().join("examples/53_capability.rce");
    let (c, o, e) = run(&["check", f.to_str().unwrap()]);
    assert_eq!(c, 0, "{o}{e}");
}

#[test]
fn rcedb_example_runs() {
    // E.8: RCEdb in RCE. Scans are in canonical key order; Synced (a bound
    // host result) moves the durable mark and never lowers it.
    let f = spec().join("examples/55_rcedb.rce");
    let p = f.to_str().unwrap();
    let (c, o, e) = run(&["test", p]);
    assert_eq!(c, 0, "{o}{e}");
    let puts = "Put(key: \"b\", val: \"2\"), Put(key: \"c\", val: \"3\"), Put(key: \"a\", val: \"1\")";
    let (c, o, e) = run(&["run", p, &format!("{puts}, Synced(upto: 3), Synced(upto: 1), Scan(start: \"\")")]);
    assert_eq!(c, 0, "{o}{e}");
    let rows: Vec<&str> = o.lines().filter(|l| l.contains("emit Row")).collect();
    assert_eq!(rows.len(), 3, "{o}");
    assert!(rows[0].contains("\"a\"") && rows[1].contains("\"b\"") && rows[2].contains("\"c\""), "scan is in canonical key order, not insertion order: {o}");
    assert!(o.contains("durable = 3") && !o.contains("durable = 1"), "the durable mark never falls: {o}");
    let (c, o, _) = run(&["run", p, &format!("{puts}, Scan(start: \"b\")")]);
    assert_eq!(c, 0);
    assert_eq!(o.lines().filter(|l| l.contains("emit Row")).count(), 2, "scan starts at `start`: {o}");
}

/// Whether a rule cell of the E.10 table (`HC-10a, HD-3` or `HC-24 to
/// HC-26, HD-2`) covers the rule `r`.
fn covers(cell: &str, r: &str) -> bool {
    let num = |x: &str| x.rsplit_once('-').and_then(|(p, n)| Some((p.to_string(), n.parse::<u32>().ok()?)));
    cell.split(", ").any(|c| match c.split_once(" to ") {
        None => c == r,
        Some((a, b)) => matches!((num(a), num(b), num(r)), (Some((p, lo)), Some((_, hi)), Some((q, k))) if p == q && lo <= k && k <= hi),
    })
}

#[test]
fn every_appendix_e_rule_is_cited_by_a_test() {
    let book = std::fs::read_to_string(spec().join("book/16-host-contract.md")).unwrap();
    let me = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host_contract.rs")).unwrap();
    for name in ["verify_on_open_cuts_bad_suffix", "foreign_and_host_entries_replay", "placement_is_a_function_of_host_entries", "admitter_order_is_d_order", "fatal_barrier_stops_admission", "capability_examples_check", "rcedb_example_runs"] {
        assert!(book.contains(&format!("`{name}`")), "Appendix E does not cite {name}");
        assert!(me.contains(&format!("fn {name}()")), "no test {name}");
    }
    for r in ["HC-1", "HC-2", "HC-3", "HC-4", "HC-5", "HC-6", "HC-7", "HC-8", "HC-9", "HC-10", "HC-11", "HC-12", "HC-13", "HC-14", "HS-1", "HS-2", "HS-3", "HS-4", "HS-5", "HS-6", "HS-7", "HS-8", "HS-9"] {
        assert!(book.contains(&format!("**{r}")), "rule {r} is not stated");
    }
    // the host rules of E.4a to E.4f and HC-5a: each is stated, has a row in
    // the table of E.10, and every test that row names exists
    let tests = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let e10 = &book[book.find("### E.10").unwrap()..book.find("### E.11").unwrap()];
    let rows: Vec<&str> = e10.lines().filter(|l| l.starts_with("| HC-") || l.starts_with("| HS-")).collect();
    for r in ["HC-5a", "HC-10a", "HC-10b", "HC-21", "HC-22", "HC-23", "HC-24", "HC-25", "HC-26", "HC-27", "HC-28", "HC-29", "HD-1", "HD-2", "HD-3", "HD-4"] {
        assert!(book.contains(&format!("**{r} ")) || book.contains(&format!("**{r}**")), "rule {r} is not stated");
        let row = rows.iter().find(|l| covers(l.split('|').nth(1).unwrap_or("").trim(), r)).unwrap_or_else(|| panic!("no E.10 row for {r}"));
        for part in row.split('|').nth(3).unwrap().split("; ") {
            let Some((file, names)) = part.trim().split_once("`: ") else { continue };
            let src = std::fs::read_to_string(tests.join(file.trim_start_matches("`tests/"))).unwrap_or_else(|e| panic!("{file}: {e}"));
            for name in names.split('`').skip(1).step_by(2) {
                assert!(src.contains(&format!("fn {name}()")), "{r}: no test {name} in {file}");
            }
        }
    }
}

#[test]
fn proposed_rules_are_marked_proposed() {
    let book = std::fs::read_to_string(spec().join("book/16-host-contract.md")).unwrap();
    let e11 = &book[book.find("### E.11").expect("E.11 present")..];
    // Rules adopted into the normative text (N-1 and N-8 became 6.3 N3)
    // stay in E.11 as a record, marked with where they were adopted.
    for r in ["N-1", "N-8"] {
        assert!(e11.contains(&format!("**{r} (adopted as ")), "{r} is not recorded as adopted in E.11");
        assert!(!e11.contains(&format!("**{r} (proposed).**")), "{r} is adopted but E.11 still calls it proposed");
    }
    for r in ["N-9", "K-1", "K-2", "K-3", "K-4", "K-5", "K-6", "K-7", "K-8", "K-9", "E-x", "S-1", "S-2", "S-3", "S-4", "C-3", "V-6", "TU-1", "TU-2", "TU-3", "TU-4", "TU-5", "TU-6", "G-1", "G-2", "G-3", "G-4", "G-5", "G-6"] {
        assert!(e11.contains(&format!("**{r} (proposed).**")), "{r} is not stated as proposed in E.11");
    }
    // N-1 and N-8 were proposed once and are adopted in 6.3 N3: they must say so.
    for r in ["N-1", "N-8"] {
        assert!(e11.contains(&format!("**{r} (adopted as 6.3 N3")), "{r} is not stated as adopted in E.11");
    }
    let normative = &book[..book.find("### E.11").unwrap()];
    assert!(!normative.contains("(proposed)"), "a proposed rule leaked into the normative sections");
}
