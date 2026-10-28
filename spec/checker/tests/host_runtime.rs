//! The host runtime of Appendix E (src/host.rs) against the reference:
//! its D agrees with `rce trace` (6.1 chain), replay with adapters
//! disconnected re-derives every state hash (HC-8, T37), verify on open cuts
//! a bad suffix (HS-4, T38) and a failed barrier stops the admitter (HS-3, T39).
use rce_spec_check::cli::trace;
use rce_spec_check::host::{Entry, FileStore, Host, MemStore, Store, replay, verify_prefix};
use rce_spec_check::parse::parse;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn ex(name: &str) -> (PathBuf, String) {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("examples").join(name);
    let s = std::fs::read_to_string(&p).unwrap();
    (p, s)
}

const EVS: &str = r#"Put(key: "b", val: "2"), Put(key: "a", val: "1"), Delete(key: "zz"), Scan(start: ""), Delete(key: "a")"#;

#[test]
fn program_entries_agree_with_reference_trace() {
    let (p, src) = ex("55_rcedb.rce");
    let m = parse(&src).unwrap();
    // the reference trace stops at the first kept outcome; use the applied prefix
    let evs = r#"Put(key: "b", val: "2"), Put(key: "a", val: "1"), Scan(start: ""), Delete(key: "a")"#;
    let t: Value = serde_json::from_str(&trace(&p.to_string_lossy(), &src, Some(evs)).unwrap()).unwrap();
    let want: Vec<String> = t["traces"][0]["steps"].as_array().unwrap().iter().map(|s| s["state_hash"].as_str().unwrap().to_string()).collect();
    let want_ev: Vec<String> = t["traces"][0]["steps"].as_array().unwrap().iter().map(|s| s["event"].to_string()).collect();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&src, evs).unwrap();
    drop(h);
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    let prog: Vec<&Entry> = es.iter().filter(|e| !e.is_host()).collect();
    assert_eq!(prog.iter().map(|e| e.state_hash.clone()).collect::<Vec<_>>(), want);
    let got_ev: Vec<String> = prog.iter().map(|e| serde_json::from_str::<Value>(&e.event).unwrap().to_string()).collect();
    assert_eq!(got_ev, want_ev);
}

#[test]
fn host_entries_leave_state_unchanged_and_replay_matches_live() {
    let (_, src) = ex("55_rcedb.rce");
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let mut live = vec![];
    for b in EVS.split("), ") {
        let b = if b.ends_with(')') { b.to_string() } else { format!("{b})") };
        h.admit(&src, &b).unwrap();
        h.wall_time(1_760_000_000_000_000_000).unwrap();
        live.push((h.len(), h.state_hash().to_string()));
    }
    drop(h);
    let es: Vec<Entry> = st.lines.iter().map(|l| Entry::parse_line(l).unwrap()).collect();
    for w in es.windows(2) {
        if w[1].is_host() {
            assert_eq!(w[1].state_hash, w[0].state_hash, "host entry {} changed state", w[1].seq);
        }
    }
    // T37: replay of every prefix gives the live state after that entry
    for (n, hash) in live {
        let text: String = st.lines[..n as usize].iter().map(|l| format!("{l}\n")).collect();
        let (k, h2, _) = replay(&m, &text).unwrap();
        assert_eq!((k, h2), (n, hash));
    }
}

#[test]
fn verify_on_open_cuts_and_logs_truncated() {
    let (_, src) = ex("55_rcedb.rce");
    let m = parse(&src).unwrap();
    let dir = std::env::temp_dir().join(format!("rce-host-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (name, damage) in [("torn", 0), ("flip", 1), ("swap", 2)] {
        let p = dir.join(format!("{name}.log"));
        let _ = std::fs::remove_file(&p);
        {
            let mut h = Host::open_file(&m, &p).unwrap();
            h.admit(&src, EVS).unwrap();
        }
        let text = std::fs::read_to_string(&p).unwrap();
        let mut lines: Vec<String> = text.lines().map(String::from).collect();
        let n = lines.len();
        let good = match damage {
            0 => {
                lines.push("{\"event\":{\"payl".into());
                n
            }
            1 => {
                let l = &mut lines[n - 3];
                *l = l.replacen("\"key\":\"a\"", "\"key\":\"A\"", 1);
                n - 3
            }
            _ => {
                lines.swap(n - 2, n - 1);
                n - 2
            }
        };
        let mut t = lines.join("\n");
        if damage != 0 {
            t.push('\n');
        }
        std::fs::write(&p, &t).unwrap();
        let refs: Vec<&str> = t.split_terminator('\n').collect();
        assert_eq!(verify_prefix(&refs, &m), good, "{name}");
        let h = Host::open_file(&m, &p).unwrap();
        drop(h);
        let after = std::fs::read_to_string(&p).unwrap();
        assert!(after.contains(&format!("\"dropped\":{}", refs.len() - good)), "{name}: {after}");
        replay(&m, &after).unwrap();
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn failed_barrier_stops_admission() {
    let (_, src) = ex("55_rcedb.rce");
    let m = parse(&src).unwrap();
    let mut st = MemStore { fail_barrier_at: Some(2), ..Default::default() };
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    let err = h.admit(&src, r#"Put(key: "x", val: "1")"#).unwrap_err();
    assert!(err.contains("StorageFault"), "{err}");
    assert!(h.stopped());
    assert!(h.admit(&src, r#"Put(key: "y", val: "2")"#).is_err());
    assert!(h.wall_time(1).is_err());
    let n = h.len();
    drop(h);
    assert_eq!(st.lines.len() as u64, n, "nothing appended after the failed barrier");
}

#[test]
fn file_store_reports_its_e3_row() {
    let p = std::env::temp_dir().join(format!("rce-host-caps-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&p);
    let (s, _) = FileStore::open(&p).unwrap();
    let row = s.caps();
    if cfg!(target_vendor = "apple") {
        assert_eq!(row, "macOS, iOS");
    } else if cfg!(target_os = "linux") {
        assert_eq!(row, "Linux");
    }
    let _ = std::fs::remove_file(&p);
}

#[test]
fn cli_host_run_then_replay() {
    let (p, _) = ex("55_rcedb.rce");
    let log = std::env::temp_dir().join(format!("rce-host-cli-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let run = |a: &[&str]| {
        let a: Vec<String> = a.iter().map(|s| s.to_string()).collect();
        let (mut o, mut e) = (vec![], vec![]);
        let c = rce_spec_check::cli::main(&a, &mut std::io::empty(), &mut o, &mut e);
        (c, String::from_utf8(o).unwrap())
    };
    let f = p.to_string_lossy();
    let l = log.to_string_lossy();
    let (c, o) = run(&["host", "run", &f, "--log", &l, r#"Put(key: "a", val: "1"), Scan(start: "")"#]);
    assert_eq!(c, 0, "{o}");
    assert!(o.contains("emit Row(key: \"a\", val: \"1\")"), "{o}");
    let (c, o) = run(&["host", "replay", &f, "--log", &l]);
    assert_eq!(c, 0, "{o}");
    assert!(o.starts_with("replay ok"), "{o}");
    // --redact prints a field as a short digest, never its value
    let (c, r) = run(&["host", "replay", &f, "--log", &l, "--redact", "rows"]);
    assert_eq!(c, 0, "{r}");
    assert_eq!(o.lines().next(), r.lines().next(), "same replay hash");
    assert!(r.contains("rows = <redacted sha256:"), "{r}");
    assert!(!r.lines().nth(1).unwrap().contains("\"a\": \"1\""), "{r}");
    let (_, r2) = run(&["host", "replay", &f, "--log", &l, "--redact", "rows"]);
    assert_eq!(r, r2, "the digest is deterministic");
    let _ = std::fs::remove_file(&log);
}

#[test]
fn record_variant_map_list_option_payloads_replay() {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/records.rce");
    let src = std::fs::read_to_string(&p).unwrap();
    let m = parse(&src).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    h.admit(&src, r#"Add(p: Point(x: 1.5, y: -2.0), tags: ["a", "b"], meta: {"k": 3, "j": 1}, note: none, s: Circle(r: 2.0)), Add(p: Point(x: 0.1, y: 0.2), tags: [], meta: {}, note: "hi", s: Square(side: 1.0))"#).unwrap();
    let live = h.state_hash().to_string();
    let n = h.len();
    drop(h);
    let text: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert!(text.contains("\"$v\":\"Circle\"") && text.contains("[\"j\",1]"), "{text}");
    let (k, hash, _) = replay(&m, &text).unwrap();
    assert_eq!((k, hash), (n, live));
}
