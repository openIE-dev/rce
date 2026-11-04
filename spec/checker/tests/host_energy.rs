//! HC-38: a package energy counter reading in D (`host.Energy`), in the D.11
//! form: written by the host when the grant `energy` is given and a counter
//! can be read, checked by replay where D is read, and never read by δ.
use rce_energy::reading::{Env, Sampling, fixtures::zone};
use rce_spec_check::host::{ENERGY_FIELDS, Entry, Host, energy_check, replay_report};
use rce_spec_check::parse::parse;
use serde_json::{Value, json};
use std::path::Path;
use std::process::Command;
use std::time::Duration;

const SRC: &str = include_str!("../../examples/01_counter.rce");

/// A fresh directory for one test.
fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rce-host-energy-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A run of three inputs with a reading of a fake powercap tree recorded
/// over its entries; the text of D, and the state hash of the run.
fn run_with_reading(dir: &Path) -> (String, String) {
    let m = parse(SRC).unwrap();
    let tree = dir.join("powercap");
    zone(&tree, "intel-rapl:0", "package-0", 1_000_000, 1_000_000_000);
    let env = Env {
        powercap: tree.clone(),
        machine: Some("test-host; cpu Test CPU; kernel Linux 0.0".into()),
        period: Duration::from_secs(3600),
    };
    let log = dir.join("d");
    let mut h = Host::open_file_with(&m, &log, vec![]).unwrap();
    let s = Sampling::start_as(&env, "rce host").unwrap();
    let first = h.len();
    h.admit(SRC, "Inc, Inc, Dec").unwrap();
    // 2.5 J at the package counter over the run
    std::fs::write(tree.join("intel-rapl:0/energy_uj"), "3500000").unwrap();
    let r = s.finish().unwrap();
    h.energy("RAPL package counter, whole package", &r.term, first)
        .unwrap();
    let live = h.state_hash().to_string();
    drop(h);
    (std::fs::read_to_string(&log).unwrap(), live)
}

fn energy_entries(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| {
            (v["event"]["type"] == "host.Energy").then(|| v["event"]["payload"].clone())
        })
        .collect()
}

#[test]
fn a_reading_is_recorded_in_the_d11_form_and_replays() {
    let t = tmp("form");
    let (text, live) = run_with_reading(&t);
    let es = energy_entries(&text);
    assert_eq!(es.len(), 1, "{text}");
    let p = &es[0];
    let keys: Vec<&str> = p.as_object().unwrap().keys().map(String::as_str).collect();
    assert_eq!(keys, ENERGY_FIELDS);
    assert_eq!(p["basis"], "reported");
    assert!((p["joules"].as_f64().unwrap() - 2.5).abs() < 1e-9, "{p}");
    assert_eq!(p["amortization"], json!({"kind": "per_call"}));
    assert_eq!(p["source"], "RAPL package counter, whole package");
    assert_eq!(
        p["provenance"]["sampler"],
        "rce host: Linux powercap RAPL energy_uj"
    );
    assert_eq!(p["provenance"]["boundary"], "package");
    assert_eq!(
        p["provenance"]["machine"],
        "test-host; cpu Test CPU; kernel Linux 0.0"
    );
    // the reading covers the run's entries, and is the entry after them
    let (first, last) = (
        p["entries"][0].as_u64().unwrap(),
        p["entries"][1].as_u64().unwrap(),
    );
    assert!(first <= last);
    // replay accepts it and gives the live state: δ never reads it
    let m = parse(SRC).unwrap();
    let r = replay_report(&m, &text).unwrap();
    assert_eq!(r.state_hash, live);
    // the same inputs with no reading give the same state
    let t2 = tmp("plain");
    let mut h = Host::open_file_with(&m, &t2.join("d"), vec![]).unwrap();
    h.admit(SRC, "Inc, Inc, Dec").unwrap();
    assert_eq!(h.state_hash(), live);
}

#[test]
fn replay_refuses_an_entry_that_is_not_a_reading() {
    let t = tmp("refuse");
    let (text, _) = run_with_reading(&t);
    let p = energy_entries(&text).remove(0);
    let seq = p["entries"][1].as_u64().unwrap() + 1;
    assert!(energy_check(&p, seq).is_ok());
    let mut cases: Vec<(&str, Value)> = vec![];
    let mut with = |what: &'static str, f: &dyn Fn(&mut Value)| {
        let mut q = p.clone();
        f(&mut q);
        cases.push((what, q));
    };
    with("an estimate", &|q| q["basis"] = json!("est"));
    with("no basis", &|q| {
        q.as_object_mut().unwrap().remove("basis");
    });
    with("a negative figure", &|q| q["joules"] = json!(-1.0));
    with("a provenance without wraps", &|q| {
        q["provenance"].as_object_mut().unwrap().remove("wraps");
    });
    with("an unknown provenance field", &|q| {
        q["provenance"]["extra"] = json!(1)
    });
    with("an amortization outside rule 3", &|q| {
        q["amortization"] = json!({"kind": "per_hour"})
    });
    with("a batch mean of 0", &|q| {
        q["amortization"] = json!({"kind": "batch_mean", "n": 0})
    });
    with("entries after the reading", &|q| {
        q["entries"] = json!([0, 99])
    });
    with("an extra member", &|q| q["note"] = json!("x"));
    with("an empty source", &|q| q["source"] = json!(""));
    for (what, q) in &cases {
        let e = energy_check(q, seq).expect_err(what);
        assert!(e.contains("is not a reading (D.11)"), "{what}: {e}");
    }
    // and replay of a D whose chain is intact but whose reading is an
    // estimate refuses it at the reading's seq: D is checked where it is read
    let m = parse(SRC).unwrap();
    let bad = rechain(
        &m,
        &text.replace("\"basis\":\"reported\"", "\"basis\":\"est\""),
    );
    assert_ne!(bad, text, "the basis was rewritten");
    let e = replay_report(&m, &bad).expect_err("replay refuses the entry");
    assert!(
        e.contains(&format!("host.Energy at seq {seq} is not a reading (D.11)")),
        "{e}"
    );
}

/// The lines of D with every hash and link recomputed, as a writer that
/// produced the edited entries would have chained them.
fn rechain(m: &rce_spec_check::ast::Module, text: &str) -> String {
    let mut prev = "0".repeat(64);
    let mut out = String::new();
    for l in text.lines() {
        let e = Entry::parse_line(l).unwrap();
        let n = Entry::new(m, e.seq, e.event, &prev, &e.state_hash);
        prev = n.hash.clone();
        out.push_str(&n.line());
        out.push('\n');
    }
    out
}

#[test]
fn without_a_counter_the_host_says_why_and_writes_no_reading() {
    // with the grant and no readable counter (as on a CI runner, where RAPL is
    // root-only or absent), the host names the reason; without the grant, it
    // reads nothing
    let t = tmp("cli");
    let f = t.join("c.rce");
    std::fs::write(&f, SRC).unwrap();
    let run = |grant: bool, d: &str| {
        let mut a = vec!["host", "run"];
        if grant {
            a.push("energy");
        }
        let d = t.join(d);
        let (fs, ds) = (
            f.to_str().unwrap().to_string(),
            d.to_str().unwrap().to_string(),
        );
        let o = Command::new(env!("CARGO_BIN_EXE_rce"))
            .args(&a)
            .args([fs.as_str(), ds.as_str(), "Inc, Inc"])
            .output()
            .unwrap();
        (
            o.status.code(),
            String::from_utf8_lossy(&o.stderr).into_owned(),
            std::fs::read_to_string(&d).unwrap_or_default(),
        )
    };
    let (c, err, text) = run(true, "d1");
    assert_eq!(c, Some(0), "{err}");
    let n = energy_entries(&text).len();
    if n == 0 {
        assert!(err.contains("rce host: energy: unmeasured: "), "{err}");
    } else {
        // a machine whose counter can be read: one reading, and replay took it
        assert_eq!(n, 1, "{text}");
        assert!(err.contains("recorded as host.Energy at seq"), "{err}");
    }
    let (c, err, text) = run(false, "d2");
    assert_eq!(c, Some(0), "{err}");
    assert!(
        energy_entries(&text).is_empty() && !err.contains("energy"),
        "{err}"
    );
}
