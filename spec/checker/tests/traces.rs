//! Reference traces (spec 10.9, informative): `rce trace --json` over the
//! example corpus equals the committed fixtures in spec/traces/, every
//! step's hash chain is the 6.1 chain, and the differential runner
//! `spec/traces/rce-trace-diff.py` reports exact match on equal traces
//! and the first divergence otherwise. Regenerate the fixtures with
//! `RCE_BLESS_TRACES=1 cargo test --test traces`.
use rce_spec_check::cli::trace;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

fn spec() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

/// Event lists traced in addition to the examples: the counter on a
/// longer list, and example 16 (two regions, no `example` lines).
const EVENT_TRACES: &[(&str, &str)] = &[
    ("01_counter", "Inc, Inc, Dec, Inc, Dec, Dec, Dec"),
    ("16_multi_region", "AddItem(name: \"milk\"), AddItem(name: \"eggs\"), RemoveItem(name: \"milk\"), AddItem(name: \"tea\"), Reset, AddItem(name: \"jam\")"),
];

/// (fixture name, trace) for the corpus: every top-level example (the
/// multi-module example 26 is the package `examples/26_use_module/`,
/// which has no state machine), plus the event-list traces.
fn corpus() -> (Vec<(String, String)>, Vec<String>) {
    let ex = spec().join("examples");
    let mut fs: Vec<PathBuf> = std::fs::read_dir(&ex)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    fs.sort();
    let (mut out, mut skipped) = (vec![], vec![]);
    for f in &fs {
        let src = std::fs::read_to_string(f).unwrap();
        let stem = f.file_stem().unwrap().to_string_lossy().into_owned();
        match trace(&f.to_string_lossy(), &src, None) {
            Ok(t) => out.push((format!("{stem}.json"), t + "\n")),
            Err(_) => skipped.push(stem),
        }
    }
    for (stem, evs) in EVENT_TRACES {
        let c = ex.join(format!("{stem}.rce"));
        let t = trace(&c.to_string_lossy(), &std::fs::read_to_string(&c).unwrap(), Some(evs)).unwrap();
        out.push((format!("{stem}.events.json"), t + "\n"));
    }
    (out, skipped)
}

#[test]
fn fixtures_are_the_reference_traces() {
    let dir = spec().join("traces");
    let (all, skipped) = corpus();
    assert!(skipped.is_empty(), "every top-level example is admitted standalone: {skipped:?}");
    assert_eq!(all.len(), 58, "56 top-level examples plus two event-list traces");
    let bless = std::env::var_os("RCE_BLESS_TRACES").is_some();
    for (n, t) in &all {
        let p = dir.join(n);
        if bless {
            std::fs::write(&p, t).unwrap();
        }
        let have = std::fs::read_to_string(&p).unwrap_or_else(|_| panic!("missing fixture {n}; bless with RCE_BLESS_TRACES=1"));
        assert_eq!(&have, t, "{n} is stale; bless with RCE_BLESS_TRACES=1");
    }
    let n_json = std::fs::read_dir(&dir).unwrap().filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "json")).count();
    assert_eq!(n_json, all.len(), "no stray fixtures");
}

/// Canonical JSON as the engine hashes it (6.1): sorted keys, no spaces, a
/// float in Rust's `{:?}` form (6.3 N3: `10.0`, `1e16`, `5e-324`). The
/// fixture text writes floats as serde_json does (`1e+16`), so each float
/// is re-rendered from its value.
fn cj(v: &serde_json::Value) -> String {
    use serde_json::Value as J;
    match v {
        J::Number(n) if n.is_f64() => format!("{:?}", n.as_f64().unwrap()),
        J::Array(xs) => format!("[{}]", xs.iter().map(cj).collect::<Vec<_>>().join(",")),
        J::Object(m) => {
            let mut ks: Vec<_> = m.iter().collect();
            ks.sort_by(|a, b| a.0.cmp(b.0));
            format!("{{{}}}", ks.iter().map(|(k, x)| format!("{}:{}", J::String((*k).clone()), cj(x))).collect::<Vec<_>>().join(","))
        }
        other => other.to_string(),
    }
}

/// 6.3 N3: the float form of canonical JSON, pinned against the toolchain.
#[test]
fn canonical_json_float_form() {
    for (x, want) in [(10.0, "10.0"), (0.1, "0.1"), (1e-4, "0.0001"), (1e-5, "1e-5"), (1e16, "1e16"),
        (9007199254740992.0, "9007199254740992.0"), (5e-324, "5e-324"), (f64::MAX, "1.7976931348623157e308"), (-0.0, "-0.0")] {
        assert_eq!(format!("{x:?}"), want);
    }
}

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

#[test]
fn every_step_is_on_the_hash_chain() {
    let (all, _) = corpus();
    let mut steps = 0;
    for (n, t) in &all {
        let v: serde_json::Value = serde_json::from_str(t).unwrap();
        assert_eq!(v["format"], "rce-trace/1");
        for tr in v["traces"].as_array().unwrap() {
            let mut prev = v["genesis"].as_str().unwrap().to_string();
            for (k, s) in tr["steps"].as_array().unwrap().iter().enumerate() {
                assert_eq!(s["seq"], k, "{n}");
                assert_eq!(s["prev_hash"], prev.as_str(), "{n} seq {k}");
                let body = cj(&serde_json::json!({"event": s["event"], "prev_hash": prev, "seq": k, "state_hash": s["state_hash"]}));
                assert_eq!(s["hash"], sha(&body).as_str(), "{n} seq {k}");
                prev = s["hash"].as_str().unwrap().to_string();
                steps += 1;
            }
        }
    }
    assert!(steps >= 20, "{steps}");
}

#[test]
fn differential_runner_reports_match_and_first_divergence() {
    if Command::new("python3").arg("--version").output().is_err() {
        return;
    }
    let script = spec().join("traces/rce-trace-diff.py");
    let (all, _) = corpus();
    let d = std::env::temp_dir().join(format!("rce-trace-diff-{}", std::process::id()));
    let (a, b) = (d.join("ref"), d.join("other"));
    std::fs::create_dir_all(&a).unwrap();
    std::fs::create_dir_all(&b).unwrap();
    for (n, t) in &all {
        std::fs::write(a.join(n), t).unwrap();
        // the other projector may order keys and space differently
        let v: serde_json::Value = serde_json::from_str(t).unwrap();
        std::fs::write(b.join(n), serde_json::to_string_pretty(&v).unwrap()).unwrap();
    }
    let o = Command::new("python3").arg(&script).arg(&a).arg(&b).output().unwrap();
    let s = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{s}");
    assert!(s.contains("58/58 exact match"), "{s}");
    // diverge at seq 2 of the counter event list: a wrong state hash
    let n = "01_counter.events.json";
    let mut v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(a.join(n)).unwrap()).unwrap();
    v["traces"][0]["steps"][2]["state_hash"] = serde_json::json!("00");
    std::fs::write(b.join(n), v.to_string()).unwrap();
    let o = Command::new("python3").arg(&script).arg(&a).arg(&b).output().unwrap();
    let s = String::from_utf8_lossy(&o.stdout);
    assert_eq!(o.status.code(), Some(1), "{s}");
    assert!(s.contains("01_counter.events.json: DIVERGES: trace 0 (events) seq 2: state_hash"), "{s}");
    assert!(s.contains("57/58 exact match"), "{s}");
    // a projector that stops early
    v["traces"][0]["steps"].as_array_mut().unwrap().truncate(1);
    std::fs::write(b.join(n), v.to_string()).unwrap();
    let o = Command::new("python3").arg(&script).arg(a.join(n)).arg(b.join(n)).output().unwrap();
    assert!(String::from_utf8_lossy(&o.stdout).contains("seq 1: step missing"), "{}", String::from_utf8_lossy(&o.stdout));
}
