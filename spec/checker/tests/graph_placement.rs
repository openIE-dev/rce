//! HC-39: before the host runs a graph term it places it, between the CPU
//! reference and wasm under wasmi, by predicted joules (`est_j`) from a
//! cost table whose source D states; a fabric is a candidate for a term
//! only after a `host.Exact` probe on that term agrees with the reference
//! at every node's level; each decision is `host.Placed` in D, and replay
//! re-derives it and gives the same state hashes.
use rce_spec_check::host::{CostRow, CostTable, Entry, Host, replay_report};
use rce_spec_check::interp::{GRAPH_CLASS, est_j, place_graph};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-graph-placement-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn example(name: &str) -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../compute/examples").join(name)).unwrap()
}

/// A table under which wasm predicts fewer joules than the CPU reference.
fn wasm_cheaper() -> CostTable {
    CostTable {
        source: "declared for this test".into(),
        rows: vec![
            CostRow { fabric: "cpu".into(), overhead_j: 0.0, work_per_j: 1.0e9 },
            CostRow { fabric: "wasm".into(), overhead_j: 0.0, work_per_j: 1.0e10 },
        ],
    }
}

/// Run `inputs` one batch at a time under `table`: D's text, and the
/// state hash after each batch.
fn run(src: &str, dir: &Path, table: &CostTable, inputs: &[&str]) -> (String, Vec<String>) {
    let m = rce_spec_check::program(src).unwrap();
    let log = dir.join("d");
    let mut h = Host::open_file_with(&m, &log, vec![]).unwrap();
    h.graph_costs(table).unwrap();
    let mut hashes = vec![];
    for i in inputs {
        h.admit(src, i).unwrap();
        hashes.push(h.state_hash().to_string());
    }
    drop(h);
    (std::fs::read_to_string(&log).unwrap(), hashes)
}

fn payloads(d: &str, kind: &str) -> Vec<Value> {
    d.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter(|v| v["event"]["type"] == kind)
        .map(|v| v["event"]["payload"].clone())
        .collect()
}

const DENSE: [&str; 3] = ["Input(x: [1.0, 2.0, 3.0])", "Input(x: [-1.0, 0.0, 1.0])", "Input(x: [0.25, -0.5, 4.0])"];

#[test]
fn the_declared_table_is_logged_with_its_source_and_places_on_the_cpu_reference() {
    let src = example("dense.rce");
    let (d, _) = run(&src, &tmp("declared"), &CostTable::declared(), &DENSE);
    let cal = payloads(&d, "host.Calibrated");
    assert_eq!(cal.len(), 2);
    assert!(cal.iter().all(|c| c["class"] == GRAPH_CLASS && c["source"] == "declared" && c["unit"] == "J"));
    let placed = payloads(&d, "host.Placed");
    assert_eq!(placed.len(), 3, "one placement per run of the term");
    for p in &placed {
        assert_eq!(p["fabric"], "cpu");
        assert_eq!(p["est_j"].as_f64().unwrap(), est_j((0.0, 1.0e9), p["work"].as_f64().unwrap()));
    }
    assert!(payloads(&d, "host.Exact").is_empty(), "a fabric that predicts more joules is not probed");
}

#[test]
fn a_cheaper_fabric_is_probed_once_then_placed_and_the_state_hashes_are_the_reference_ones() {
    let src = example("dense.rce");
    let (dc, hc) = run(&src, &tmp("cpu"), &CostTable::declared(), &DENSE);
    let (dw, hw) = run(&src, &tmp("wasm"), &wasm_cheaper(), &DENSE);
    let ex = payloads(&dw, "host.Exact");
    assert_eq!(ex.len(), 1, "one probe per term");
    assert_eq!(ex[0]["probe"], ex[0]["reference"]);
    assert!(payloads(&dw, "host.Placed").iter().all(|p| p["fabric"] == "wasm"));
    // every node of dense.rce is exact, so the state is the same bits
    assert_eq!(hc, hw);
    let m = rce_spec_check::program(&src).unwrap();
    for (d, h) in [(&dc, &hc), (&dw, &hw)] {
        assert_eq!(&replay_report(&m, d).unwrap().state_hash, h.last().unwrap());
    }
}

#[test]
fn each_term_of_a_block_is_probed_once_then_placed_on_wasm_and_replays() {
    let src = example("block.rce");
    let ins = ["Tokens(ids: [0, 1, 2])", "Tokens(ids: [3, 0, 1])", "Tokens(ids: [5, 0, 1])"];
    let (dc, hc) = run(&src, &tmp("block-cpu"), &CostTable::declared(), &ins);
    let (dw, hw) = run(&src, &tmp("block-wasm"), &wasm_cheaper(), &ins);
    let ex = payloads(&dw, "host.Exact");
    // each term of the block is probed once, on its first operands
    let classes: std::collections::BTreeSet<_> = ex.iter().map(|e| e["class"].to_string()).collect();
    assert_eq!(classes.len(), ex.len());
    assert!(ex.len() > 1 && ex.iter().all(|e| e["probe"] == e["reference"]), "every node at its level");
    assert_eq!(hc, hw, "the block's nodes are exact on both fabrics");
    assert!(payloads(&dw, "host.Placed").iter().all(|p| p["fabric"] == "wasm"));
    let m = rce_spec_check::program(&src).unwrap();
    assert_eq!(replay_report(&m, &dw).unwrap().state_hash, *hw.last().unwrap());
    assert_eq!(replay_report(&m, &dc).unwrap().state_hash, *hc.last().unwrap());
    // the trap of the third input is the same on both fabrics
    assert_eq!(replay_report(&m, &dw).unwrap().kept_outcomes.len(), replay_report(&m, &dc).unwrap().kept_outcomes.len());
}

#[test]
fn replay_requires_change_when_a_placement_is_not_the_one_d_derives() {
    let src = example("dense.rce");
    let m = rce_spec_check::program(&src).unwrap();
    let (d, _) = run(&src, &tmp("forged"), &wasm_cheaper(), &DENSE[..1]);
    // rewrite the placement to the CPU reference and chain the entries again
    let mut prev = "0".repeat(64);
    let mut out = String::new();
    for l in d.lines() {
        let e = Entry::parse_line(l).unwrap();
        let ev = if e.kind() == "host.Placed" { e.event.replace("\"fabric\":\"wasm\"", "\"fabric\":\"cpu\"") } else { e.event.clone() };
        let n = Entry::new(&m, e.seq, ev, &prev, &e.state_hash);
        prev = n.hash.clone();
        out.push_str(&n.line());
        out.push('\n');
    }
    let err = replay_report(&m, &out).map(|r| r.state_hash).expect_err("the forged placement is found");
    assert!(err.contains("placement re-derives wasm"), "{err}");
}

#[test]
fn placement_takes_the_least_predicted_joules_among_probed_fabrics() {
    let mut costs = BTreeMap::new();
    costs.insert("cpu".to_string(), (0.0, 1.0e9));
    costs.insert("wasm".to_string(), (1.0e-6, 1.0e10));
    let mut probes = BTreeMap::new();
    // not probed: never a candidate
    assert_eq!(place_graph(&costs, &probes, "k", 1.0e6).0, "cpu");
    probes.insert(("wasm".to_string(), "k".to_string()), true);
    // 1e6 operations: cpu 1e-3 J, wasm 1.01e-4 J
    assert_eq!(place_graph(&costs, &probes, "k", 1.0e6), ("wasm".to_string(), 1.0e-6 + 1.0e6 / 1.0e10));
    // 100 operations: the overhead dominates
    assert_eq!(place_graph(&costs, &probes, "k", 100.0).0, "cpu");
    // a probe that disagreed, or a probe of another term, admits nothing
    probes.insert(("wasm".to_string(), "k".to_string()), false);
    probes.insert(("wasm".to_string(), "j".to_string()), true);
    assert_eq!(place_graph(&costs, &probes, "k", 1.0e6).0, "cpu");
    // a tie goes to the CPU reference
    costs.insert("wasm".to_string(), (0.0, 1.0e9));
    assert_eq!(place_graph(&costs, &probes, "j", 1.0e6).0, "cpu");
}

#[test]
fn a_cost_table_states_its_source() {
    assert!(CostTable::parse(r#"{"source":"measured","rows":[]}"#).is_err());
    assert!(CostTable::parse(r#"{"rows":[]}"#).is_err());
    assert!(CostTable::parse(r#"{"source":"declared","rows":[{"fabric":"gpu","overhead_j":0,"work_per_j":1}]}"#).is_err());
    assert!(CostTable::parse(r#"{"source":"declared","rows":[{"fabric":"cpu","overhead_j":0,"work_per_j":0}]}"#).is_err());
    let t = CostTable::parse(r#"{"source":"MacBook Pro M5 Max, macmon package counter","rows":[{"fabric":"cpu","overhead_j":0,"work_per_j":2e9}]}"#).unwrap();
    assert_eq!(t.rows, vec![CostRow { fabric: "cpu".into(), overhead_j: 0.0, work_per_j: 2.0e9 }]);
}
