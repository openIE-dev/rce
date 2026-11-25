//! The hosted competence catalog service — the online target (rung 8–9).
//!
//! Serves the seeded stdlib basis as intent-keyed competence over HTTP:
//!   GET  /health           liveness
//!   GET  /meter            the ecology meter (|H_corpus|, reuse, cost/act)
//!   GET  /basis            basis size + sample intents
//!   POST /lookup           {operation, shape} -> G0 hit (owned handler) or miss
//!   POST /execute          {operation, shape, args} -> run owned competence at G0
//!   POST /admit            {handler_kind, input_key, ...} -> admit crystallized competence
//!
//! The Catalog is not Send (rcevc-core's Repo holds a non-Send Proposer), so a
//! dedicated worker thread owns it and handlers communicate via channels. A G0
//! hit executes the real stdlib operation through the RCE runtime at the
//! interpreter floor — zero model calls. The Law of Coding as a service.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::{extract::State, routing::{get, post}, Json, Router};
use competence_catalog::energy::EnergyMeter;
use competence_catalog::runtime::{GeneralReplayChecker, RuntimeBridge};
use competence_catalog::build::BuildCache;
use competence_catalog::canonical::{canonicalize, canonical_key};
use competence_catalog::form::FormMiner;
use competence_catalog::lut_emit::{builtin_luts, emit_expr};
use competence_catalog::semantic::{semantic_key, Probe, SemanticResolver};
use competence_catalog::symbolic::symbolically_equivalent;
use competence_catalog::threshold::{self, Measurements};
use competence_catalog::{bench, conform, fabric, fleet};
use competence_catalog::{basis_gate, intent_key, Catalog, CompetenceRecord, Provenance};
use rcevc_core::{Checker, State as RState, Unit};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::{mpsc, oneshot};

enum Req {
    Meter(oneshot::Sender<Value>),
    Lookup { op: String, shape: String, reply: oneshot::Sender<Value> },
    Execute { op: String, shape: String, args: Value, reply: oneshot::Sender<Value> },
    Admit { rec: CompetenceRecord, reply: oneshot::Sender<Value> },
    Measure { op: String, shape: String, args: Value, n: usize, reply: oneshot::Sender<Value> },
    Export(oneshot::Sender<Value>),
    Import { feed: Vec<RecordWire>, reply: oneshot::Sender<Value> },
    Sync { peers: Vec<String>, reply: oneshot::Sender<Value> },
    Resolve { shape: String, probes: Vec<ProbeWire>, reply: oneshot::Sender<Value> },
    Build { src: String, reply: oneshot::Sender<Value> },
    AdmitProgram { src: String, probes: Vec<ProgramProbeWire>, reply: oneshot::Sender<Value> },
    RunProgram { key: String, seed: HashMap<String, i128>, reply: oneshot::Sender<Value> },
}

/// A whole-program behavioral probe: a seeded environment and the expected return.
#[derive(Deserialize)]
struct ProgramProbeWire {
    seed: HashMap<String, i128>,
    expected: i128,
}

#[derive(Deserialize)]
struct AdmitProgramReq {
    src: String,
    probes: Vec<ProgramProbeWire>,
}

#[derive(Deserialize)]
struct RunProgramReq {
    key: String,
    #[serde(default)]
    seed: HashMap<String, i128>,
}

#[derive(Deserialize)]
struct BuildReq {
    src: String,
}

#[derive(Deserialize)]
struct SyncReq {
    peers: Vec<String>,
}

/// A behavioral probe on the wire: input arguments and the expected output.
#[derive(Deserialize)]
struct ProbeWire {
    args: Value,
    expected: String,
}

/// A semantic intent spec: a type signature plus behavioral examples.
#[derive(Deserialize)]
struct ResolveReq {
    shape: String,
    probes: Vec<ProbeWire>,
}

/// A competence record on the wire (circulation feed). Provenance is re-stamped
/// `Reported` on import; the receiving engine re-verifies by replay regardless.
#[derive(Deserialize)]
struct RecordWire {
    input_key: String,
    handler_kind: String,
    #[serde(default)]
    probe_input: String,
    #[serde(default)]
    probe_output: String,
    #[serde(default)]
    result: String,
    #[serde(default)]
    cost_uj: u64,
}

struct AppState {
    tx: mpsc::Sender<Req>,
    basis_count: usize,
    basis_sample: Vec<(String, String)>,
    entries: Arc<HashMap<String, (String, String, String)>>,
}

#[derive(Deserialize)]
struct IntentReq {
    operation: String,
    shape: String,
}

#[derive(Deserialize)]
struct ExecuteReq {
    operation: String,
    shape: String,
    #[serde(default)]
    args: Value,
}

#[derive(Deserialize)]
struct MeasureReq {
    operation: String,
    shape: String,
    #[serde(default)]
    args: Value,
    #[serde(default = "default_n")]
    n: usize,
}

fn default_n() -> usize {
    50
}

#[derive(Deserialize)]
struct AdmitReq {
    handler_kind: String,
    input_key: String,
    #[serde(default)]
    probe_input: String,
    #[serde(default)]
    probe_output: String,
    #[serde(default)]
    result: String,
    #[serde(default)]
    cost_uj: u64,
}

async fn health() -> &'static str {
    "ok"
}

async fn meter(State(st): State<Arc<AppState>>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Meter(tx)).await.unwrap();
    Json(rx.await.unwrap())
}

async fn basis_info(State(st): State<Arc<AppState>>) -> Json<Value> {
    let sample: Vec<Value> = st.basis_sample.iter()
        .map(|(op, sh)| json!({"operation": op, "shape": sh})).collect();
    Json(json!({"count": st.basis_count, "sample": sample}))
}

async fn lookup(State(st): State<Arc<AppState>>, Json(req): Json<IntentReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Lookup { op: req.operation, shape: req.shape, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn execute(State(st): State<Arc<AppState>>, Json(req): Json<ExecuteReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Execute { op: req.operation, shape: req.shape, args: req.args, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn admit(State(st): State<Arc<AppState>>, Json(req): Json<AdmitReq>) -> Json<Value> {
    let rec = CompetenceRecord {
        input_key: req.input_key,
        handler_kind: req.handler_kind,
        probe_input: req.probe_input,
        probe_output: req.probe_output,
        result: req.result,
        cost_uj: req.cost_uj,
        provenance: Provenance::Reported,
    };
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Admit { rec, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn admit_program(State(st): State<Arc<AppState>>, Json(req): Json<AdmitProgramReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::AdmitProgram { src: req.src, probes: req.probes, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn run_program(State(st): State<Arc<AppState>>, Json(req): Json<RunProgramReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::RunProgram { key: req.key, seed: req.seed, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn measure(State(st): State<Arc<AppState>>, Json(req): Json<MeasureReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Measure { op: req.operation, shape: req.shape, args: req.args, n: req.n, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn export(State(st): State<Arc<AppState>>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Export(tx)).await.unwrap();
    Json(rx.await.unwrap())
}

async fn import(State(st): State<Arc<AppState>>, Json(feed): Json<Vec<RecordWire>>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Import { feed, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn sync(State(st): State<Arc<AppState>>, Json(req): Json<SyncReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Sync { peers: req.peers, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn resolve(State(st): State<Arc<AppState>>, Json(req): Json<ResolveReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Resolve { shape: req.shape, probes: req.probes, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

async fn build_ep(State(st): State<Arc<AppState>>, Json(req): Json<BuildReq>) -> Json<Value> {
    let (tx, rx) = oneshot::channel();
    st.tx.send(Req::Build { src: req.src, reply: tx }).await.unwrap();
    Json(rx.await.unwrap())
}

#[derive(Deserialize)]
struct EquivalentReq {
    a: String,
    b: String,
}

#[derive(Deserialize)]
struct EmitReq {
    src: String,
    lang: String,
}

/// LUT-driven deterministic emission: the target language is a data table, not
/// hand-written emitter code. Same IR + same LUT → byte-identical output.
async fn emit_ep(Json(req): Json<EmitReq>) -> Json<Value> {
    let luts = builtin_luts();
    match luts.iter().find(|l| l.name == req.lang) {
        Some(lut) => match emit_expr(&req.src, lut) {
            Ok(out) => Json(json!({"lang": req.lang, "output": out, "method": "lut-deterministic"})),
            Err(e) => Json(json!({"error": e})),
        },
        None => Json(json!({"error": format!("unknown language {}", req.lang),
            "available": luts.iter().map(|l| l.name.clone()).collect::<Vec<_>>()})),
    }
}

async fn languages_ep() -> Json<Value> {
    let luts = builtin_luts();
    Json(json!({"count": luts.len(), "languages": luts.iter().map(|l| l.name.clone()).collect::<Vec<_>>(),
        "method": "each language is a data LUT; emission is deterministic lookup"}))
}

/// Unbounded equivalence on the polynomial fragment (symbolic canonical form);
/// outside the fragment, declines (fall back to bounded exhaustive testing).
async fn equivalent_ep(Json(req): Json<EquivalentReq>) -> Json<Value> {
    match symbolically_equivalent(&req.a, &req.b) {
        Some(eq) => Json(json!({"equivalent": eq, "method": "symbolic",
            "scope": "all inputs (polynomial fragment)"})),
        None => Json(json!({"equivalent": Value::Null, "method": "outside-fragment",
            "note": "non-polynomial; fall back to bounded exhaustive equivalence"})),
    }
}

#[derive(Deserialize)]
struct OrderReq {
    /// Live measurements: metric name → value (e.g. executor_coverage, gate_tests).
    measurements: HashMap<String, f64>,
}

/// The gated dependency order, resolved from live measurements: which stages are
/// complete to threshold, which are unlocked (the parallel frontier), and why the
/// rest are blocked. A dependent is workable only after its prerequisites are
/// fully completed — the cart-before-horse guard, as a service.
async fn order_ep(Json(req): Json<OrderReq>) -> Json<Value> {
    let dag = threshold::rce_dag();
    let mut m = Measurements::new();
    for (k, v) in &req.measurements {
        m.set(k, *v);
    }
    let done = threshold::complete(&dag, &m);
    let report: Vec<Value> = threshold::order_report(&dag, &m).into_iter().map(|(id, complete, unlocked, blockers)| {
        json!({"stage": id, "complete": complete, "unlocked": unlocked, "blocked_by": blockers})
    }).collect();
    Json(json!({
        "complete": done.iter().collect::<Vec<_>>(),
        "frontier": threshold::frontier(&dag, &m),
        "acyclic": threshold::acyclic(&dag),
        "report": report,
        "rule": "a dependent is workable only after every prerequisite is complete to threshold",
    }))
}

/// Live executor-coverage sweep over the whole basis (the ≥90% bar).
/// Runs on a dedicated large-stack thread: the reference interpreter recurses
/// deeply for some ops, which would overflow a runtime worker's small stack.
async fn coverage_ep(State(st): State<Arc<AppState>>) -> Json<Value> {
    let entries = st.entries.clone();
    let (covered, value, trap) = tokio::task::spawn_blocking(move || {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let bridge = RuntimeBridge::new_inprocess();
                bridge.sweep_coverage(&entries)
            })
            .expect("coverage thread")
            .join()
            .expect("coverage sweep")
    })
    .await
    .expect("coverage task");
    let total = st.entries.len();
    Json(json!({"total": total, "covered": covered, "returned_value": value, "trapped": trap,
        "coverage_frac": covered as f64 / total as f64}))
}

/// RCE-BENCH: the fixed suite through the gate, with joule receipts (no goalpost).
async fn bench_ep() -> Json<Value> {
    let corpus = std::collections::HashSet::new();
    let rs = bench::run_suite(|t| t.intent.to_string(), &corpus);
    let (pass, joules) = bench::summary(&rs);
    let receipts: Vec<Value> = rs.iter().map(|r| json!({
        "id": r.id, "correct": r.correct, "key": r.key, "est_uj": r.est_uj, "reused": r.reused
    })).collect();
    Json(json!({"receipts": receipts, "pass_rate": pass, "total_uj": joules}))
}

#[derive(Deserialize)]
struct ConformReq {
    source: String,
    lang: String,
    #[serde(default)]
    accept: Option<i128>,
}

/// RCE-Conform: verified legacy→modern conversion.
async fn conform_ep(Json(req): Json<ConformReq>) -> Json<Value> {
    match conform::convert(&req.source, &req.lang, req.accept) {
        Ok(c) => Json(json!({"original": c.original, "form": c.form, "key": c.key,
            "lang": c.target_lang, "modern": c.modern, "repaired": c.repaired, "verified": c.verified})),
        Err(e) => Json(json!({"error": e})),
    }
}

/// Heterogeneous compute: the global fabric table + lowest-joule placement.
async fn fabrics_ep() -> Json<Value> {
    let fs = fabric::builtin_fabrics();
    let best = fabric::place(&fs).map(|f| f.name).unwrap_or("");
    let rows: Vec<Value> = fs.iter().map(|f| json!({
        "name": f.name, "kind": format!("{:?}", f.kind), "has_fma": f.has_fma,
        "lower": fabric::lower_muladd(f)
    })).collect();
    Json(json!({"count": fs.len(), "place": best, "fabrics": rows}))
}

fn default_acts() -> usize { 200_000 }

#[derive(Deserialize)]
struct EnergyQuery { #[serde(default = "default_acts")] acts: usize }

/// This node's wall-metered energy for a batch of acts (bare-metal, `Measured`).
async fn energy_ep(axum::extract::Query(q): axum::extract::Query<EnergyQuery>) -> Json<Value> {
    let node = std::env::var("RCE_NODE").unwrap_or_else(|_| "local".into());
    let r = tokio::task::spawn_blocking(move || fleet::measure_local(&node, q.acts)).await.unwrap();
    match r {
        Ok(rd) => Json(json!(rd)),
        Err(e) => Json(json!({"error": e})),
    }
}

#[derive(Deserialize)]
struct FleetQuery { nodes: String, #[serde(default = "default_acts")] acts: usize }

/// Fleet-wide aggregation: collect wall-metered readings from the given nodes and total them.
async fn fleet_ep(axum::extract::Query(q): axum::extract::Query<FleetQuery>) -> Json<Value> {
    let nodes: Vec<String> = q.nodes.split(',').map(|s| s.trim().to_string()).collect();
    let acts = q.acts;
    let readings = tokio::task::spawn_blocking(move || fleet::collect(&nodes, acts)).await.unwrap();
    let (tot, acts_total, mean) = fleet::aggregate(&readings);
    Json(json!({"nodes": readings, "total_uj": tot, "total_acts": acts_total, "mean_uj_per_act": mean,
        "mean_us_per_act": fleet::aggregate_latency(&readings)}))
}


#[derive(Deserialize)]
struct CanonReq {
    src: String,
}

/// Canonicalization-on-save: reshape an expression to canonical form, killing the
/// accidental-representation tax. Pure (no worker round-trip).
async fn canonicalize_ep(Json(req): Json<CanonReq>) -> Json<Value> {
    match canonicalize(&req.src) {
        Ok(c) => Json(json!({"canonical": c, "key": canonical_key(&req.src).unwrap_or_default()})),
        Err(e) => Json(json!({"error": e})),
    }
}

#[derive(Deserialize)]
struct MineReq {
    shape: String,
    probes: Vec<ProbeWire>,
}

/// Mine the categorical form + filling that covers an intent spec (Conal/DéjàVu).
async fn mine_ep(Json(req): Json<MineReq>) -> Json<Value> {
    let miner = FormMiner::new(RuntimeBridge::new_inprocess());
    let probes: Vec<Probe> = req.probes.iter()
        .map(|p| Probe::new(p.args.clone(), p.expected.clone())).collect();
    match miner.mine(&req.shape, &probes) {
        Some(m) => Json(json!({"form": m.form, "op": m.op, "workflow": m.workflow.name})),
        None => Json(json!({"error": "no categorical form covers the intent"})),
    }
}

/// The general G0 dispatch: look up the operation's signature and execute it
/// through the shape-driven runtime bridge. `args` is a positional JSON array
/// matching the signature's arity. Any basis operation whose shape is
/// marshalable executes; the rest report why honestly (no hardcoded handlers).
fn dispatch(
    bridge: &RuntimeBridge,
    entries: &HashMap<String, (String, String, String)>,
    handler: &str,
    args: &Value,
) -> Result<String, String> {
    let (module, name, shape) = entries.get(handler)
        .ok_or_else(|| format!("no signature indexed for {handler}"))?;
    bridge.call_general(module, name, shape, args)
}

/// CALM monotone import: skip owned intents (idempotent), admit each novel record
/// only by replay at the border. Returns (accepted, refused, skipped). Shared by
/// `/import` and `/sync` so federation preserves the admission law.
fn do_import(cat: &mut Catalog, replay: &GeneralReplayChecker, feed: &[RecordWire]) -> (usize, usize, usize) {
    let owned: HashSet<String> = cat.records().iter().map(|r| r.input_key.clone()).collect();
    let (mut accepted, mut refused, mut skipped) = (0, 0, 0);
    for w in feed {
        if owned.contains(&w.input_key) {
            skipped += 1;
            continue;
        }
        let rec = CompetenceRecord {
            input_key: w.input_key.clone(),
            handler_kind: w.handler_kind.clone(),
            probe_input: w.probe_input.clone(),
            probe_output: w.probe_output.clone(),
            result: w.result.clone(),
            cost_uj: w.cost_uj,
            provenance: Provenance::Reported,
        };
        let unit = Unit::new(rec.encode(), &[]);
        if replay.check_unit(&RState::empty(), &rec.input_key, &unit).is_ok()
            && cat.admit(rec).is_accepted()
        {
            accepted += 1;
        } else {
            refused += 1;
        }
    }
    (accepted, refused, skipped)
}

#[tokio::main]
async fn main() {
    let index = std::fs::read_to_string("../stdlib-basis-index.json").expect("basis index");
    let entries: Vec<Value> = serde_json::from_str(&index).expect("parse index");
    let basis: Vec<(String, String)> = entries.iter()
        .map(|e| (e["operation"].as_str().unwrap().to_string(), e["shape"].as_str().unwrap().to_string()))
        .collect();
    let basis_count = basis.len();
    let basis_sample: Vec<(String, String)> = basis.iter().take(8).cloned().collect();

    // operation -> (module, name, shape): the signature table the general
    // executor marshals against. Every indexed basis operation is executable
    // up to the shapes the pure runtime bridge can marshal.
    let entries: HashMap<String, (String, String, String)> = entries.iter()
        .map(|e| (
            e["operation"].as_str().unwrap().to_string(),
            (
                e["module"].as_str().unwrap().to_string(),
                e["name"].as_str().unwrap().to_string(),
                e["shape"].as_str().unwrap().to_string(),
            ),
        ))
        .collect();

    let entries_arc = Arc::new(entries.clone());
    let (tx, mut rx) = mpsc::channel::<Req>(256);

    // The catalog worker: owns the non-Send Catalog + bridge in one thread.
    std::thread::spawn(move || {
        let mut cat = Catalog::with_gate(basis_gate()).expect("catalog");
        let n = cat.seed_basis_index(&basis);
        // In-process execution through the linked reference interpreter: no
        // process fork per act (≈27× lower measured energy), self-contained.
        let bridge = RuntimeBridge::new_inprocess();
        // Crystallized competence (/admit) is admitted only by per-record replay:
        // the runtime re-executes the declared spectest and must reproduce it.
        let sig_entries: Vec<(String, String, String, String)> = entries.iter()
            .map(|(op, (m, nm, sh))| (op.clone(), m.clone(), nm.clone(), sh.clone()))
            .collect();
        let replay = GeneralReplayChecker::from_index(bridge.clone(), &sig_entries);
        // The semantic resolver: the basis indexed by shape, so an intent spec
        // (shape + behavioral probes) resolves to owned competence by replay.
        let mut semantic = SemanticResolver::new(bridge.clone(), &sig_entries);
        // Whole-program competence admitted by replay, served at G0 by content key.
        let mut programs: HashMap<String, competence_catalog::lut_emit::Stmt> = HashMap::new();
        for (op, _m, _n, sh) in &sig_entries {
            semantic.index(&CompetenceRecord {
                input_key: intent_key(op, sh),
                handler_kind: op.clone(),
                probe_input: String::new(),
                probe_output: String::new(),
                result: String::new(),
                cost_uj: 1,
                provenance: Provenance::Reported,
            });
        }
        // The hardware meter for measured_j (None if no source can measure here).
        let mut energy = EnergyMeter::open().ok();
        // The content-addressed incremental build layer (Blaze ingested).
        let mut build_cache = BuildCache::new();
        eprintln!("catalog worker ready (|H_corpus|={n}, meter={})",
            energy.as_ref().map(|m| m.source()).unwrap_or("none"));
        while let Some(req) = rx.blocking_recv() {
            match req {
                Req::Meter(reply) => {
                    let m = cat.meter();
                    let _ = reply.send(json!({
                        "catalog_size": m.catalog_size, "reuse_len": m.reuse_len,
                        "proposer_calls": m.proposer_calls, "hits": m.hits, "misses": m.misses,
                        "acts": m.acts, "reuse_rate": m.reuse_rate,
                        "workload_est_uj": m.workload_est_uj, "cost_per_act_uj": m.cost_per_act_uj,
                    }));
                }
                Req::Lookup { op, shape, reply } => {
                    let r = match cat.hit_intent(&op, &shape) {
                        Some(h) => json!({"hit": true, "handler": h, "cost_uj": cat.costs().g0}),
                        None => json!({"hit": false, "handler": null, "cost_uj": cat.costs().g4}),
                    };
                    let _ = reply.send(r);
                }
                Req::Execute { op, shape, args, reply } => {
                    let handler = cat.hit_intent(&op, &shape);
                    let r = match handler {
                        None => json!({"hit": false, "error": "no owned competence; escalate to proposer"}),
                        Some(h) => match dispatch(&bridge, &entries, &h, &args) {
                            Ok(v) => json!({"hit": true, "handler": h, "result": v, "cost_uj": 1}),
                            Err(e) => json!({"hit": true, "handler": h, "error": e}),
                        },
                    };
                    let _ = reply.send(r);
                }
                Req::Admit { rec, reply } => {
                    let unit = Unit::new(rec.encode(), &[]);
                    if replay.check_unit(&RState::empty(), &rec.input_key, &unit).is_ok() {
                        let out = cat.admit(rec);
                        let _ = reply.send(json!({"accepted": out.is_accepted(), "gate": "replay"}));
                    } else {
                        let _ = reply.send(json!({"accepted": false, "gate": "replay",
                            "error": "spectest did not reproduce; refused"}));
                    }
                }
                Req::AdmitProgram { src, probes, reply } => {
                    let r = match competence_catalog::program::lift_program(&src) {
                        Err(e) => json!({"accepted": false, "gate": "replay", "error": e}),
                        Ok(stmt) => {
                            let ps: Vec<(competence_catalog::program::Env, i128)> = probes.iter()
                                .map(|p| (p.seed.iter().map(|(k, v)| (k.clone(), *v)).collect(), p.expected))
                                .collect();
                            if competence_catalog::program::replay_admit(&stmt, &ps) {
                                match competence_catalog::program::program_key(&stmt) {
                                    Ok(key) => {
                                        programs.insert(key.clone(), stmt);
                                        json!({"accepted": true, "gate": "replay", "key": key, "cost_uj": 1})
                                    }
                                    Err(e) => json!({"accepted": false, "gate": "replay", "error": e}),
                                }
                            } else {
                                json!({"accepted": false, "gate": "replay",
                                    "error": "program did not reproduce every probe; refused"})
                            }
                        }
                    };
                    let _ = reply.send(r);
                }
                Req::RunProgram { key, seed, reply } => {
                    let r = match programs.get(&key) {
                        None => json!({"hit": false, "error": "no owned whole-program competence; escalate"}),
                        Some(stmt) => {
                            let mut env: competence_catalog::program::Env =
                                seed.iter().map(|(k, v)| (k.clone(), *v)).collect();
                            match competence_catalog::program::eval_program(stmt, &mut env, &mut competence_catalog::program::FnTable::new()) {
                                Ok(v) => json!({"hit": true, "key": key, "result": v, "cost_uj": 1, "model_calls": 0}),
                                Err(e) => json!({"hit": true, "key": key, "error": e}),
                            }
                        }
                    };
                    let _ = reply.send(r);
                }
                Req::Measure { op, shape, args, n, reply } => {
                    // Pure lookup (no ecology metering); measure a batch of G0
                    // executions under the hardware meter.
                    let handler = cat.lookup_intent(&op, &shape);
                    let r = match handler {
                        None => json!({"error": "no owned competence for this intent"}),
                        Some(h) => match &mut energy {
                            None => json!({"provenance": "Modelled", "modelled_floor_uj": 1,
                                "error": "no hardware meter can measure here; no Measured figure claimed"}),
                            Some(meter) => {
                                let (b, e, hh, aa) = (&bridge, &entries, &h, &args);
                                match meter.measure_mean(n, || { let _ = dispatch(b, e, hh, aa); }) {
                                    Ok(uj) => json!({"provenance": "Measured", "source": meter.source(),
                                        "measured_uj_per_act": uj, "n": n, "modelled_floor_uj": 1,
                                        "note": "in-process execution; whole-package window, concurrent work included"}),
                                    Err(e) => json!({"provenance": "Modelled", "error": format!("measurement failed: {e}")}),
                                }
                            }
                        },
                    };
                    let _ = reply.send(r);
                }
                Req::Export(reply) => {
                    // Only crystallized competence circulates: the conformance-
                    // seeded basis has no spectest (it is shared via the stdlib),
                    // so the feed carries records that replay can re-verify.
                    let recs: Vec<CompetenceRecord> = cat.records().into_iter()
                        .filter(|r| !r.probe_input.is_empty()).collect();
                    let feed: Vec<Value> = recs.iter().map(|r| json!({
                        "input_key": r.input_key, "handler_kind": r.handler_kind,
                        "probe_input": r.probe_input, "probe_output": r.probe_output,
                        "result": r.result, "cost_uj": r.cost_uj,
                    })).collect();
                    let _ = reply.send(json!({"count": feed.len(), "feed": feed}));
                }
                Req::Import { feed, reply } => {
                    let (accepted, refused, skipped) = do_import(&mut cat, &replay, &feed);
                    let _ = reply.send(json!({"total": feed.len(), "accepted": accepted,
                        "refused": refused, "skipped": skipped, "gate": "replay"}));
                }
                Req::Sync { peers, reply } => {
                    // Federation: pull each peer's /export feed over HTTP and
                    // merge it (replay-gated). CALM monotone — nodes converge.
                    let (mut total, mut accepted, mut refused, mut skipped) = (0, 0, 0, 0);
                    let mut peer_reports = Vec::new();
                    for peer in &peers {
                        let url = format!("{}/export", peer.trim_end_matches('/'));
                        let fetched = ureq::get(&url).call().map_err(|e| e.to_string())
                            .and_then(|r| r.into_json::<Value>().map_err(|e| e.to_string()));
                        match fetched {
                            Ok(v) => {
                                let feed: Vec<RecordWire> = serde_json::from_value(v["feed"].clone()).unwrap_or_default();
                                let (a, r, s) = do_import(&mut cat, &replay, &feed);
                                total += feed.len(); accepted += a; refused += r; skipped += s;
                                peer_reports.push(json!({"peer": peer, "fetched": feed.len(),
                                    "accepted": a, "refused": r, "skipped": s}));
                            }
                            Err(e) => peer_reports.push(json!({"peer": peer, "error": e.to_string()})),
                        }
                    }
                    let _ = reply.send(json!({"peers": peer_reports, "total": total,
                        "accepted": accepted, "refused": refused, "skipped": skipped,
                        "corpus_size": cat.meter().catalog_size}));
                }
                Req::Resolve { shape, probes, reply } => {
                    // Semantic intent resolution: behavior, not name. I → 0 on a hit.
                    let ps: Vec<Probe> = probes.iter()
                        .map(|p| Probe::new(p.args.clone(), p.expected.clone())).collect();
                    let r = match semantic.resolve(&shape, &ps) {
                        Some(hit) => json!({"hit": true, "handler": hit.handler,
                            "key": hit.key, "cost_uj": 1}),
                        None => json!({"hit": false, "key": semantic_key(&shape, &ps),
                            "error": "novel behavior relative to the corpus; escalate"}),
                    };
                    let _ = reply.send(r);
                }
                Req::Build { src, reply } => {
                    let rec = build_cache.build(&src);
                    let _ = reply.send(json!({"key": rec.key, "ok": rec.ok,
                        "evidence": rec.evidence.word(), "diagnostic": rec.diagnostic,
                        "hits": build_cache.hits, "misses": build_cache.misses,
                        "hit_rate": build_cache.hit_rate()}));
                }
            }
        }
    });

    let state = Arc::new(AppState { tx, basis_count, basis_sample, entries: entries_arc });
    let app = Router::new()
        .route("/health", get(health))
        .route("/meter", get(meter))
        .route("/basis", get(basis_info))
        .route("/lookup", post(lookup))
        .route("/execute", post(execute))
        .route("/admit", post(admit))
        .route("/admit_program", post(admit_program))
        .route("/run_program", post(run_program))
        .route("/measure", post(measure))
        .route("/export", get(export))
        .route("/import", post(import))
        .route("/sync", post(sync))
        .route("/resolve", post(resolve))
        .route("/canonicalize", post(canonicalize_ep))
        .route("/mine", post(mine_ep))
        .route("/build", post(build_ep))
        .route("/equivalent", post(equivalent_ep))
        .route("/order", post(order_ep))
        .route("/coverage", get(coverage_ep))
        .route("/bench", get(bench_ep))
        .route("/conform", post(conform_ep))
        .route("/fabrics", get(fabrics_ep))
        .route("/energy", get(energy_ep))
        .route("/fleet", get(fleet_ep))
        .route("/emit", post(emit_ep))
        .route("/languages", get(languages_ep))
        .with_state(state);

    let port = std::env::var("RCE_CATALOG_PORT").ok()
        .and_then(|p| p.parse::<u16>().ok()).unwrap_or(8765);
    let addr = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    println!("competence catalog serving on http://{addr} (|H_corpus|={basis_count})");
    axum::serve(listener, app).await.unwrap();
}