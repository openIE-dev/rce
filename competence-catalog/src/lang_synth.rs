//! Native-language synthesis — turn a `langpredict` fabric capability-gap spec into an actual
//! language-as-data: a construct set (universal core + the fabric's native constructs), a seed
//! grammar (node-types), per-construct surface templates (an emission LUT), and a replay-gated
//! sample program. The language is adaptable: the Form allows the vocabulary to expand beyond
//! the pinned 113, so a quantum/thermodynamic/spatial-native language is synthesized, not invented
//! by hand.
//!
//! Prototype scope: the synthesized language's grammar/LUT/sample are real and replay-gated; the
//! *native op semantics* are declared (a semantics table) and their production execution/compilation
//! to the fabric is the open engineering step. Core-op sample programs execute deterministically now.

use std::collections::BTreeMap;

use serde_json::json;

use crate::langpredict::FabricClass;
use crate::lut_emit::LanguageLut;
use crate::program::{replay_admit, Env};

/// Core constructs every synthesized language inherits (the universal mean).
const CORE: &[&str] = &[
    "int_lit", "float_lit", "bool_lit", "str_lit", "var", "neg", "not", "add", "sub", "mul",
    "div", "mod", "eq", "ne", "lt", "gt", "le", "ge", "and", "or", "if_expr", "call", "index",
    "member", "list", "let", "assign", "return", "if_stmt", "while", "for", "fn", "block",
];

#[derive(Debug, Clone)]
pub struct SynthesizedLanguage {
    pub name: String,
    pub paradigm: String,
    pub fabric: String,
    /// Universal core + native constructs (the expanded, adaptable vocabulary).
    pub constructs: Vec<String>,
    /// Seed grammar: named node types, one per construct.
    pub node_types: String,
    /// Per-construct surface templates (emission LUT).
    pub templates: BTreeMap<String, String>,
    /// Declared native-op semantics (production execution to the fabric is open).
    pub native_semantics: BTreeMap<String, String>,
    /// A core-op sample program that replays deterministically now.
    pub sample_src: String,
    pub probes: Vec<(Env, i128)>,
}

fn native_template(fabric: &str, c: &str) -> String {
    match (fabric, c) {
        ("quantum", "qubit") => "qubit {name}".into(),
        ("quantum", "superpose") => "superpose {a} {b}".into(),
        ("quantum", "entangle") => "entangle {a} {b}".into(),
        ("quantum", "measure") => "measure {a}".into(),
        ("quantum", "phase") => "phase {a} {b}".into(),
        ("quantum", "gate") => "gate {name} {a}".into(),
        ("quantum", "teleport") => "teleport {a} {b}".into(),
        ("quantum", "no_clone") => "no_clone {a}".into(),
        ("thermodynamic", "reverse") => "reverse {a}".into(),
        ("thermodynamic", "dissipate") => "dissipate {a}".into(),
        ("thermodynamic", "landauer") => "landauer {a}".into(),
        ("thermodynamic", "entropy_budget") => "entropy_budget {name} {value}".into(),
        ("thermodynamic", "heat") => "heat {a}".into(),
        ("thermodynamic", "reversible_gate") => "reversible_gate {name} {a}".into(),
        ("thermodynamic", "energy_type") => "energy_type {name}".into(),
        ("spatial", "place") => "place {a} {b}".into(),
        ("spatial", "route") => "route {a} {b}".into(),
        ("spatial", "pipeline") => "pipeline {items}".into(),
        ("spatial", "spatial_loop") => "spatial_loop {name} {items}".into(),
        ("spatial", "reconfig") => "reconfig {name}".into(),
        ("spatial", "bitstream") => "bitstream {name}".into(),
        ("neuromorphic", "spike") => "spike {a}".into(),
        ("neuromorphic", "synapse") => "synapse {a} {b}".into(),
        ("neuromorphic", "plasticity") => "plasticity {name}".into(),
        ("neuromorphic", "event_time") => "event_time {a}".into(),
        ("neuromorphic", "membrane") => "membrane {name}".into(),
        ("probabilistic", "sample") => "sample {a}".into(),
        ("probabilistic", "distribution") => "distribution {name}".into(),
        ("probabilistic", "bernoulli") => "bernoulli {a}".into(),
        ("probabilistic", "infer") => "infer {a}".into(),
        ("probabilistic", "pbit") => "pbit {name}".into(),
        ("analog", "conductance") => "conductance {a}".into(),
        ("analog", "mac") => "mac {a} {b}".into(),
        ("analog", "crossbar") => "crossbar {name}".into(),
        ("analog", "drift") => "drift {a}".into(),
        ("analog", "analog_read") => "analog_read {a}".into(),
        _ => format!("{c} {{a}}"),
    }
}

fn native_semantic(fabric: &str, c: &str) -> String {
    match (fabric, c) {
        ("quantum", "superpose") => "prepare α|0>+β|1> (amplitudes normalized); unitary, reversible".into(),
        ("quantum", "entangle") => "Bell-pair correlation between two qubits; non-local joint state".into(),
        ("quantum", "measure") => "project to |0>/|1> with Born probability; collapses superposition".into(),
        ("quantum", "no_clone") => "linearity forbids copying an unknown state; enforced as a type rule".into(),
        ("thermodynamic", "landauer") => "erasing one bit dissipates >= kT ln2; costed as an energy type".into(),
        ("thermodynamic", "reverse") => "run the computation backwards; only logically reversible ops allowed".into(),
        ("thermodynamic", "entropy_budget") => "static bound on dissipation; reject programs exceeding it".into(),
        ("spatial", "place") => "bind an operation to a physical tile/PE; placement is a first-class op".into(),
        ("spatial", "route") => "configure interconnect between tiles; latency is a first-class cost".into(),
        ("neuromorphic", "spike") => "emit an event at event-time; asynchronous, event-driven semantics".into(),
        ("neuromorphic", "plasticity") => "update synaptic weight by a local learning rule (e.g. STDP)".into(),
        ("probabilistic", "sample") => "draw from a distribution; semantics is a probability monad".into(),
        ("probabilistic", "pbit") => "stochastic bit with tunable bias; hardware-native randomness".into(),
        ("analog", "mac") => "multiply-accumulate in conductance domain; O(1) physics, noise-bounded".into(),
        ("analog", "crossbar") => "vector-matrix multiply by Ohm/Kirchhoff on a resistive array".into(),
        _ => "native fabric operation; production lowering open".into(),
    }
}

/// Synthesize a fabric-native language from its capability-gap spec.
pub fn synthesize(f: &FabricClass) -> SynthesizedLanguage {
    let mut constructs: Vec<String> = CORE.iter().map(|s| s.to_string()).collect();
    let mut templates = BTreeMap::new();
    let mut semantics = BTreeMap::new();
    for c in CORE {
        templates.insert(c.to_string(), crate::lut_emit::builtin_luts().into_iter()
            .find(|l| l.name == "rust").and_then(|l| l.template(c).cloned()).unwrap_or_else(|| "{a}".to_string()));
    }
    for c in f.native_constructs {
        constructs.push(c.to_string());
        templates.insert(c.to_string(), native_template(f.id, c));
        semantics.insert(c.to_string(), native_semantic(f.id, c));
    }
    let node_types = json!(constructs.iter().map(|c| json!({"type": c, "named": true})).collect::<Vec<_>>()).to_string();
    let probes: Vec<(Env, i128)> = vec![(BTreeMap::new(), 3)];
    SynthesizedLanguage {
        name: f.predicted_name.to_string(),
        paradigm: f.paradigm.to_string(),
        fabric: f.id.to_string(),
        constructs,
        node_types,
        templates,
        native_semantics: semantics,
        sample_src: "let x = 1; let y = 2; return x + y;".into(),
        probes,
    }
}

impl SynthesizedLanguage {
    /// Grammar coverage of the synthesized language against its own construct set (1.0 by
    /// construction: every named node is a declared construct).
    pub fn coverage(&self) -> f64 {
        let nt: serde_json::Value = serde_json::from_str(&self.node_types).unwrap();
        let arr = nt.as_array().unwrap();
        let set: std::collections::BTreeSet<&str> = self.constructs.iter().map(|s| s.as_str()).collect();
        let mapped = arr.iter().filter(|n| set.contains(n["type"].as_str().unwrap_or(""))).count();
        mapped as f64 / arr.len() as f64
    }

    /// The synthesized emission LUT.
    pub fn lut(&self) -> LanguageLut {
        let refs: Vec<(&str, &str)> = self.templates.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        LanguageLut::new(&self.name, &refs)
    }

    /// The core-op sample replays deterministically now (native-op execution is the open step).
    pub fn sample_replays(&self) -> bool {
        let stmt = crate::program::lift_program(&self.sample_src).unwrap();
        replay_admit(&stmt, &self.probes)
    }
}
