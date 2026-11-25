//! The semantic intent key — the corpus-independent `I` measure of the Law of
//! Coding, made tractable and falsifiable.
//!
//! An intent is a *spec*: a type signature (`shape`) plus a behavioral spec
//! (input→output examples). The intent is **semantically covered** by the corpus
//! iff some owned competence reproduces the behavioral spec under replay —
//! regardless of its name. Coverage is behavior, not string match, so a novel
//! surface form of an already-owned behavior hits G0: `I = H_task − H_corpus → 0`.
//!
//! The semantic key is the name-free identity of an intent: the hash of the
//! normalized shape and the canonical behavioral spec. Equivalent specs collide;
//! novel behavior does not. Resolution reuses the admission law's replay as the
//! semantic-equivalence check — the gate and the resolver are the same mechanism.
//!
//! Honesty: behavioral equivalence is relative to the finite spec. A candidate
//! that agrees on the spec's probes is a match *for that spec*; more probes
//! tighten the equivalence. This is the deterministic notion of "same intent"
//! given examples, not a claim of full functional equality.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::runtime::RuntimeBridge;
use crate::CompetenceRecord;

/// A behavioral probe: an input argument array and its declared output.
pub struct Probe {
    /// Positional JSON arguments.
    pub args: Value,
    /// The expected result text.
    pub expected: String,
}

impl Probe {
    /// A probe from JSON arguments and an expected output.
    pub fn new(args: Value, expected: impl Into<String>) -> Self {
        Probe { args, expected: expected.into() }
    }
}

/// The name-free semantic identity of an intent: SHA-256 of the normalized shape
/// and the canonical (sorted) behavioral spec. Two intents with the same type
/// signature and the same behavior on the spec inputs share this key — the
/// corpus-independent measure that makes `I → 0` when the behavior is owned.
pub fn semantic_key(shape: &str, probes: &[Probe]) -> String {
    let mut pairs: Vec<String> = probes.iter()
        .map(|p| format!("{}=>{}", p.args, p.expected))
        .collect();
    pairs.sort();
    let mut canon = String::from(shape);
    for p in pairs {
        canon.push('|');
        canon.push_str(&p);
    }
    rcevc_core::sha256(canon.as_bytes()).hex()
}

/// Resolves an intent spec to an owned competence by replay-verified behavioral
/// equivalence. Competences are indexed by shape; resolution replays the spec's
/// probes against each same-shape candidate and returns the first that reproduces
/// all of them. The name is irrelevant — only behavior selects.
pub struct SemanticResolver {
    bridge: RuntimeBridge,
    sigs: HashMap<String, (String, String, String)>,
    by_shape: HashMap<String, Vec<String>>,
}

/// A semantic resolution: the owned operation that covers the spec.
pub struct SemanticHit {
    /// The operation (module.name) whose behavior covers the intent.
    pub handler: String,
    /// The semantic key of the resolved intent spec.
    pub key: String,
}

impl SemanticResolver {
    /// A resolver over a signature table (operation, module, name, shape).
    pub fn new(bridge: RuntimeBridge, entries: &[(String, String, String, String)]) -> Self {
        let sigs = entries.iter()
            .map(|(op, m, n, sh)| (op.clone(), (m.clone(), n.clone(), sh.clone())))
            .collect();
        SemanticResolver { bridge, sigs, by_shape: HashMap::new() }
    }

    /// Index an admitted competence by its shape (the type signature).
    pub fn index(&mut self, record: &CompetenceRecord) {
        if let Some((_, _, sh)) = self.sigs.get(&record.handler_kind) {
            self.by_shape.entry(sh.clone()).or_default().push(record.handler_kind.clone());
        }
    }

    /// The number of distinct shapes indexed (the behavioral type surface).
    pub fn shape_count(&self) -> usize {
        self.by_shape.len()
    }

    /// A canonical behavioral fingerprint of an operation: the hash of its
    /// outputs over a deterministic, type-derived input set. Stronger than a
    /// finite user spec — two operations with the same fingerprint agree on the
    /// canonical battery, tightening semantic equivalence beyond the probes a
    /// caller happens to supply. A trap on an input is recorded as `!trap`.
    pub fn fingerprint(&self, operation: &str) -> Option<String> {
        let (m, n, sh) = self.sigs.get(operation)?;
        let inputs = canonical_inputs(sh);
        let mut canon = String::from(sh.as_str());
        for args in inputs {
            let out = self.bridge.call_general(m, n, sh, &Value::Array(args))
                .unwrap_or_else(|_| "!trap".into());
            canon.push('|');
            canon.push_str(&out);
        }
        Some(rcevc_core::sha256(canon.as_bytes()).hex())
    }

    /// Bounded-exhaustive behavioral fingerprint: enumerate a canonical bounded
    /// domain *exhaustively* (the entire domain for finite types like `bool`)
    /// and hash the outputs. For pure deterministic functions, agreement on the
    /// whole bounded domain **is** full functional equivalence on that domain —
    /// bounded translation validation (the Alive2 insight), deterministic, no
    /// SMT. A trap on an input is recorded as `!trap`.
    pub fn exhaustive_fingerprint(&self, operation: &str) -> Option<String> {
        let (m, n, sh) = self.sigs.get(operation)?;
        let inputs = exhaustive_inputs(sh);
        let mut canon = String::from(sh.as_str());
        for args in inputs {
            let out = self.bridge.call_general(m, n, sh, &Value::Array(args))
                .unwrap_or_else(|_| "!trap".into());
            canon.push('|');
            canon.push_str(&out);
        }
        Some(rcevc_core::sha256(canon.as_bytes()).hex())
    }

    /// Bounded functional equivalence: two operations agree on the entire
    /// canonical bounded domain iff their exhaustive fingerprints match. For a
    /// finite domain (e.g. `bool`) this is complete functional equality.
    pub fn bounded_equivalent(&self, op1: &str, op2: &str) -> bool {
        match (self.exhaustive_fingerprint(op1), self.exhaustive_fingerprint(op2)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    /// Resolve an intent spec: replay the probes against each same-shape
    /// candidate; the first that reproduces every probe covers the intent (G0).
    /// `None` means the behavior is novel relative to the corpus — escalate.
    pub fn resolve(&self, shape: &str, probes: &[Probe]) -> Option<SemanticHit> {
        let candidates = self.by_shape.get(shape)?;
        for op in candidates {
            let (m, n, sh) = self.sigs.get(op)?;
            let covers = probes.iter().all(|p| {
                self.bridge.call_general(m, n, sh, &p.args)
                    .map(|got| got == p.expected)
                    .unwrap_or(false)
            });
            if covers {
                return Some(SemanticHit { handler: op.clone(), key: semantic_key(shape, probes) });
            }
        }
        None
    }
}

/// Split top-level commas respecting nesting.
fn split_args(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let (mut depth, mut cur) = (0, String::new());
    for ch in s.chars() {
        match ch {
            '[' | '(' => { depth += 1; cur.push(ch); }
            ']' | ')' => { depth -= 1; cur.push(ch); }
            ',' if depth == 0 => { out.push(cur.trim().to_string()); cur = String::new(); }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() { out.push(cur.trim().to_string()); }
    out.into_iter().filter(|x| !x.is_empty()).collect()
}

fn base_type(t: &str) -> String {
    t.strip_prefix("&mut").or_else(|| t.strip_prefix('&')).unwrap_or(t).to_string()
}

/// Deterministic sample values for a base type (the canonical battery).
fn sample_values(base: &str) -> Vec<Value> {
    let b = base;
    if b.starts_with("i") || b.starts_with("u") || b == "usize" || b == "isize" {
        return vec![json!(0), json!(1), json!(2), json!(7), json!(-1), json!(2024), json!(1900), json!(2000)];
    }
    if b == "f32" || b == "f64" {
        return vec![json!(0.0), json!(1.0), json!(-1.0), json!(0.5), json!(2.0)];
    }
    if b == "bool" {
        return vec![json!(true), json!(false)];
    }
    if b == "text" {
        return vec![json!(""), json!("a"), json!("abc"), json!("file.txt")];
    }
    if let Some(inner) = b.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        let e = sample_values(inner);
        let pick = |i: usize| e.get(i).cloned().unwrap_or(json!(0));
        return vec![json!([]), json!([pick(1)]), json!([pick(3), pick(1), pick(2)]), json!([pick(1), pick(2), pick(3)])];
    }
    if b == "Complex" {
        return vec![json!({"re": 0.0, "im": 0.0}), json!({"re": 1.0, "im": 2.0}), json!({"re": 3.0, "im": 4.0})];
    }
    vec![json!(0)]
}

/// Canonical input tuples for a shape: a bounded cartesian product of the
/// type-derived sample batteries. Deterministic, so fingerprints are comparable.
pub fn canonical_inputs(shape: &str) -> Vec<Vec<Value>> {
    let argstr = shape.split("->").next().unwrap_or("").trim();
    let argstr = argstr.strip_prefix('(').and_then(|s| s.strip_suffix(')')).unwrap_or(argstr);
    let argtypes = split_args(argstr);
    if argtypes.is_empty() {
        return vec![vec![]];
    }
    let samples: Vec<Vec<Value>> = argtypes.iter().map(|t| sample_values(&base_type(t))).collect();
    let mut combos: Vec<Vec<Value>> = vec![vec![]];
    for s in &samples {
        let mut next = Vec::new();
        'outer: for c in &combos {
            for v in s {
                let mut nc = c.clone();
                nc.push(v.clone());
                next.push(nc);
                if next.len() >= 24 { break 'outer; }
            }
        }
        combos = next;
    }
    combos
}

/// Exhaustive values for a base type over a canonical bounded domain. `bool` is
/// complete (its whole domain); integers a bounded range; the rest a dense grid.
fn exhaustive_values(base: &str) -> Vec<Value> {
    let b = base;
    if b == "bool" {
        return vec![json!(true), json!(false)];
    }
    if b.starts_with('i') || b.starts_with('u') || b == "usize" || b == "isize" {
        return (-50..=50).map(|i| json!(i)).collect();
    }
    if b == "f32" || b == "f64" {
        return (-10..=10).map(|i| json!(i as f64 * 0.5)).collect();
    }
    if b == "text" {
        return vec![json!(""), json!("a"), json!("b"), json!("abc"), json!("file.txt"), json!("*.txt")];
    }
    if let Some(inner) = b.strip_prefix("List[").and_then(|s| s.strip_suffix(']')) {
        if inner == "i64" || inner == "usize" {
            return vec![json!([]), json!([0]), json!([1]), json!([-1]), json!([1, 2]),
                json!([2, 1]), json!([1, 2, 3]), json!([-1, 0, 1]), json!([3, 1, 2])];
        }
        let e = exhaustive_values(inner);
        let pick = |i: usize| e.get(i).cloned().unwrap_or(json!(0));
        return vec![json!([]), json!([pick(0)]), json!([pick(1), pick(2)])];
    }
    if b == "Complex" {
        return vec![json!({"re": 0.0, "im": 0.0}), json!({"re": 1.0, "im": 0.0}),
            json!({"re": 0.0, "im": 1.0}), json!({"re": -1.0, "im": 2.0}), json!({"re": 3.0, "im": 4.0})];
    }
    if b == "Op" {
        return ["Add", "Mul", "Sub", "Max", "Min", "Abs", "Even", "Odd", "Not", "Neg"]
            .iter().map(|t| json!(t)).collect();
    }
    vec![json!(0)]
}

/// Exhaustive input tuples over the canonical bounded domain (cartesian product,
/// capped). For finite argument types this enumerates the entire domain.
pub fn exhaustive_inputs(shape: &str) -> Vec<Vec<Value>> {
    let argstr = shape.split("->").next().unwrap_or("").trim();
    let argstr = argstr.strip_prefix('(').and_then(|s| s.strip_suffix(')')).unwrap_or(argstr);
    let argtypes = split_args(argstr);
    if argtypes.is_empty() {
        return vec![vec![]];
    }
    let domains: Vec<Vec<Value>> = argtypes.iter().map(|t| exhaustive_values(&base_type(t))).collect();
    let mut combos: Vec<Vec<Value>> = vec![vec![]];
    for d in &domains {
        let mut next = Vec::new();
        'outer: for c in &combos {
            for v in d {
                let mut nc = c.clone();
                nc.push(v.clone());
                next.push(nc);
                if next.len() >= 4096 { break 'outer; }
            }
        }
        combos = next;
    }
    combos
}
