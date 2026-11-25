//! The public gear-ladder engine — deterministic automated coding as a
//! competence-before-model cascade (compile CHARTER gears G0–G5, Appendix D.12).
//!
//! An intent spec (shape + behavioral probes) is resolved by the cheapest gear
//! that covers it, and every gear above G0 passes the same replay gate before
//! its output is used:
//!
//! - **G0** — stored competence by semantic key (owned, ≈1 µJ, zero model calls).
//! - **G1** — proven rewrite rules (a deterministic rule table).
//! - **G2** — cheapest extraction: search the basis library for an operation
//!   whose behavior covers the spec, replay-verify, and crystallize it.
//! - **G3–G5** — a proposer (a model) names a candidate; its every output is
//!   replay-verified before admission. The proposer is an interface; the engine
//!   never trusts it.
//!
//! The meter records which gear resolved each act and its cost — the falsifier
//! that competence precedes models and cost per act decays as the corpus grows.
//! This is a fresh public build; it shares no code with the private engine.

use std::collections::HashMap;

use serde_json::Value;

use crate::learning::{Hindsight, Promotion};
use crate::runtime::RuntimeBridge;
use crate::semantic::{semantic_key, Probe, SemanticResolver};
use crate::synthesis::{Composition, Synthesizer};
use crate::{intent_key, CompetenceRecord, Provenance};

/// A proposer (G3–G5): a model that names a candidate operation for an intent.
/// Its output is never trusted — the engine replay-verifies before admission.
pub trait Proposer {
    /// Propose an operation (module.name) claimed to cover the spec, or `None`.
    fn propose(&self, shape: &str, probes: &[Probe]) -> Option<String>;
    /// The gear this proposer sits on (3..=5).
    fn gear(&self) -> u8;
}

/// A proven rewrite rule (G1): maps a spec to a known operation deterministically.
pub struct Rule {
    /// The shape the rule applies to.
    pub shape: String,
    /// A predicate on the probes selecting the rule.
    pub applies: Box<dyn Fn(&[Probe]) -> bool + Send + Sync>,
    /// The operation the rule rewrites to.
    pub handler: String,
}

/// The outcome of resolving one intent through the ladder.
pub struct Resolution {
    /// The gear that covered the intent (0..=5).
    pub gear: u8,
    /// The operation whose behavior covers the spec.
    pub handler: String,
    /// Cost of this act, microjoules (gear-ordered).
    pub cost_uj: u64,
}

/// Cost per gear (microjoules), competence-before-model: G0 at the interpreter
/// floor, rising through rules, extraction, and model proposers.
pub const GEAR_COST: [u64; 6] = [1, 10, 1_000, 5_000_000, 5_000_000, 5_000_000];

/// The gear-ladder engine over the real stdlib basis.
pub struct Engine {
    g0: SemanticResolver,
    rules: Vec<Rule>,
    bridge: RuntimeBridge,
    sigs: HashMap<String, (String, String, String)>,
    proposer: Box<dyn Proposer>,
    synth: Synthesizer,
    /// Synthesized compositions reused at G0: (shape, program-backed competence).
    programs: Vec<(String, Composition)>,
    /// The public learning loop: hindsight over verified resolutions.
    pub hindsight: Hindsight,
    /// Acts resolved per gear (the ecology falsifier).
    pub gear_acts: [u64; 6],
}

impl Engine {
    /// An engine over a signature table (operation, module, name, shape), a rule
    /// table (G1), and a proposer (G3–G5). The basis is indexed for G0/G2.
    pub fn new(
        bridge: RuntimeBridge,
        entries: &[(String, String, String, String)],
        rules: Vec<Rule>,
        proposer: Box<dyn Proposer>,
    ) -> Self {
        // G0 starts empty: it holds only crystallized competence. The basis
        // library lives in `sigs` for G2 extraction, which crystallizes into G0.
        let g0 = SemanticResolver::new(bridge.clone(), entries);
        let synth = Synthesizer::new(bridge.clone(), entries);
        let sigs = entries.iter()
            .map(|(op, m, n, sh)| (op.clone(), (m.clone(), n.clone(), sh.clone())))
            .collect();
        Engine { g0, rules, bridge, sigs, proposer, synth, programs: Vec::new(), hindsight: Hindsight::new(), gear_acts: [0; 6] }
    }

    /// Whether a handler (single op or synthesized program) covers a spec.
    fn covers_handler(&self, handler: &str, probes: &[Probe]) -> bool {
        if self.sigs.contains_key(handler) {
            return self.covers(handler, probes);
        }
        self.programs.iter().any(|(_, comp)| comp.expr == handler && self.covers_program(comp, probes))
    }

    /// Record a verified resolution in hindsight (the learning loop).
    fn learn(&mut self, shape: &str, probes: &[Probe], handler: &str, gear: u8) {
        self.hindsight.record(&semantic_key(shape, probes), shape, handler, gear);
    }

    /// Replay-verify a synthesized program against a spec (the gate).
    fn covers_program(&self, comp: &Composition, probes: &[Probe]) -> bool {
        probes.iter().all(|p| {
            let first = match p.args.as_array().and_then(|a| a.first()) {
                Some(f) => f,
                None => return false,
            };
            let event = format!("Calc(x: {first})");
            self.bridge.run_caller("synth", &comp.program, &event)
                .map(|got| got == p.expected).unwrap_or(false)
        })
    }

    /// Replay-verify that an operation covers a spec (the gate every gear passes).
    fn covers(&self, op: &str, probes: &[Probe]) -> bool {
        let Some((m, n, sh)) = self.sigs.get(op) else { return false };
        probes.iter().all(|p| {
            self.bridge.call_general(m, n, sh, &p.args)
                .map(|got| got == p.expected).unwrap_or(false)
        })
    }

    /// Crystallize a verified operation into G0 so the next instance hits the floor.
    fn crystallize(&mut self, op: &str, shape: &str, probes: &[Probe]) {
        if let Some(first) = probes.first() {
            self.g0.index(&CompetenceRecord {
                input_key: intent_key(op, shape),
                handler_kind: op.to_string(),
                probe_input: first.args.to_string(),
                probe_output: first.expected.clone(),
                result: String::new(),
                cost_uj: GEAR_COST[2],
                provenance: Provenance::Modelled,
            });
        }
    }

    /// Resolve an intent spec by the cheapest covering gear. `None` if no gear
    /// (including the proposer) covers it.
    pub fn resolve(&mut self, shape: &str, probes: &[Probe]) -> Option<Resolution> {
        let key = semantic_key(shape, probes);
        // G0 learned cache: hindsight recalls a prior verified resolution. The
        // recall is still replay-verified — learning never bypasses the gate.
        if let Some(exp) = self.hindsight.recall(&key) {
            let handler = exp.handler.clone();
            if self.covers_handler(&handler, probes) {
                self.gear_acts[0] += 1;
                self.learn(shape, probes, &handler, 0);
                return Some(Resolution { gear: 0, handler, cost_uj: GEAR_COST[0] });
            }
        }
        // G0: owned competence by semantic key.
        if let Some(hit) = self.g0.resolve(shape, probes) {
            self.gear_acts[0] += 1;
            self.learn(shape, probes, &hit.handler, 0);
            return Some(Resolution { gear: 0, handler: hit.handler, cost_uj: GEAR_COST[0] });
        }
        // G0 also reuses previously synthesized programs.
        for (sh, comp) in &self.programs {
            if sh == shape && self.covers_program(comp, probes) {
                let handler = comp.expr.clone();
                self.gear_acts[0] += 1;
                self.learn(shape, probes, &handler, 0);
                return Some(Resolution { gear: 0, handler, cost_uj: GEAR_COST[0] });
            }
        }
        // G1: proven rewrite rules, each replay-verified.
        for rule in &self.rules {
            if rule.shape == shape && (rule.applies)(probes) && self.covers(&rule.handler, probes) {
                let handler = rule.handler.clone();
                self.crystallize(&handler, shape, probes);
                self.gear_acts[1] += 1;
                self.learn(shape, probes, &handler, 1);
                return Some(Resolution { gear: 1, handler, cost_uj: GEAR_COST[1] });
            }
        }
        // G2: cheapest extraction, candidates reranked by reuse (promotion).
        let candidates: Vec<String> = self.sigs.iter()
            .filter(|(_, (_, _, sh))| sh == shape).map(|(op, _)| op.clone()).collect();
        for op in Promotion::rerank(&candidates, &self.hindsight) {
            if self.covers(&op, probes) {
                self.crystallize(&op, shape, probes);
                self.gear_acts[2] += 1;
                self.learn(shape, probes, &op, 2);
                return Some(Resolution { gear: 2, handler: op, cost_uj: GEAR_COST[2] });
            }
        }
        // G2 learned synthesis: compose basis operations, experienced parts first.
        if let Some(comp) = self.synth.synthesize_learned(shape, probes, &self.hindsight) {
            let handler = comp.expr.clone();
            self.programs.push((shape.to_string(), comp));
            self.gear_acts[2] += 1;
            self.learn(shape, probes, &handler, 2);
            return Some(Resolution { gear: 2, handler, cost_uj: GEAR_COST[2] });
        }
        // G3–G5: the proposer names a candidate; the gate verifies before use.
        let gear = self.proposer.gear().clamp(3, 5);
        if let Some(candidate) = self.proposer.propose(shape, probes) {
            if self.covers(&candidate, probes) {
                self.crystallize(&candidate, shape, probes);
                self.gear_acts[gear as usize] += 1;
                self.learn(shape, probes, &candidate, gear);
                return Some(Resolution { gear, handler: candidate, cost_uj: GEAR_COST[gear as usize] });
            }
        }
        None
    }

    /// Total acts resolved.
    pub fn acts(&self) -> u64 {
        self.gear_acts.iter().sum()
    }

    /// Mean cost per act (decays toward the floor as G0 reuse grows).
    pub fn cost_per_act(&self) -> f64 {
        let acts = self.acts();
        if acts == 0 { return 0.0; }
        let total: u64 = self.gear_acts.iter().enumerate()
            .map(|(g, n)| n * GEAR_COST[g]).sum();
        total as f64 / acts as f64
    }
}

/// A deterministic proposer for tests/examples: names a fixed operation. The
/// engine still replay-verifies it — a wrong name is refused at the gate.
pub struct FixedProposer {
    /// The operation this proposer always names.
    pub handler: String,
    /// The gear it sits on.
    pub gear: u8,
}

impl Proposer for FixedProposer {
    fn propose(&self, _shape: &str, _probes: &[Probe]) -> Option<String> {
        Some(self.handler.clone())
    }
    fn gear(&self) -> u8 {
        self.gear
    }
}

/// A silent proposer (no model): G3–G5 never cover, so resolution stops at G2.
pub struct NoProposer;

impl Proposer for NoProposer {
    fn propose(&self, _shape: &str, _probes: &[Probe]) -> Option<String> {
        None
    }
    fn gear(&self) -> u8 {
        3
    }
}

/// Convenience: a probe from JSON args and an expected output.
pub fn probe(args: Value, expected: impl Into<String>) -> Probe {
    Probe::new(args, expected)
}
