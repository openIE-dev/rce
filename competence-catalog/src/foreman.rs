//! The competence-before-model foreman — the gear ladder's G0 rung as an engine.
//!
//! A resolve first looks up the intent in the catalog. A hit executes the owned
//! competence through the reference runtime at the interpreter floor (≈1 µJ,
//! zero model calls). A miss escalates to the proposer: the operation is
//! executed, the result crystallized into a record, and the record admitted only
//! by per-record replay. The next instance of that intent hits G0. The ecology
//! meter records the reuse curve and the falling cost per competent act — the
//! Law of Coding falsifier, `I = H_task − H_corpus → 0`, priced in joules.
//!
//! This is the public, working cascade over the real stdlib basis. The richer
//! private engine (hindsight, promotion, the full G1–G5 ladder) is separate; this
//! is the G0 competence floor it rests on.

use std::collections::{BTreeMap, HashMap, HashSet};

use rcevc_core::{Gate, Outcome};
use serde_json::Value;

use crate::runtime::{GeneralReplayChecker, RuntimeBridge};
use crate::{intent_key, Catalog, CompetenceRecord, Meter, Provenance};

/// The outcome of resolving one intent.
pub struct Resolved {
    /// The computed result text.
    pub result: String,
    /// Cost of this act, microjoules (G0 floor on a hit, proposer on a miss).
    pub cost_uj: u64,
    /// Whether the catalog owned the competence (G0 hit, zero model calls).
    pub hit: bool,
}

/// A competence-before-model foreman over the real stdlib basis.
pub struct Foreman {
    cat: Catalog,
    bridge: RuntimeBridge,
    sigs: HashMap<String, (String, String, String)>,
}

impl Foreman {
    /// A foreman whose crystallized admissions are replay-gated. `entries` is the
    /// signature table (operation, module, name, shape).
    pub fn new(bridge: RuntimeBridge, entries: &[(String, String, String, String)]) -> Result<Self, rcevc_core::Reason> {
        let sigs = entries.iter()
            .map(|(op, m, n, sh)| (op.clone(), (m.clone(), n.clone(), sh.clone())))
            .collect();
        let checker = GeneralReplayChecker::from_index(bridge.clone(), entries);
        let cat = Catalog::with_gate(Gate::new().with(checker))?;
        Ok(Foreman { cat, bridge, sigs })
    }

    /// Seed a pre-verified basis competence by replaying its spectest.
    pub fn seed(&mut self, operation: &str, shape: &str, probe_args: &Value, probe_output: &str) -> Outcome {
        let rec = CompetenceRecord {
            input_key: intent_key(operation, shape),
            handler_kind: operation.to_string(),
            probe_input: probe_args.to_string(),
            probe_output: probe_output.to_string(),
            result: String::new(),
            cost_uj: self.cat.costs().g4,
            provenance: Provenance::Reported,
        };
        self.cat.admit(rec)
    }

    fn execute(&self, operation: &str, args: &Value) -> Result<String, String> {
        let (m, n, sh) = self.sigs.get(operation)
            .ok_or_else(|| format!("no signature for {operation}"))?;
        self.bridge.call_general(m, n, sh, args)
    }

    /// Resolve an intent: G0 hit executes owned competence at the floor; a miss
    /// pays the proposer, crystallizes the result, and admits it by replay.
    pub fn resolve(&mut self, operation: &str, shape: &str, args: &Value) -> Result<Resolved, String> {
        let owned = self.cat.hit_intent(operation, shape); // meters the hit/miss
        match owned {
            Some(_) => {
                let result = self.execute(operation, args)?;
                Ok(Resolved { result, cost_uj: self.cat.costs().g0, hit: true })
            }
            None => {
                // The proposer computes the result, then crystallizes it.
                let result = self.execute(operation, args)?;
                let rec = CompetenceRecord {
                    input_key: intent_key(operation, shape),
                    handler_kind: operation.to_string(),
                    probe_input: args.to_string(),
                    probe_output: result.clone(),
                    result: result.clone(),
                    cost_uj: self.cat.costs().g4,
                    provenance: Provenance::Modelled,
                };
                self.cat.admit(rec); // replay-gated: a bad crystallization is refused
                Ok(Resolved { result, cost_uj: self.cat.costs().g4 + self.cat.costs().per_check, hit: false })
            }
        }
    }

    /// The ecology meter (reuse rate, cost per act, proposer calls).
    pub fn meter(&self) -> Meter {
        self.cat.meter()
    }

    /// Distinct admitted competences (|H_corpus|).
    pub fn corpus_size(&self) -> usize {
        self.cat.meter().catalog_size
    }

    /// Export the admitted competence as a feed for another engine. Each record
    /// carries its spectest so the receiver re-verifies by replay (circulation
    /// never bypasses the admission law).
    pub fn export(&self) -> Vec<CompetenceRecord> {
        self.cat.records()
    }

    /// Import a feed from another engine. Every novel record is admitted only by
    /// per-record replay; already-owned intents are skipped (idempotent). A
    /// negatively-transferring record is refused at the border, not merged.
    pub fn import(&mut self, feed: &[CompetenceRecord]) -> ImportReport {
        let owned: HashSet<String> = self.cat.records().iter().map(|r| r.input_key.clone()).collect();
        let (mut accepted, mut refused, mut skipped) = (0, 0, 0);
        for r in feed {
            if owned.contains(&r.input_key) {
                skipped += 1;
                continue;
            }
            if self.cat.admit(r.clone()).is_accepted() {
                accepted += 1;
            } else {
                refused += 1;
            }
        }
        ImportReport { total: feed.len(), accepted, refused, skipped }
    }
}

/// The outcome of importing a competence feed across engines.
#[derive(Clone, Copy, Debug)]
pub struct ImportReport {
    /// Records in the feed.
    pub total: usize,
    /// Novel records admitted by replay.
    pub accepted: usize,
    /// Records refused (spectest did not reproduce — negative transfer guarded).
    pub refused: usize,
    /// Already-owned intents skipped (idempotent re-import).
    pub skipped: usize,
}

/// CALM monotone merge of two feeds: the union keyed by the name-free intent
/// key. Commutative, associative, idempotent — replicas converge without
/// coordination, and re-import never duplicates or regresses competence.
pub fn merge_feeds(a: &[CompetenceRecord], b: &[CompetenceRecord]) -> Vec<CompetenceRecord> {
    let mut map: BTreeMap<String, CompetenceRecord> = BTreeMap::new();
    for r in a.iter().chain(b.iter()) {
        map.entry(r.input_key.clone()).or_insert_with(|| r.clone());
    }
    map.into_values().collect()
}
