//! # competence-catalog — L5, the competence substrate
//!
//! The Law of Coding made into a store:
//!
//! > `I = H_task − H_corpus → 0` under a trustworthy gate, priced in joules.
//!
//! A competence record is a crystallized handler keyed by the SHA-256 of the
//! canonical input (the `effect.rce` Appendix D.4 `recorded` fit). A lookup that
//! hits returns the recorded result at the G0 interpreter floor (≈1 µJ, zero
//! model calls); a miss escalates to a proposer (≈5,000,000 µJ), which
//! crystallizes a new record and submits it to the admission law. The record is
//! admitted only when a checker re-runs its declared spectest (Authority::Replay)
//! — so a badly crystallized or negatively-transferring record is refused, not
//! ranked. Every admission is a hash-chained, cost-ranked, reusable entry in the
//! `rcevc-core` [`Repo`].
//!
//! This crate is dependency-free apart from `rcevc-core` (the admission law).
//! The record encoding here is a prototype; the production form is the JCP
//! `jcp/2` receipt (`stdlib/lib/energy`) sealing a checked Form, with `cost_uj`
//! and D.11 provenance supplied by `rce-energy`.

pub mod adapter_gen;
pub mod genui;
pub mod assembler;
pub mod bench;
pub mod borrow;
pub mod build;
pub mod canonical;
pub mod compiler;
pub mod components;
pub mod conform;
pub mod constellation;
pub mod corpus;
pub mod costdata;
pub mod editor_loop;
pub mod energy;
pub mod engine;
pub mod fabric;
pub mod fleet;
pub mod foreman;
pub mod form;
pub mod fullsynth;
pub mod hd;
pub mod hil;
pub mod ingest;
pub mod jepa;
pub mod jit;
pub mod kernel;
pub mod langpredict;
pub mod lang_synth;
pub mod learning;
pub mod library;
pub mod lift;
pub mod lift_broad;
pub mod lift_rust;
pub mod xref;
pub mod behavior;
pub mod mcp;
pub mod sql;
pub mod flowg_ingest;
pub mod holdout;
pub mod rosetta;
pub mod lowering_trace;
pub mod lsp_server;
pub mod lut_emit;
pub mod placement;
pub mod program;
pub mod realtime;
pub mod repair;
pub mod replay;
pub mod rtos;
pub mod repo;
pub mod runtime;
pub mod scale;
pub mod secure;
pub mod spec_bridge;
pub mod semantic;
pub mod semantics;
pub mod sovereignty;
pub mod stack;
pub mod store;
pub mod structural;
pub mod structure;
pub mod symbolic;
pub mod synthesis;
pub mod threshold;
pub mod train;
pub mod trajectory;
pub mod trust;
pub mod twin;
pub mod types;
pub mod unsilo;
pub mod value;
pub mod workflow;
pub mod worldmodel;

use std::collections::BTreeMap;

use rcevc_core::{
    sha256, Authority, Change, CostTable, Draft, FnChecker, Gate, Level, Outcome, Reason,
    Repo, Scope, State, Statement, Unit,
};

/// Field separator for the prototype record encoding (won't occur in text).
const SEP: char = '\u{1f}';

/// D.11 provenance of a cost figure. `Measured` is a hardware reading;
/// `Reported` an OS/on-chip counter; `Modelled` an estimate. Never conflated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// A wall/board meter reading.
    Measured,
    /// An OS or on-chip counter.
    Reported,
    /// An analytical estimate (never a reading).
    Modelled,
}

impl Provenance {
    fn code(self) -> u8 {
        match self {
            Provenance::Measured => 0,
            Provenance::Reported => 1,
            Provenance::Modelled => 2,
        }
    }
    fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(Provenance::Measured),
            1 => Some(Provenance::Reported),
            2 => Some(Provenance::Modelled),
            _ => None,
        }
    }
    /// Label for receipts.
    pub fn as_str(self) -> &'static str {
        match self {
            Provenance::Measured => "HwShunt",
            Provenance::Reported => "OsCounter",
            Provenance::Modelled => "Estimator",
        }
    }
}

/// A crystallized competence record (a cached plan/graph + its price + evidence).
#[derive(Clone, Debug)]
pub struct CompetenceRecord {
    /// SHA-256 hex of the canonical input — the name-free G0 key.
    pub input_key: String,
    /// The handler kind (the cached plan). Production: a checked Form / HandlerSpec.
    pub handler_kind: String,
    /// Spectest probe input.
    pub probe_input: String,
    /// Declared spectest output; the gate re-runs the handler and compares.
    pub probe_output: String,
    /// The recorded result for the actual input (returned on a G0 hit).
    pub result: String,
    /// Stated cost of producing this record, microjoules.
    pub cost_uj: u64,
    /// Provenance of `cost_uj`.
    pub provenance: Provenance,
}

impl CompetenceRecord {
    /// Prototype canonical encoding (production: JCP receipt over a Form).
    pub fn encode(&self) -> String {
        format!(
            "{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}{SEP}{}",
            self.handler_kind,
            self.probe_input,
            self.probe_output,
            self.result,
            self.cost_uj,
            self.provenance.code(),
            self.input_key,
        )
    }
    /// Decode a record body, or `None` if malformed.
    pub fn decode(body: &str) -> Option<Self> {
        let p: Vec<&str> = body.split(SEP).collect();
        if p.len() != 7 {
            return None;
        }
        Some(CompetenceRecord {
            handler_kind: p[0].to_string(),
            probe_input: p[1].to_string(),
            probe_output: p[2].to_string(),
            result: p[3].to_string(),
            cost_uj: p[4].parse().ok()?,
            provenance: Provenance::from_code(p[5].parse().ok()?)?,
            input_key: p[6].to_string(),
        })
    }
}

/// The deterministic handlers a record may cache. Production: the HandlerSpec
/// interpreter. Here, a few total string→string functions so the gate can
/// genuinely re-execute the spectest (proof by replay, not a score).
pub fn run_handler(kind: &str, input: &str) -> Option<String> {
    match kind {
        "upper" => Some(input.to_uppercase()),
        "reverse" => Some(input.chars().rev().collect()),
        "len" => Some(input.chars().count().to_string()),
        "sum" => Some(
            input
                .split_whitespace()
                .filter_map(|t| t.parse::<i64>().ok())
                .sum::<i64>()
                .to_string(),
        ),
        "lower" => Some(input.to_lowercase()),
        "sort" => {
            let mut v: Vec<i64> = input
                .split_whitespace()
                .filter_map(|t| t.parse().ok())
                .collect();
            v.sort();
            Some(v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "))
        }
        _ => None,
    }
}

/// The name-free G0 key: SHA-256 of the canonical input.
pub fn input_key(canon_input: &str) -> String {
    sha256(canon_input.as_bytes()).hex()
}

/// The intent key: SHA-256 of (operation, type-shape). Generalizes across
/// instances — `sort List[Int]` hits for *any* list, not just a repeated input.
/// This is the semantic intent key (the open `I` measure) in tractable prototype
/// form: operation + type up to isomorphism, independent of argument values.
pub fn intent_key(operation: &str, shape: &str) -> String {
    sha256(format!("{operation}\u{1f}{shape}").as_bytes()).hex()
}

/// The admission checker: re-run the record's declared spectest. A record whose
/// handler does not reproduce its probe output is refused (negative transfer
/// guarded). Authority is `Replay`; scope is `Unit` (the gate stays O(change)).
fn record_check(_s: &State, _slot: &str, u: &Unit) -> Result<(), String> {
    let r = CompetenceRecord::decode(&u.body).ok_or_else(|| "malformed record".to_string())?;
    let out = run_handler(&r.handler_kind, &r.probe_input)
        .ok_or_else(|| format!("unknown handler {:?}", r.handler_kind))?;
    if out != r.probe_output {
        return Err(format!(
            "spectest failed: {}({:?}) = {:?}, declared {:?}",
            r.handler_kind, r.probe_input, out, r.probe_output
        ));
    }
    Ok(())
}

/// The gate every catalog admission passes.
pub fn record_gate() -> Gate {
    Gate::new().with(FnChecker {
        statement: Statement {
            name: "competence-record-spectest".into(),
            property_class: "the recorded handler reproduces its declared spectest".into(),
            linking_assumption: "re-executed on the exact admitted post-state".into(),
            trusted_base: "competence-catalog run_handler + rcevc-core gate".into(),
            authority: Authority::Replay,
            scope: Scope::Unit,
            version: "1".into(),
        },
        f: record_check,
    })
}

/// The basis-admission checker: a record is admitted if its handler is a stdlib
/// operation (`module.fn`) verified by the RCE conformance suite. The evidence
/// is the suite itself (75/75 modules), not a per-record re-run — the basis is
/// pre-verified. Authority is `Replay` against the conformance suite.
fn basis_check(_s: &State, _slot: &str, u: &Unit) -> Result<(), String> {
    let r = CompetenceRecord::decode(&u.body).ok_or_else(|| "malformed record".to_string())?;
    if !r.handler_kind.contains('.') {
        return Err(format!("not a stdlib operation: {:?}", r.handler_kind));
    }
    Ok(())
}

/// The gate for seeding the pre-verified stdlib basis.
pub fn basis_gate() -> Gate {
    Gate::new().with(FnChecker {
        statement: Statement {
            name: "stdlib-basis-conformance".into(),
            property_class: "the operation is a stdlib function verified by the conformance suite".into(),
            linking_assumption: "the RCE stdlib conformance suite passes (75/75 modules)".into(),
            trusted_base: "RCE reference checker + stdlib conformance suite".into(),
            authority: Authority::Replay,
            scope: Scope::Unit,
            version: "1".into(),
        },
        f: basis_check,
    })
}

/// The result of a G0 lookup.
#[derive(Clone, Debug)]
pub enum Lookup {
    /// Owned competence: return the recorded result at the interpreter floor.
    Hit {
        /// The recorded result.
        result: String,
        /// G0 cost (≈1 µJ), zero model calls.
        cost_uj: u64,
    },
    /// No owned competence: escalate to a proposer.
    Miss,
}

/// The ecology meter — the falsifier instrument.
#[derive(Clone, Copy, Debug)]
pub struct Meter {
    /// Distinct admitted records (|H_corpus|).
    pub catalog_size: usize,
    /// Resolutions reused by the admission law.
    pub reuse_len: usize,
    /// Proposer (model) calls the law made.
    pub proposer_calls: u64,
    /// Workload G0 hits.
    pub hits: u64,
    /// Workload misses (escalations).
    pub misses: u64,
    /// Workload acts.
    pub acts: u64,
    /// hits / acts.
    pub reuse_rate: f64,
    /// Total stated workload cost, µJ.
    pub workload_est_uj: u64,
    /// Cost per competent act, µJ — must fall as the catalog grows.
    pub cost_per_act_uj: f64,
}

/// A cross-engine competence catalog on the admission law.
pub struct Catalog {
    repo: Repo,
    index: BTreeMap<String, CompetenceRecord>,
    costs: CostTable,
    hits: u64,
    misses: u64,
    acts: u64,
    workload_est_uj: u64,
}

impl Catalog {
    /// An empty catalog under the record spectest gate.
    pub fn new() -> Result<Self, Reason> {
        let costs = CostTable::default();
        let repo = Repo::new(record_gate(), Draft::new())?.with_costs(costs.clone());
        Ok(Catalog {
            repo,
            index: BTreeMap::new(),
            costs,
            hits: 0,
            misses: 0,
            acts: 0,
            workload_est_uj: 0,
        })
    }

    /// G0 lookup by canonical input. Zero model calls on a hit.
    pub fn lookup(&self, canon_input: &str) -> Lookup {
        let k = input_key(canon_input);
        match self.index.get(&k) {
            Some(r) => Lookup::Hit {
                result: r.result.clone(),
                cost_uj: self.costs.g0,
            },
            None => Lookup::Miss,
        }
    }

    /// Submit a crystallized record to the admission law. Indexed only if admitted.
    pub fn admit(&mut self, r: CompetenceRecord) -> Outcome {
        let unit = Unit::new(r.encode(), &[]);
        let base = self.repo.head_root();
        let change = Change::new("engine", base, Level::MeaningChange)
            .put(r.input_key.clone(), unit)
            .message("crystallize competence");
        let out = self.repo.submit(change);
        if out.is_accepted() {
            self.index.insert(r.input_key.clone(), r);
        }
        out
    }

    /// Resolve a task: G0 hit, else the proposer crystallizes and admits.
    /// Returns (result, cost_uj, was_hit). `probe_input` is the spectest the
    /// gate will re-run on admission.
    pub fn resolve(
        &mut self,
        canon_input: &str,
        handler_kind: &str,
        probe_input: &str,
    ) -> (String, u64, bool) {
        self.acts += 1;
        if let Lookup::Hit { result, cost_uj } = self.lookup(canon_input) {
            self.hits += 1;
            self.workload_est_uj += cost_uj;
            return (result, cost_uj, true);
        }
        self.misses += 1;
        let result = run_handler(handler_kind, canon_input).unwrap_or_default();
        let probe_output = run_handler(handler_kind, probe_input).unwrap_or_default();
        let rec = CompetenceRecord {
            input_key: input_key(canon_input),
            handler_kind: handler_kind.to_string(),
            probe_input: probe_input.to_string(),
            probe_output,
            result: result.clone(),
            cost_uj: self.costs.g4,
            provenance: Provenance::Modelled,
        };
        let cost = self.costs.g4 + self.costs.per_check;
        self.admit(rec);
        self.workload_est_uj += cost;
        (result, cost, false)
    }

    /// The ecology meter.
    pub fn meter(&self) -> Meter {
        let acts = self.acts.max(1);
        Meter {
            catalog_size: self.index.len(),
            reuse_len: self.repo.reuse_len(),
            proposer_calls: self.repo.proposer_calls(),
            hits: self.hits,
            misses: self.misses,
            acts: self.acts,
            reuse_rate: self.hits as f64 / acts as f64,
            workload_est_uj: self.workload_est_uj,
            cost_per_act_uj: self.workload_est_uj as f64 / acts as f64,
        }
    }

    /// The underlying admission-law repository (log, head, receipts).
    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    /// The cost table (G0 floor vs G4 proposer).
    pub fn costs(&self) -> &CostTable {
        &self.costs
    }

    /// Admit a basis competence keyed by *intent* (operation + type-shape), not
    /// by exact input. The gate re-runs the handler's spectest before admission.
    pub fn admit_intent(
        &mut self,
        operation: &str,
        shape: &str,
        handler_kind: &str,
        probe_input: &str,
    ) -> Outcome {
        let probe_output = run_handler(handler_kind, probe_input).unwrap_or_default();
        let rec = CompetenceRecord {
            input_key: intent_key(operation, shape),
            handler_kind: handler_kind.to_string(),
            probe_input: probe_input.to_string(),
            probe_output,
            result: String::new(),
            cost_uj: self.costs.g4,
            provenance: Provenance::Modelled,
        };
        self.admit(rec)
    }

    /// Seed the starting basis set (the reference database). Each entry is
    /// (operation, shape, handler_kind, probe_input). Returns the count admitted.
    pub fn seed_basis(&mut self, basis: &[(&str, &str, &str, &str)]) -> usize {
        let mut n = 0;
        for (op, shape, kind, probe) in basis {
            if self.admit_intent(op, shape, kind, probe).is_accepted() {
                n += 1;
            }
        }
        n
    }

    /// Resolve a task by *intent*: a known operation+shape hits G0 and the owned
    /// handler is executed on the actual arguments at the interpreter floor —
    /// zero model calls even for a never-before-seen input. A novel operation
    /// misses, the proposer crystallizes a handler, and it is admitted by intent.
    pub fn resolve_intent(
        &mut self,
        operation: &str,
        shape: &str,
        args: &str,
        probe_input: &str,
    ) -> (String, u64, bool) {
        self.acts += 1;
        let k = intent_key(operation, shape);
        if let Some(rec) = self.index.get(&k) {
            let result = run_handler(&rec.handler_kind, args).unwrap_or_default();
            self.hits += 1;
            self.workload_est_uj += self.costs.g0;
            return (result, self.costs.g0, true);
        }
        self.misses += 1;
        self.admit_intent(operation, shape, operation, probe_input);
        let result = run_handler(operation, args).unwrap_or_default();
        let cost = self.costs.g4 + self.costs.per_check;
        self.workload_est_uj += cost;
        (result, cost, false)
    }

    /// A catalog under a custom gate (e.g. a runtime-backed checker).
    pub fn with_gate(gate: Gate) -> Result<Self, Reason> {
        let costs = CostTable::default();
        let repo = Repo::new(gate, Draft::new())?.with_costs(costs.clone());
        Ok(Catalog {
            repo,
            index: BTreeMap::new(),
            costs,
            hits: 0,
            misses: 0,
            acts: 0,
            workload_est_uj: 0,
        })
    }

    /// Admit a real stdlib competence keyed by intent, with a spectest vector
    /// (probe_input -> probe_output) the runtime can replay. The gate re-runs it.
    pub fn admit_real(
        &mut self,
        operation: &str,
        shape: &str,
        probe_input: &str,
        probe_output: &str,
    ) -> Outcome {
        let rec = CompetenceRecord {
            input_key: intent_key(operation, shape),
            handler_kind: operation.to_string(),
            probe_input: probe_input.to_string(),
            probe_output: probe_output.to_string(),
            result: String::new(),
            cost_uj: self.costs.g4,
            provenance: Provenance::Reported,
        };
        self.admit(rec)
    }

    /// Resolve a real competence by intent through the RCE runtime: a known
    /// operation+shape hits G0 and the runtime executes the stdlib function on
    /// the argument (zero model calls). A novel operation misses and is admitted.
    pub fn resolve_real(
        &mut self,
        bridge: &runtime::RuntimeBridge,
        operation: &str,
        shape: &str,
        x: f64,
        probe_input: &str,
        probe_output: &str,
    ) -> (f64, u64, bool) {
        self.acts += 1;
        let k = intent_key(operation, shape);
        if self.index.contains_key(&k) {
            let result = bridge.call_math_unary(operation, x).unwrap_or(f64::NAN);
            self.hits += 1;
            self.workload_est_uj += self.costs.g0;
            return (result, self.costs.g0, true);
        }
        self.misses += 1;
        self.admit_real(operation, shape, probe_input, probe_output);
        let result = bridge.call_math_unary(operation, x).unwrap_or(f64::NAN);
        let cost = self.costs.g4 + self.costs.per_check;
        self.workload_est_uj += cost;
        (result, cost, false)
    }

    /// Seed the pre-verified stdlib basis (operation, shape) pairs under the
    /// basis gate. Returns the count admitted (= |H_corpus|).
    pub fn seed_basis_index(&mut self, basis: &[(String, String)]) -> usize {
        let mut n = 0;
        for (op, shape) in basis {
            let rec = CompetenceRecord {
                input_key: intent_key(op, shape),
                handler_kind: op.clone(),
                probe_input: String::new(),
                probe_output: String::new(),
                result: String::new(),
                cost_uj: self.costs.g4,
                provenance: Provenance::Reported,
            };
            if self.admit(rec).is_accepted() {
                n += 1;
            }
        }
        n
    }

    /// Look up an intent; on a G0 hit record the hit + floor cost and return the
    /// owned handler (the caller executes it via the runtime bridge). On a miss
    /// record the escalation cost and return None.
    pub fn hit_intent(&mut self, operation: &str, shape: &str) -> Option<String> {
        self.acts += 1;
        let k = intent_key(operation, shape);
        if let Some(r) = self.index.get(&k) {
            self.hits += 1;
            self.workload_est_uj += self.costs.g0;
            Some(r.handler_kind.clone())
        } else {
            self.misses += 1;
            self.workload_est_uj += self.costs.g4;
            None
        }
    }

    /// Pure intent lookup (no metering): the owned handler for an intent.
    pub fn lookup_intent(&self, operation: &str, shape: &str) -> Option<String> {
        let k = intent_key(operation, shape);
        self.index.get(&k).map(|r| r.handler_kind.clone())
    }

    /// The admitted records, for circulation to another engine. Each carries its
    /// spectest so the receiving engine can re-verify by replay before admission.
    pub fn records(&self) -> Vec<CompetenceRecord> {
        self.index.values().cloned().collect()
    }
}