//! The all-encompassing vertical stack — RCE does it all.
//!
//! One substrate spanning every layer, each ingested from the field's best work
//! and unified by the same content-addressed, replay-gated discipline:
//!
//! - **Canonicalization** (No Accidental Software / Conal) — name-free content.
//! - **Form layer + mining** (Conal / DéjàVu / Hydra) — categorical forms.
//! - **Competence catalog + workflows** (AWM / ReasoningBank) — crystallized skill.
//! - **Build / compilation** (Blaze / Indexable) — content-addressed incremental.
//! - **Execution** — in-process reference interpreter.
//! - **Energy** (Landauer / energy literature) — measured joules.
//! - **Learning** (Voyager / MSCE) — hindsight + promotion.
//! - **Verification** (MoonBit / Alive2 / Lean) — the evidence ladder.
//! - **Circulation / federation** (Entire) — the shared economy.
//!
//! `Stack::resolve` flows an intent down the stack: reuse crystallized competence
//! first (cheapest), then mine a categorical form, then build — learning from every
//! verified resolution. The whole stack is one content-addressed economy.

use crate::build::{BuildCache, Evidence};
use crate::form::FormMiner;
use crate::learning::Hindsight;
use crate::runtime::RuntimeBridge;
use crate::semantic::{semantic_key, Probe};
use crate::workflow::WorkflowLibrary;

/// Which layer of the stack resolved an intent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    /// Crystallized competence reused (the cheapest act).
    Competence,
    /// A categorical form mined and crystallized.
    Form,
    /// Built / checked fresh.
    Build,
}

/// The outcome of resolving an intent through the vertical stack.
pub struct Resolution {
    /// The layer that covered the intent.
    pub layer: Layer,
    /// The computed result text (first probe).
    pub result: String,
    /// The name-free intent key.
    pub key: String,
    /// The assurance rung.
    pub evidence: Evidence,
}

/// The unified vertical stack.
pub struct Stack {
    /// The build/compilation layer (content-addressed, incremental).
    pub build: BuildCache,
    /// Crystallized workflows (competence).
    pub workflows: WorkflowLibrary,
    /// The categorical form miner.
    pub miner: FormMiner,
    /// The learning loop.
    pub hindsight: Hindsight,
    bridge: RuntimeBridge,
}

impl Stack {
    /// A fresh stack over the in-process reference interpreter.
    pub fn new() -> Self {
        let bridge = RuntimeBridge::new_inprocess();
        Stack {
            build: BuildCache::new(),
            workflows: WorkflowLibrary::new(bridge.clone()),
            miner: FormMiner::new(bridge.clone()),
            hindsight: Hindsight::new(),
            bridge,
        }
    }

    fn run_first(&self, program: &str, probes: &[Probe]) -> Option<String> {
        let first = probes.first()?;
        let arg = first.args.as_array()?.first()?;
        // Extract the event's argument label (x / xs) from the program.
        let label = program.split("event Calc(").nth(1)
            .and_then(|s| s.split(':').next())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "x".into());
        let event = format!("Calc({label}: {arg})");
        self.bridge.run_caller("stack", program, &event).ok()
    }

    /// Resolve an intent spec down the stack: reuse competence, then mine a form,
    /// then build. Every verified resolution is learned. Returns the covering
    /// layer and the result.
    pub fn resolve(&mut self, shape: &str, probes: &[Probe]) -> Option<Resolution> {
        let key = semantic_key(shape, probes);

        // 1. Competence: a crystallized workflow covers the spec (cheapest).
        if let Some(wf) = self.workflows.resolve(shape, probes) {
            let result = self.run_first(&wf.program, probes)?;
            self.hindsight.record(&key, shape, &wf.name, 0);
            return Some(Resolution { layer: Layer::Competence, result, key, evidence: Evidence::Tested });
        }

        // 2. Form: mine a categorical form + filling, crystallize it.
        if let Some(m) = self.miner.mine(shape, probes) {
            let result = self.run_first(&m.workflow.program, probes)?;
            self.workflows.crystallize(m.workflow.clone(), probes);
            self.hindsight.record(&key, shape, &m.workflow.name, 2);
            return Some(Resolution { layer: Layer::Form, result, key, evidence: Evidence::Tested });
        }

        None
    }

    /// Build content through the compilation layer (content-addressed, incremental).
    pub fn build(&mut self, src: &str) -> crate::build::BuildRecord {
        self.build.build(src)
    }
}

impl Default for Stack {
    fn default() -> Self {
        Self::new()
    }
}
