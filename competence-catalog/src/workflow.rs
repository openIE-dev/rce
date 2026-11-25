//! Crystallized workflows — Agent Workflow Memory / ReasoningBank ingested into
//! RCE's gated framework.
//!
//! AWM induces reusable multi-step routines from successful trajectories; RCE
//! adds the gate. A workflow is a named, multi-step Form procedure (a pipeline of
//! basis operations) crystallized as a single competence unit. It is admitted
//! only by replaying its spectest, indexed by shape, and reused at G0 when a
//! novel intent spec matches its behavior. The library persists and grows with
//! use — competence at the granularity of *procedures*, not just operations.

use std::collections::HashMap;

use serde_json::Value;

use crate::runtime::RuntimeBridge;
use crate::semantic::Probe;

/// A crystallized multi-step competence: a named Form procedure.
#[derive(Clone)]
pub struct Workflow {
    /// A human-readable name.
    pub name: String,
    /// The type signature (input shape -> output).
    pub shape: String,
    /// The basis operations composed, in pipeline order.
    pub ops: Vec<String>,
    /// The Form program that computes the workflow.
    pub program: String,
}

/// A replay-gated library of crystallized workflows.
pub struct WorkflowLibrary {
    bridge: RuntimeBridge,
    by_shape: HashMap<String, Vec<Workflow>>,
    count: usize,
}

impl WorkflowLibrary {
    /// An empty library over a runtime bridge.
    pub fn new(bridge: RuntimeBridge) -> Self {
        WorkflowLibrary { bridge, by_shape: HashMap::new(), count: 0 }
    }

    /// Crystallize a workflow: admit it only if its program reproduces every
    /// probe under replay (the gate). Returns whether it was admitted.
    pub fn crystallize(&mut self, wf: Workflow, probes: &[Probe]) -> bool {
        let covers = probes.iter().all(|p| {
            let first = match p.args.as_array().and_then(|a| a.first()) {
                Some(f) => f,
                None => return false,
            };
            let event = format!("Calc(x: {first})");
            self.bridge.run_caller("wf", &wf.program, &event)
                .map(|got| got == p.expected).unwrap_or(false)
        });
        if !covers {
            return false;
        }
        self.by_shape.entry(wf.shape.clone()).or_default().push(wf);
        self.count += 1;
        true
    }

    /// Resolve an intent spec to a crystallized workflow by replay-verified
    /// behavior (name-free). The cheapest competence that covers the spec.
    pub fn resolve(&self, shape: &str, probes: &[Probe]) -> Option<&Workflow> {
        let candidates = self.by_shape.get(shape)?;
        candidates.iter().find(|wf| {
            probes.iter().all(|p| {
                let first = match p.args.as_array().and_then(|a| a.first()) {
                    Some(f) => f,
                    None => return false,
                };
                let event = format!("Calc(x: {first})");
                self.bridge.run_caller("wf", &wf.program, &event)
                    .map(|got| got == p.expected).unwrap_or(false)
            })
        })
    }

    /// Number of crystallized workflows.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Whether the library is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

/// Build a pipeline workflow program: `op_n(...op_2(op_1(x)))` over unary basis
/// operations, all sharing the same input/output type `t`. The result is a Form
/// program ready to crystallize.
pub fn pipeline_program(name: &str, t: &str, ops: &[(String, String, String)]) -> String {
    // ops: (module, name, _) — call prefix is the last module segment.
    let mut expr = "a".to_string();
    for (m, n, _) in ops {
        let prefix = m.rsplit('.').next().unwrap_or(m);
        expr = format!("{prefix}.{n}({expr})");
    }
    let uses: Vec<String> = ops.iter().map(|(m, _, _)| format!("use {m}")).collect();
    let init = if t == "f64" || t == "f32" { "0.0" } else if t == "bool" { "false" } else { "0" };
    format!(
        "module wf_{name}\n\n{}\n\nstate Out\n    v: {t} = {init}\n\n\
         event Calc(x: {t})\n\nstep Out on Calc(x: a)\n    v = {expr}\n",
        uses.join("\n")
    )
}

/// Convenience: a probe from JSON args and an expected output.
pub fn probe(args: Value, expected: impl Into<String>) -> Probe {
    Probe::new(args, expected)
}
