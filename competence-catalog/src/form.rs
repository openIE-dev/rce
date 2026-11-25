//! The categorical form layer + mining — Conal Elliott's compiling-to-categories
//! and the DéjàVu/Hydra corpus premise ingested into RCE.
//!
//! Most programs are instantiations of a few categorical forms: `map` (functor),
//! `filter` (subobject), `fold` (catamorphism). These forms are already present in
//! the basis (`col.func`); the form layer recognizes that an intent is an
//! instantiation of a form and *mines* the filling (the defunctionalized Op tag)
//! that reproduces the intent's behavior. The mined form+filling crystallizes as
//! a workflow. This is the "99% of programs map to a form" layer: the form is the
//! reusable structure, the Op tag is the panache slot, and the gate verifies the
//! instantiation by replay.

use serde_json::Value;

use crate::runtime::RuntimeBridge;
use crate::semantic::Probe;
use crate::workflow::{Workflow, WorkflowLibrary};

/// A categorical form: a reusable combinator pattern from the basis.
pub struct Form {
    /// The categorical name (map / filter / fold).
    pub name: &'static str,
    /// The basis combinator.
    pub combinator: &'static str,
    /// The module providing it.
    pub module: &'static str,
    /// The intent shape it covers.
    pub shape: &'static str,
}

/// The canonical categorical vocabulary mined from the basis (Conal's vocabulary
/// made concrete: functor / subobject / catamorphism over lists).
pub fn canonical_forms() -> Vec<Form> {
    vec![
        Form { name: "map", combinator: "map", module: "col.func", shape: "(&[i64],&Op)->List[i64]" },
        Form { name: "filter", combinator: "filter", module: "col.func", shape: "(&[i64],&Op)->List[i64]" },
    ]
}

/// The defunctionalized Op fillings (the panache slot of a form).
pub const OP_TAGS: &[&str] = &["Add", "Mul", "Sub", "Max", "Min", "EqZero", "GtZero",
    "LtZero", "Even", "Odd", "Pos", "Neg", "Abs", "Not"];

/// A mined instantiation: a categorical form plus the filling that covers the intent.
pub struct FormMatch {
    /// The categorical form name.
    pub form: String,
    /// The Op filling.
    pub op: String,
    /// The instantiated workflow.
    pub workflow: Workflow,
}

/// Mines the categorical form + filling that reproduces an intent spec.
pub struct FormMiner {
    bridge: RuntimeBridge,
}

impl FormMiner {
    /// A miner over a runtime bridge.
    pub fn new(bridge: RuntimeBridge) -> Self {
        FormMiner { bridge }
    }

    fn program(form: &Form, op: &str) -> String {
        let prefix = form.module.rsplit('.').next().unwrap_or(form.module);
        format!(
            "module form_{}\n\nuse {}\n\nstate Out\n    v: List[i64] = []\n\n\
             event Calc(xs: List[i64])\n\nstep Out on Calc(xs: a)\n    var o: Op = {op}\n    v = {prefix}.{}(&a, &o)\n",
            form.name, form.module, form.combinator
        )
    }

    fn covers(&self, form: &Form, op: &str, probes: &[Probe]) -> bool {
        let program = Self::program(form, op);
        probes.iter().all(|p| {
            let first = match p.args.as_array().and_then(|a| a.first()) {
                Some(f) => f,
                None => return false,
            };
            let event = format!("Calc(xs: {first})");
            self.bridge.run_caller("form", &program, &event)
                .map(|got| got == p.expected).unwrap_or(false)
        })
    }

    /// Mine the form + filling covering the spec, or `None`. Searches the
    /// categorical vocabulary × Op fillings; every candidate is replay-verified.
    pub fn mine(&self, shape: &str, probes: &[Probe]) -> Option<FormMatch> {
        for form in canonical_forms() {
            if form.shape != shape {
                continue;
            }
            for op in OP_TAGS {
                if self.covers(&form, op, probes) {
                    let program = Self::program(&form, op);
                    let workflow = Workflow {
                        name: format!("{}-{}", form.name, op),
                        shape: shape.to_string(),
                        ops: vec![format!("{}.{}", form.module, form.combinator), op.to_string()],
                        program,
                    };
                    return Some(FormMatch { form: form.name.to_string(), op: op.to_string(), workflow });
                }
            }
        }
        None
    }

    /// Mine and crystallize into a workflow library in one step (the elastic
    /// basis growing by mined, verified competence).
    pub fn mine_and_crystallize(&self, lib: &mut WorkflowLibrary, shape: &str, probes: &[Probe]) -> Option<FormMatch> {
        let m = self.mine(shape, probes)?;
        if lib.crystallize(m.workflow.clone(), probes) {
            Some(m)
        } else {
            None
        }
    }
}

/// Convenience: a probe from JSON args and an expected output.
pub fn probe(args: Value, expected: impl Into<String>) -> Probe {
    Probe::new(args, expected)
}
