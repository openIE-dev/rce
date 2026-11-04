//! Frontend-time effect gate.
//!
//! After `capability_rewrite` has stamped `CapabilityRef` nodes with
//! their slot's declared effects, this pass walks the rewritten
//! `ProgramGraph` and refuses to certify any function that calls a
//! capability whose effects fall in the *elevated* taxonomy
//! (`regulatory:*`, `crypto:secret-material`, `pii:*`) without
//! declaring it.
//!
//! The substrate's compose-time gates already refuse to compile a
//! `wallet-bank-slice` whose required compliance gates aren't
//! satisfied. This pass pushes that one level higher: into the
//! Lux/JMax frontend, where a function that calls a
//! `regulatory:financial` slot MUST be tagged with an `allows`
//! attribute that names that effect class. If the tag is missing,
//! compile fails with a list of the un-tagged call sites.
//!
//! v1 takes the per-function allow list as a side table
//! (`EffectAllowMap`) supplied by the caller. The Lux frontend can
//! populate it from a function-level `#[allows("regulatory:financial")]`
//! attribute when the parser learns to surface attributes on
//! `FunctionGraph`. Until then, the cap-cli gate command takes the
//! allow list as an explicit `--allows` flag, which is sufficient
//! to demonstrate the gate firing end-to-end.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::{GraphNode, ProgramGraph};

/// Effect prefixes considered *elevated* — calls to slots whose
/// declared effects match any of these require an explicit allow on
/// the enclosing function. Compile fails otherwise.
///
/// The defaults map onto the substrate's regulatory + sensitive-
/// material taxonomy:
/// - `regulatory:financial`, `regulatory:identity` (KYC/AML/etc.)
/// - `crypto:secret-material` (key/seed handling)
/// - `pii:read`, `pii:write`, `pii:transmit`
pub const DEFAULT_ELEVATED_PREFIXES: &[&str] = &[
    "regulatory:",
    "crypto:secret",
    "pii:",
];

/// Per-function explicit allow list.
///
/// `by_fn[fn_name]` is the set of effect strings that function is
/// permitted to use — anything not in the set that matches an
/// `elevated_prefixes` entry produces a violation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EffectAllowMap {
    pub by_fn: BTreeMap<String, Vec<String>>,
}

impl EffectAllowMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn allow(&mut self, fn_name: impl Into<String>, effect: impl Into<String>) {
        self.by_fn
            .entry(fn_name.into())
            .or_default()
            .push(effect.into());
    }

    fn function_allows(&self, fn_name: &str, effect: &str) -> bool {
        match self.by_fn.get(fn_name) {
            Some(list) => list.iter().any(|e| e == effect),
            None => false,
        }
    }
}

/// One violation emitted by `check_effect_gates`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectGateViolation {
    /// Function that contains the offending call.
    pub function: String,
    /// Index into `ProgramGraph.functions[fn].nodes` for diagnostics.
    pub node_index: usize,
    /// Slot the offending call resolves to.
    pub slot: String,
    /// Method on that slot.
    pub method: String,
    /// The specific elevated effect that wasn't allowed.
    pub effect: String,
}

/// Run the effect gate. Returns one violation per (function, call,
/// elevated effect) triple that wasn't covered by `allows`.
///
/// The `allows` argument is the *explicit* CLI / programmatic
/// override. Per-function declarations carried inside the
/// `ProgramGraph.allows` map are merged in automatically — the
/// substrate treats source-comment `@allows:` declarations and CLI
/// `--allows` flags as equivalent. CLI declarations win on the
/// rare case where the same function is allow-listed both ways.
pub fn check_effect_gates(
    program: &ProgramGraph,
    allows: &EffectAllowMap,
    elevated_prefixes: &[&str],
) -> Vec<EffectGateViolation> {
    let merged = merge_program_allows(program, allows);
    let mut out = Vec::new();
    for func in &program.functions {
        for (idx, node) in func.nodes.iter().enumerate() {
            if let GraphNode::CapabilityRef { slot, method, effects, .. } = node {
                for e in effects {
                    if !is_elevated(e, elevated_prefixes) {
                        continue;
                    }
                    if merged.function_allows(&func.name, e) {
                        continue;
                    }
                    out.push(EffectGateViolation {
                        function: func.name.clone(),
                        node_index: idx,
                        slot: slot.clone(),
                        method: method.clone(),
                        effect: e.clone(),
                    });
                }
            }
        }
    }
    out
}

/// Build the effective allow map by overlaying `cli` on top of
/// `program.allows`. Both contribute; CLI wins on duplicate (effect
/// listed in both is harmless).
fn merge_program_allows(program: &ProgramGraph, cli: &EffectAllowMap) -> EffectAllowMap {
    let mut merged = EffectAllowMap::new();
    for (func, effects) in &program.allows {
        for e in effects {
            merged.allow(func.clone(), e.clone());
        }
    }
    for (func, effects) in &cli.by_fn {
        for e in effects {
            merged.allow(func.clone(), e.clone());
        }
    }
    merged
}

fn is_elevated(effect: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|p| effect.starts_with(p))
}

#[cfg(test)]
mod tests {
    // Tests live in tests/effect_check.rs (integration test) so they
    // compile independently of the rest of the rce-graph test
    // binary, which has pre-existing import errors in unrelated
    // modules.
}
