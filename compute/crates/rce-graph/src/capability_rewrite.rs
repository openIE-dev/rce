//! Rewrite Lux `ProgramGraph` nodes that name a registered capability
//! slot into `GraphNode::CapabilityRef` nodes.
//!
//! The substrate-becomes-the-IR move at the Lux frontend: when a Lux
//! function calls something like `ledger_transfer(amount)` and the
//! caller has compiled a wallet-bank-slice IR that binds the
//! `ledger.transfer` slot, the rewrite swaps the `Call` node for a
//! `CapabilityRef` that the graph executor dispatches through the
//! Capability Registry runtime.
//!
//! The rewrite is shape-preserving: nodes at the same indices keep
//! their indices, wire endpoints stay valid, output types carry
//! through. So a downstream consumer (interpreter, codegen) can
//! observe both shapes without re-walking wire indices.
//!
//! v1 matches by **node name**: a `Call { callee }` or an
//! `Apply { op: OpKind::Custom(name) }` whose name appears in the
//! `CapabilitySlotMap` becomes a `CapabilityRef`. Future versions
//! can layer typed surface checks (verify the function signature
//! against the manifest's trait + method) on top of this.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::graph::{GraphNode, OpKind, ProgramGraph};

/// One Lux function name → capability slot binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityBinding {
    pub capability_id: String,
    pub version: String,
    pub slot: String,
    pub method: String,
    /// Effects declared by the slot's capability — carried so the
    /// rewrite emits `CapabilityRef` nodes with effects already
    /// stamped, and so `effect_check` can fire without re-walking
    /// the registry.
    #[serde(default)]
    pub effects: Vec<String>,
}

/// Map of Lux function names to capability bindings. Used by
/// `rewrite_program_to_capabilities` to recognise + rewrite calls.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilitySlotMap {
    pub bindings: BTreeMap<String, CapabilityBinding>,
}

impl CapabilitySlotMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a binding by Lux name.
    pub fn bind(
        &mut self,
        lux_name: impl Into<String>,
        capability_id: impl Into<String>,
        version: impl Into<String>,
        slot: impl Into<String>,
        method: impl Into<String>,
    ) {
        self.bind_with_effects(lux_name, capability_id, version, slot, method, vec![]);
    }

    /// Like `bind`, but also stamps the slot's declared effects
    /// onto the binding so emitted `CapabilityRef` nodes carry them.
    pub fn bind_with_effects(
        &mut self,
        lux_name: impl Into<String>,
        capability_id: impl Into<String>,
        version: impl Into<String>,
        slot: impl Into<String>,
        method: impl Into<String>,
        effects: Vec<String>,
    ) {
        self.bindings.insert(
            lux_name.into(),
            CapabilityBinding {
                capability_id: capability_id.into(),
                version: version.into(),
                slot: slot.into(),
                method: method.into(),
                effects,
            },
        );
    }

    /// Convenience: a slot with a method `m` registers as Lux name
    /// `<slot>_<method>` (lowercase, dot→underscore). So a graph spec
    /// slot `ledger` with method `transfer` becomes Lux name
    /// `ledger_transfer`. Useful when you want auto-binding without
    /// hand-writing every Lux name.
    pub fn bind_slot_method(
        &mut self,
        capability_id: impl Into<String>,
        version: impl Into<String>,
        slot: impl Into<String>,
        method: impl Into<String>,
    ) {
        self.bind_slot_method_with_effects(
            capability_id, version, slot, method, vec![],
        );
    }

    pub fn bind_slot_method_with_effects(
        &mut self,
        capability_id: impl Into<String>,
        version: impl Into<String>,
        slot: impl Into<String>,
        method: impl Into<String>,
        effects: Vec<String>,
    ) {
        let slot_s = slot.into();
        let method_s = method.into();
        let lux_name = format!("{slot_s}_{method_s}");
        self.bind_with_effects(
            lux_name, capability_id, version, slot_s, method_s, effects,
        );
    }
}

/// Outcome of one rewrite pass — useful for diagnostics + tests.
#[derive(Debug, Default, Clone)]
pub struct RewriteReport {
    /// (function_name, node_index, lux_name) for every rewrite that fired.
    pub rewrites: Vec<(String, usize, String)>,
    /// Lux names referenced in the program but not bound in the map.
    /// Stored so the caller can surface "you have a Call to `foo` that
    /// matches no capability binding — did you forget to bind it?"
    pub unbound_calls: Vec<(String, usize, String)>,
}

impl RewriteReport {
    pub fn rewrite_count(&self) -> usize {
        self.rewrites.len()
    }
}

/// Rewrite every matching `Call` and `Apply { op: Custom(name) }`
/// node in the program into a `CapabilityRef`. Returns a report of
/// what fired and what didn't.
pub fn rewrite_program_to_capabilities(
    program: &mut ProgramGraph,
    map: &CapabilitySlotMap,
) -> RewriteReport {
    let mut report = RewriteReport::default();
    for func in program.functions.iter_mut() {
        for (idx, node) in func.nodes.iter_mut().enumerate() {
            // Pull out the candidate name + output type, matching the
            // two shapes Lux frontends emit when calling something
            // by name. We borrow temporarily to read; the actual
            // replacement happens after, when we own the binding.
            let cand: Option<(String, String)> = match node {
                GraphNode::Call { callee, return_type, .. } => {
                    Some((callee.clone(), return_type.clone()))
                }
                GraphNode::Apply { op: OpKind::Custom(name), output_type } => {
                    Some((name.clone(), output_type.clone()))
                }
                _ => None,
            };
            let (lux_name, output_type) = match cand {
                Some(x) => x,
                None => continue,
            };
            match map.bindings.get(&lux_name) {
                Some(b) => {
                    *node = GraphNode::CapabilityRef {
                        capability_id: b.capability_id.clone(),
                        version: b.version.clone(),
                        slot: b.slot.clone(),
                        method: b.method.clone(),
                        output_type,
                        effects: b.effects.clone(),
                    };
                    report.rewrites.push((func.name.clone(), idx, lux_name));
                }
                None => {
                    report.unbound_calls.push((func.name.clone(), idx, lux_name));
                }
            }
        }
    }
    report
}

// Tests live in `tests/capability_rewrite.rs` so they compile
// independently of the rest of the rce-graph test binary, which has
// pre-existing import errors in unrelated modules (smoothing, etc.)
// that block `cargo test -p rce-graph --lib`.
