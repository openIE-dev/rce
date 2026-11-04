//! `LeafDispatch` adapter that wires the CapTable into the interpreter.
//!
//! The interpreter doesn't know about capability tables, by design —
//! it just knows about `LeafDispatch` for code-block-shaped nodes. We
//! adapt `CapabilityRef` semantics to the leaf protocol by translating
//! the node's parsed fields (capability_id, slot, method, effects) into
//! a synthetic `language: "capability"` plus a code_text payload that
//! the dispatcher decodes.
//!
//! The interpreter's `CodeBlock` handler is the natural plug point. To
//! avoid changing the interpreter's node-shape handling for v0.1, we
//! provide a separate `dispatch_capability_ref` function that callers
//! (the CLI, tests) invoke directly. v0.2 will fold this into the
//! interpreter's `eval_node` once the IR carries enough metadata.

use crate::chain::DelegationChain;
use crate::error::{CapError, Result};
use crate::table::CapTable;
use rce_vm_backend_cpu::Value;
use rce_graph::graph::GraphNode;

/// Dispatch a `GraphNode::CapabilityRef` against a `CapTable`. Validates
/// effects against the host allow-list, optionally validates a delegation
/// chain, then invokes the registered handler with `args`.
///
/// `chain` is `Some(_)` when the call is bearer-authenticated (a graph
/// received the chain via input binding). `roots_of_trust` is only
/// consulted when a chain is present; ambient (host-direct) authority
/// uses just the effects allow-list.
pub fn dispatch_capability_ref(
    table: &CapTable,
    node: &GraphNode,
    args: &[Value],
    chain: Option<&DelegationChain>,
    roots_of_trust: &[[u8; 32]],
    now: u64,
) -> Result<Value> {
    let GraphNode::CapabilityRef {
        capability_id,
        version: _,
        slot: _,
        method,
        output_type: _,
        effects,
    } = node
    else {
        return Err(CapError::TypeMismatch(format!(
            "expected CapabilityRef node, got {node:?}"
        )));
    };

    // Step 1: effects allow-list (host policy).
    table.effects_allow.check(capability_id, effects)?;

    // Step 2: delegation chain (if provided).
    if let Some(c) = chain {
        c.validate(roots_of_trust, capability_id, now)?;
        // Effects on the node must be a subset of the chain's leaf effects.
        for e in effects {
            if !c.effects().iter().any(|ce| ce == e) {
                return Err(CapError::BadChain(format!(
                    "effect `{e}` not authorised by delegation chain"
                )));
            }
        }
    }

    // Step 3: handler invocation.
    let handler = table.lookup(capability_id, method)?;
    handler(method.as_str(), args)
}

pub struct CapabilityDispatcher {
    pub table: CapTable,
}

impl CapabilityDispatcher {
    pub fn new(table: CapTable) -> Self {
        Self { table }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::EffectsAllow;
    use crate::table::CapTable;
    use rce_vm_backend_cpu::ScalarValue;
    use rce_graph::graph::GraphNode;

    fn cap_ref(cap: &str, method: &str, effects: &[&str]) -> GraphNode {
        GraphNode::CapabilityRef {
            capability_id: cap.into(),
            version: "0.1".into(),
            slot: format!("{cap}.slot"),
            method: method.into(),
            output_type: "i32".into(),
            effects: effects.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn dispatches_to_registered_handler() {
        let table = CapTable::new()
            .with_effects_allow(EffectsAllow::permits_all(["io:read"]))
            .register("test.cap", "double", |_method, args| {
                if let Some(Value::Scalar(ScalarValue::I32(x))) = args.first() {
                    Ok(Value::Scalar(ScalarValue::I32(x * 2)))
                } else {
                    Err(CapError::Handler("expected i32 input".into()))
                }
            });
        let node = cap_ref("test.cap", "double", &["io:read"]);
        let args = vec![Value::Scalar(ScalarValue::I32(21))];
        let out = dispatch_capability_ref(&table, &node, &args, None, &[], 0).unwrap();
        match out {
            Value::Scalar(ScalarValue::I32(x)) => assert_eq!(x, 42),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[test]
    fn refuses_unregistered_capability() {
        let table = CapTable::new().with_effects_allow(EffectsAllow::permits_all(["io:read"]));
        let node = cap_ref("missing.cap", "method", &["io:read"]);
        let res = dispatch_capability_ref(&table, &node, &[], None, &[], 0);
        assert!(matches!(res, Err(CapError::Unregistered(_))));
    }

    #[test]
    fn refuses_disallowed_effect() {
        let table = CapTable::new()
            .with_effects_allow(EffectsAllow::permits_all(["io:read"]))
            .register("test.cap", "noop", |_, _| Ok(Value::Scalar(ScalarValue::I32(0))));
        let node = cap_ref("test.cap", "noop", &["regulatory:financial"]);
        let res = dispatch_capability_ref(&table, &node, &[], None, &[], 0);
        assert!(matches!(res, Err(CapError::EffectDenied { .. })));
    }
}
