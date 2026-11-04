//! Integration tests for `capability_rewrite`. Lives in `tests/` so
//! it compiles independently of unit-test errors in unrelated modules.

use rce_graph::capability_rewrite::{
    rewrite_program_to_capabilities, CapabilitySlotMap,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, ProgramGraph,
};

fn program_with_calls() -> ProgramGraph {
    ProgramGraph {
        entry: "main".into(),
        allows: std::collections::BTreeMap::new(),
        functions: vec![FunctionGraph {
            name: "main".into(),
            nodes: vec![
                GraphNode::Literal {
                    value: LiteralValue::I64(100),
                    data_type: "i64".into(),
                },
                GraphNode::Call {
                    callee: "ledger_transfer".into(),
                    arg_types: vec!["i64".into()],
                    return_type: "Receipt".into(),
                },
                GraphNode::Apply {
                    op: OpKind::Custom("wal_fsync".into()),
                    output_type: "()".into(),
                },
                GraphNode::Call {
                    callee: "println".into(),
                    arg_types: vec!["String".into()],
                    return_type: "()".into(),
                },
            ],
            wires: vec![],
            output_node: 1,
            return_type: "Receipt".into(),
        }],
    }
}

fn wallet_map() -> CapabilitySlotMap {
    let mut m = CapabilitySlotMap::new();
    m.bind_slot_method(
        "joulesperbit.pay.ledger",
        "1.0.0",
        "ledger",
        "transfer",
    );
    m.bind_slot_method("joulesperbit.db.wal", "1.0.0", "wal", "fsync");
    m
}

#[test]
fn rewrites_call_to_capability_ref() {
    let mut p = program_with_calls();
    let report = rewrite_program_to_capabilities(&mut p, &wallet_map());
    assert_eq!(report.rewrite_count(), 2);
    match &p.functions[0].nodes[1] {
        GraphNode::CapabilityRef { capability_id, slot, method, output_type, .. } => {
            assert_eq!(capability_id, "joulesperbit.pay.ledger");
            assert_eq!(slot, "ledger");
            assert_eq!(method, "transfer");
            assert_eq!(output_type, "Receipt");
        }
        other => panic!("expected CapabilityRef, got {other:?}"),
    }
}

#[test]
fn rewrites_apply_custom_to_capability_ref() {
    let mut p = program_with_calls();
    rewrite_program_to_capabilities(&mut p, &wallet_map());
    match &p.functions[0].nodes[2] {
        GraphNode::CapabilityRef { capability_id, slot, method, .. } => {
            assert_eq!(capability_id, "joulesperbit.db.wal");
            assert_eq!(slot, "wal");
            assert_eq!(method, "fsync");
        }
        other => panic!("expected CapabilityRef, got {other:?}"),
    }
}

#[test]
fn unbound_calls_are_reported_not_rewritten() {
    let mut p = program_with_calls();
    let report = rewrite_program_to_capabilities(&mut p, &wallet_map());
    assert!(report.unbound_calls.iter().any(|(_, _, n)| n == "println"));
    assert!(matches!(
        p.functions[0].nodes[3],
        GraphNode::Call { ref callee, .. } if callee == "println"
    ));
}

#[test]
fn rewrite_preserves_node_indices_and_count() {
    let mut p = program_with_calls();
    let before_count = p.functions[0].nodes.len();
    rewrite_program_to_capabilities(&mut p, &wallet_map());
    assert_eq!(p.functions[0].nodes.len(), before_count);
    assert!(matches!(p.functions[0].nodes[0], GraphNode::Literal { .. }));
}

#[test]
fn empty_map_is_a_no_op_with_diagnostics() {
    let mut p = program_with_calls();
    let map = CapabilitySlotMap::new();
    let report = rewrite_program_to_capabilities(&mut p, &map);
    assert_eq!(report.rewrite_count(), 0);
    // Three name-bearing nodes: 2 Calls + 1 Apply(Custom).
    assert_eq!(report.unbound_calls.len(), 3);
}

/// Build a CapabilitySlotMap from a `RceIr` JSON shape (the kind
/// `cap compile --target rce` emits) — proves the registry IR can
/// drive the Lux rewrite directly without manual binding.
#[test]
fn slot_map_can_be_built_from_rce_ir_json() {
    let ir_json = serde_json::json!({
        "intent": "wallet-test",
        "nodes": [
            {
                "id": "ledger.transfer",
                "name": "ledger.transfer",
                "op_kind": {
                    "capability_ref": {
                        "id": "joulesperbit.pay.ledger",
                        "version": "1.0.0",
                        "slot": "ledger",
                        "method": "transfer"
                    }
                },
                "energy_cost_pj": 81720525u64,
                "pure": false
            },
            {
                "id": "wal.fsync",
                "name": "wal.fsync",
                "op_kind": {
                    "capability_ref": {
                        "id": "joulesperbit.db.wal",
                        "version": "1.0.0",
                        "slot": "wal",
                        "method": "fsync"
                    }
                },
                "energy_cost_pj": 2253906250u64,
                "pure": false
            }
        ],
        "edges": [],
        "budget_pj": 0u64,
        "limit_pj": 0u64
    });

    // Decode the IR and build a slot map by walking each node.
    let mut map = CapabilitySlotMap::new();
    for node in ir_json["nodes"].as_array().unwrap() {
        let cap = &node["op_kind"]["capability_ref"];
        map.bind_slot_method(
            cap["id"].as_str().unwrap(),
            cap["version"].as_str().unwrap(),
            cap["slot"].as_str().unwrap(),
            cap["method"].as_str().unwrap(),
        );
    }

    let mut p = program_with_calls();
    let report = rewrite_program_to_capabilities(&mut p, &map);
    assert_eq!(report.rewrite_count(), 2);
}
