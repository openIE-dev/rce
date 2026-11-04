//! Integration tests for the frontend-time effect gate.

use rce_graph::capability_rewrite::{
    rewrite_program_to_capabilities, CapabilitySlotMap,
};
use rce_graph::effect_check::{
    check_effect_gates, EffectAllowMap, DEFAULT_ELEVATED_PREFIXES,
};
use rce_graph::graph::{
    FunctionGraph, GraphNode, LiteralValue, OpKind, ProgramGraph,
};

fn program_with_two_functions() -> ProgramGraph {
    ProgramGraph {
        entry: "main".into(),
        allows: std::collections::BTreeMap::new(),
        functions: vec![
            FunctionGraph {
                name: "pay".into(),
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
                ],
                wires: vec![],
                output_node: 1,
                return_type: "Receipt".into(),
            },
            FunctionGraph {
                name: "main".into(),
                nodes: vec![GraphNode::Call {
                    callee: "pay".into(),
                    arg_types: vec!["i64".into()],
                    return_type: "Receipt".into(),
                }],
                wires: vec![],
                output_node: 0,
                return_type: "Receipt".into(),
            },
        ],
    }
}

fn wallet_map_with_effects() -> CapabilitySlotMap {
    let mut m = CapabilitySlotMap::new();
    m.bind_slot_method_with_effects(
        "joulesperbit.pay.ledger",
        "1.0.0",
        "ledger",
        "transfer",
        // Elevated effects: regulatory + pii.
        vec![
            "regulatory:financial".into(),
            "pii:read".into(),
            "mem:owned".into(),
        ],
    );
    m.bind_slot_method_with_effects(
        "joulesperbit.db.wal",
        "1.0.0",
        "wal",
        "fsync",
        // Non-elevated only.
        vec!["fs:append".into(), "fs:write".into()],
    );
    m
}

#[test]
fn flags_unallowed_elevated_effect() {
    let mut p = program_with_two_functions();
    let map = wallet_map_with_effects();
    rewrite_program_to_capabilities(&mut p, &map);

    let allows = EffectAllowMap::new(); // empty — pay declared nothing
    let violations = check_effect_gates(&p, &allows, DEFAULT_ELEVATED_PREFIXES);
    // ledger.transfer brings two elevated effects (regulatory:financial,
    // pii:read), wal.fsync brings none. Two violations expected.
    assert_eq!(violations.len(), 2, "got: {:?}", violations);
    for v in &violations {
        assert_eq!(v.function, "pay");
        assert_eq!(v.slot, "ledger");
        assert_eq!(v.method, "transfer");
    }
    let effects: Vec<_> = violations.iter().map(|v| v.effect.as_str()).collect();
    assert!(effects.contains(&"regulatory:financial"));
    assert!(effects.contains(&"pii:read"));
}

#[test]
fn allow_list_silences_specific_effects() {
    let mut p = program_with_two_functions();
    let map = wallet_map_with_effects();
    rewrite_program_to_capabilities(&mut p, &map);

    // pay() declares it allows both elevated effects.
    let mut allows = EffectAllowMap::new();
    allows.allow("pay", "regulatory:financial");
    allows.allow("pay", "pii:read");

    let violations = check_effect_gates(&p, &allows, DEFAULT_ELEVATED_PREFIXES);
    assert!(violations.is_empty(), "got: {:?}", violations);
}

#[test]
fn allow_in_wrong_function_does_not_silence() {
    let mut p = program_with_two_functions();
    let map = wallet_map_with_effects();
    rewrite_program_to_capabilities(&mut p, &map);

    // main declares the allow, but pay is the one calling the slot.
    let mut allows = EffectAllowMap::new();
    allows.allow("main", "regulatory:financial");
    allows.allow("main", "pii:read");

    let violations = check_effect_gates(&p, &allows, DEFAULT_ELEVATED_PREFIXES);
    assert_eq!(violations.len(), 2);
    assert!(violations.iter().all(|v| v.function == "pay"));
}

#[test]
fn non_elevated_effects_are_never_flagged() {
    let mut p = program_with_two_functions();
    let map = wallet_map_with_effects();
    rewrite_program_to_capabilities(&mut p, &map);

    // Allow only regulatory + pii. wal.fsync's mem/fs effects are
    // non-elevated so the gate doesn't care about them either way.
    let mut allows = EffectAllowMap::new();
    allows.allow("pay", "regulatory:financial");
    allows.allow("pay", "pii:read");

    let violations = check_effect_gates(&p, &allows, DEFAULT_ELEVATED_PREFIXES);
    assert!(violations.is_empty(), "got: {:?}", violations);
}

#[test]
fn elevated_prefix_list_is_overridable() {
    let mut p = program_with_two_functions();
    let map = wallet_map_with_effects();
    rewrite_program_to_capabilities(&mut p, &map);

    // Custom: only regulatory: counts as elevated. pii is downgraded.
    let custom: &[&str] = &["regulatory:"];
    let allows = EffectAllowMap::new();
    let violations = check_effect_gates(&p, &allows, custom);
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].effect, "regulatory:financial");
}

#[test]
fn unbound_calls_dont_crash_the_check() {
    // Same program, but with a call that has no slot binding.
    let mut p = program_with_two_functions();
    p.functions[1].nodes.push(GraphNode::Call {
        callee: "log_info".into(),
        arg_types: vec![],
        return_type: "()".into(),
    });
    let map = wallet_map_with_effects();
    rewrite_program_to_capabilities(&mut p, &map);

    let mut allows = EffectAllowMap::new();
    allows.allow("pay", "regulatory:financial");
    allows.allow("pay", "pii:read");
    let violations = check_effect_gates(&p, &allows, DEFAULT_ELEVATED_PREFIXES);
    assert!(violations.is_empty());
}
