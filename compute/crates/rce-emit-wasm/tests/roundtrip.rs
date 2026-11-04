//! Round-trip integration tests for `rce-emit-wasm`.
//!
//! Build a small rce `FunctionGraph` (matmul + bias-add), emit
//! Wasm, instantiate with wasmtime, and verify the output matches
//! a hand-computed reference.

use rce_emit_wasm::{emit_wasm, EmitOptions};
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Param, TensorLiteralData, Wire, WireKind};
use std::sync::Arc;
use wasmtime::{Engine, Instance, Module, Store};

/// 2x3 input · 3x2 weight + 1x2 bias = 2x2 output.
/// Hand-rolls the graph and weights/bias so we can compute the
/// expected result with a calculator.
#[test]
fn matmul_plus_bias_runs_in_wasm() {
    // ── Build graph ────────────────────────────────────────────────
    let m: usize = 2;
    let k: usize = 3;
    let n: usize = 2;

    // Weight (k, n) = (3, 2). Use simple values.
    let w_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w_bytes = floats_to_bytes(&w_vals);
    // Bias (n,) = (2,). Simple values.
    let b_vals: Vec<f32> = vec![10.0, 20.0];
    let b_bytes = floats_to_bytes(&b_vals);

    let m_wire = WireKind::Move;
    let nodes = vec![
        // 0: input X (m, k)
        GraphNode::Param(Param {
            name: "x".into(),
            data_type: "f32".into(),
        }),
        // 1: weight W (k, n) inlined
        GraphNode::TensorLiteral(TensorLiteralData {
            data: Arc::from(w_bytes.into_boxed_slice()),
            shape: vec![k, n],
            dtype: "f32".into(),
            name: "w".into(),
        }),
        // 2: bias B (n,) inlined
        GraphNode::TensorLiteral(TensorLiteralData {
            data: Arc::from(b_bytes.into_boxed_slice()),
            shape: vec![n],
            dtype: "f32".into(),
            name: "b".into(),
        }),
        // 3: matmul(X, W)
        GraphNode::Apply {
            op: OpKind::Custom("matmul".into()),
            output_type: "f32".into(),
        },
        // 4: add(matmul, B) — bias add
        GraphNode::Apply {
            op: OpKind::Add,
            output_type: "f32".into(),
        },
    ];
    let wires = vec![
        Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: m_wire },
        Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: m_wire },
        Wire { from: 3, to: 4, port_in_idx: 0, wire_kind: m_wire },
        Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: m_wire },
    ];
    let graph = FunctionGraph {
        name: "linear".into(),
        nodes,
        wires,
        output_node: 4,
        return_type: "f32".into(),
    };

    // ── Emit Wasm ──────────────────────────────────────────────────
    let opts = EmitOptions::new(vec![vec![m, k]]);
    let wasm = emit_wasm(&graph, &opts).expect("emit_wasm");

    // ── Instantiate + run ──────────────────────────────────────────
    let engine = Engine::default();
    let module = Module::new(&engine, &wasm).expect("Module::new");
    let mut store: Store<()> = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("Instance::new");
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("memory export");

    // Lay caller-side buffers AFTER the module's scratch region.
    // Scratch holds: weight (3*2*4=24) + bias (2*4=8) + matmul_out (2*2*4=16) + bias_out (16) = 64 bytes.
    // Round-up to 16-byte alignment per element happens in the emitter,
    // so total scratch is something like 16 + 16 + 16 + 16 = 64 bytes.
    // Place input + output well past that.
    let x_off: usize = 4096;
    let out_off: usize = 8192;

    let x_vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // (2, 3)
    let x_bytes = floats_to_bytes(&x_vals);
    memory.write(&mut store, x_off, &x_bytes).expect("write x");

    let run = instance
        .get_typed_func::<(i32, i32), ()>(&mut store, "run")
        .expect("get_typed_func run");
    run.call(&mut store, (x_off as i32, out_off as i32))
        .expect("run.call");

    // ── Read output + compute reference ─────────────────────────────
    let mut out_bytes = vec![0u8; m * n * 4];
    memory.read(&store, out_off, &mut out_bytes).expect("read out");
    let got = bytes_to_floats(&out_bytes);

    // Reference: out[i, j] = sum_k x[i, k] * w[k, j] + b[j]
    let mut want = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0_f32;
            for kk in 0..k {
                s += x_vals[i * k + kk] * w_vals[kk * n + j];
            }
            want[i * n + j] = s + b_vals[j];
        }
    }

    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-4,
            "mismatch at {i}: got {g}, want {w} (full got={:?}, want={:?})",
            got, want
        );
    }
}

/// Verify the emitter rejects a graph it can't lower.
#[test]
fn unsupported_op_returns_error() {
    let nodes = vec![
        GraphNode::Param(Param {
            name: "x".into(),
            data_type: "f32".into(),
        }),
        GraphNode::Param(Param {
            name: "y".into(),
            data_type: "f32".into(),
        }),
        // Sub is not in v0 scope.
        GraphNode::Apply {
            op: OpKind::Sub,
            output_type: "f32".into(),
        },
    ];
    let wires = vec![
        Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
        Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
    ];
    let graph = FunctionGraph {
        name: "sub".into(),
        nodes,
        wires,
        output_node: 2,
        return_type: "f32".into(),
    };
    let opts = EmitOptions::new(vec![vec![2, 2], vec![2, 2]]);
    let res = emit_wasm(&graph, &opts);
    assert!(res.is_err(), "expected emit error for unsupported op");
}

fn floats_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_floats(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
