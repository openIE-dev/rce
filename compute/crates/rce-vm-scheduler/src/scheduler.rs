//! Async scheduler: spawns one tokio task per graph node, connects them
//! with per-wire `oneshot::channel`s, and overlaps independent branches.
//!
//! The scheduler **does not duplicate** the per-node evaluation logic. It
//! reuses `rce_vm_backend_cpu::interpret` for single-node evaluation by
//! constructing a one-node sub-graph with literal inputs. That keeps
//! kernel coverage (CPU + wgpu via OpDispatch + wasm via LeafDispatch)
//! identical between the synchronous interpreter and the scheduler — the
//! only thing that changes is *when* nodes run.

use crate::error::{Result, ScheduleError};
use crate::placement::{pick_priced, PricedBackend};
use rce_vm_backend_cpu::{
    interpret_with_config, HostTensor, Inputs, InterpretConfig, LeafDispatch, OpDispatch, Value,
};
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Param, Wire, WireKind};
use std::sync::Arc;
use tokio::sync::oneshot;
use tokio::task::JoinSet;

/// Bundle of dispatchers passed across `await` points. Concrete sync
/// dispatchers live behind `Arc<dyn _ + Send + Sync>` so each per-node task
/// can hold its own clone without forcing static lifetimes.
#[derive(Clone, Default)]
pub struct Backends {
    pub leaves: Vec<Arc<dyn LeafDispatch + Send + Sync>>,
    pub ops: Vec<Arc<dyn OpDispatch + Send + Sync>>,
    /// Priced backends consulted via [`pick_priced`] before the legacy chain.
    /// When this is empty (the default), the scheduler runs exactly as it
    /// did before — full backward compatibility. When non-empty, every
    /// `Apply { op: Custom(name), .. }` whose `name` is claimed by at least
    /// one priced backend is routed to the lowest-joule one, REPLACING the
    /// first-claim-wins `OpDispatch` chain for the ops it covers. Ops not
    /// claimed by any priced backend fall through unchanged, so placement
    /// is incrementally adoptable op-by-op.
    pub placement: Vec<Arc<dyn PricedBackend>>,
}

/// Synchronous wrapper that builds a tokio runtime and runs the graph.
/// Use this from non-async callers (CLI, tests). Panics if it can't
/// create a runtime.
pub fn run(graph: &FunctionGraph, inputs: Inputs, backends: Backends) -> Result<Value> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| ScheduleError::Join(format!("runtime build: {e}")))?;
    runtime.block_on(run_async(graph.clone(), inputs, backends))
}

/// Async entry point. Spawns per-node tasks and awaits the output.
pub async fn run_async(
    graph: FunctionGraph,
    inputs: Inputs,
    backends: Backends,
) -> Result<Value> {
    let n = graph.nodes.len();
    if graph.output_node >= n {
        return Err(ScheduleError::OutputNodeOutOfRange(graph.output_node, n));
    }

    // Build per-node senders/receivers. For each wire, create one
    // oneshot channel; the producer holds the sender, the consumer holds
    // the receiver tagged with the destination port.
    let mut senders_by_node: Vec<Vec<oneshot::Sender<Value>>> = (0..n).map(|_| Vec::new()).collect();
    let mut receivers_by_node: Vec<Vec<(i32, oneshot::Receiver<Value>)>> =
        (0..n).map(|_| Vec::new()).collect();

    for w in &graph.wires {
        if w.from >= n || w.to >= n {
            return Err(ScheduleError::InvalidWire(format!(
                "wire {}->{} out of bounds (n={n})",
                w.from, w.to
            )));
        }
        let (tx, rx) = oneshot::channel::<Value>();
        senders_by_node[w.from].push(tx);
        receivers_by_node[w.to].push((w.port_in_idx, rx));
    }

    // Output capture: a separate channel from output_node to the runner.
    let (out_tx, out_rx) = oneshot::channel::<Value>();
    let mut out_tx_for_node: Vec<Option<oneshot::Sender<Value>>> = (0..n).map(|_| None).collect();
    out_tx_for_node[graph.output_node] = Some(out_tx);

    // Cycle detection — the topo sort would catch this anyway, but doing
    // it up front means we never spawn tasks on a cyclic graph.
    if has_cycle(&graph) {
        return Err(ScheduleError::Cyclic);
    }

    let graph = Arc::new(graph);
    let inputs = Arc::new(inputs);
    let backends = Arc::new(backends);

    let mut tasks: JoinSet<Result<()>> = JoinSet::new();

    for node_idx in 0..n {
        let receivers = std::mem::take(&mut receivers_by_node[node_idx]);
        let senders = std::mem::take(&mut senders_by_node[node_idx]);
        let out_tx = out_tx_for_node[node_idx].take();

        let graph = Arc::clone(&graph);
        let inputs = Arc::clone(&inputs);
        let backends = Arc::clone(&backends);

        tasks.spawn(async move {
            // 1. Await inputs.
            let mut by_port: Vec<(i32, Value)> = Vec::with_capacity(receivers.len());
            for (port, rx) in receivers {
                let v = rx
                    .await
                    .map_err(|_| ScheduleError::InputChannelClosed(node_idx))?;
                by_port.push((port, v));
            }
            by_port.sort_by_key(|(p, _)| *p);
            let args: Vec<Value> = by_port.into_iter().map(|(_, v)| v).collect();

            // 2. Evaluate.
            let result = eval_node_async(&graph, node_idx, args, inputs, backends).await?;

            // 3. Fan out to consumers.
            for tx in senders {
                let _ = tx.send(result.clone());
            }
            // 4. Output capture (if this is the graph's output node).
            if let Some(tx) = out_tx {
                let _ = tx.send(result);
            }
            Ok(())
        });
    }

    // Drain the JoinSet; surface the first error (if any).
    let mut first_err: Option<ScheduleError> = None;
    while let Some(res) = tasks.join_next().await {
        match res {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                first_err.get_or_insert(e);
            }
            Err(e) => {
                first_err.get_or_insert(ScheduleError::Join(format!("{e}")));
            }
        }
    }
    if let Some(e) = first_err {
        return Err(e);
    }

    out_rx
        .await
        .map_err(|_| ScheduleError::InputChannelClosed(graph.output_node))
}

/// Single-node evaluation: build a one-node graph (with the same node, plus
/// fake `Param` placeholders for each gathered argument) and call the
/// existing `interpret_with_config`. This guarantees kernel coverage and
/// dispatch behaviour are identical to the synchronous interpreter.
async fn eval_node_async(
    graph: &FunctionGraph,
    node_idx: usize,
    args: Vec<Value>,
    inputs: Arc<Inputs>,
    backends: Arc<Backends>,
) -> Result<Value> {
    let node = graph.nodes[node_idx].clone();
    // For Param nodes, no inputs were collected — read from the bound inputs.
    if let GraphNode::Param(_) = &node {
        // Synchronous; no need to spawn_blocking.
        let mini = build_singleton(&node, &[]);
        let v = interpret_with_config(&mini, inputs.as_ref(), config_view(backends.as_ref()))
            .map_err(ScheduleError::Interpret)?;
        return Ok(v);
    }

    // Heavier nodes (Apply, CodeBlock, TensorLiteral) go through
    // spawn_blocking so the scheduler thread isn't held by a long matmul.
    let backends_for_blocking = Arc::clone(&backends);
    let inputs_for_blocking = Arc::clone(&inputs);
    let res = tokio::task::spawn_blocking(move || -> Result<Value> {
        // Placement-driven dispatch (Stage 4 + 4b): when any priced
        // backend in the registry claims `op_name`, pick the lowest-joule
        // one and route there directly — REPLACING the first-claim-wins
        // OpDispatch chain for that op. Empty `placement` (the default)
        // falls through unchanged. Two extension points:
        //
        // * Custom ops use their explicit name (qmm / matmul / silu / …).
        // * Built-in OpKind variants (Add / Mul / … and the rest of the
        //   arithmetic / comparison / logic / Concat surface) use a
        //   canonical lowercase name ("add" / "mul" / …), so a future
        //   priced backend that lists "add" in `handles()` routes
        //   element-wise tensor-add through Metal (etc.) without any
        //   scheduler change. Backends that don't claim the op fall
        //   through to the legacy interpreter exactly as before.
        //
        // Inputs and outputs flow as `Value`, so device-resident tensors
        // (`Value::DeviceTensor`) pass straight through to the next
        // dispatch — DeviceTensor preservation. A chain of consecutive
        // same-device ops never demotes to host between dispatches.
        if let GraphNode::Apply { op, .. } = &node {
            if !backends_for_blocking.placement.is_empty() {
                if let Some(op_name) = crate::placement::opkind_name(op) {
                    let mut shapes: Vec<&[usize]> = Vec::with_capacity(args.len());
                    let mut has_tensor = false;
                    for v in &args {
                        if let Some(s) = crate::placement::value_shape(v) {
                            shapes.push(s);
                            has_tensor = true;
                        }
                    }
                    if has_tensor {
                        if let Some((be, _pj, _measured)) =
                            pick_priced(&backends_for_blocking.placement, op_name, &shapes)
                        {
                            let out = be.execute(op_name, &args).map_err(|e| {
                                ScheduleError::Join(format!("placement {}: {}", be.id(), e))
                            })?;
                            return Ok(out);
                        }
                    }
                }
            }
        }
        // Fall through: legacy first-claim-wins via the OpDispatch chain.
        let mini = build_singleton(&node, &args);
        let mut combined: Inputs = inputs_for_blocking.as_ref().clone();
        for (i, v) in args.into_iter().enumerate() {
            combined.insert(format!("__arg{i}"), v);
        }
        interpret_with_config(&mini, &combined, config_view(backends_for_blocking.as_ref()))
            .map_err(ScheduleError::Interpret)
    })
    .await
    .map_err(|e| ScheduleError::Join(format!("{e}")))??;
    Ok(res)
}

/// Build a one-node `FunctionGraph` whose only Apply/CodeBlock/Literal node
/// receives its arguments through synthetic `__arg{i}` Param placeholders.
fn build_singleton(node: &GraphNode, args: &[Value]) -> FunctionGraph {
    let mut nodes: Vec<GraphNode> = Vec::with_capacity(args.len() + 1);
    let mut wires: Vec<Wire> = Vec::with_capacity(args.len());
    for i in 0..args.len() {
        nodes.push(GraphNode::Param(Param {
            name: format!("__arg{i}"),
            data_type: data_type_for(&args[i]),
        }));
    }
    let target_idx = nodes.len();
    nodes.push(node.clone());
    for (i, _) in args.iter().enumerate() {
        wires.push(Wire {
            from: i,
            to: target_idx,
            port_in_idx: i as i32,
            wire_kind: WireKind::Move,
        });
    }
    FunctionGraph {
        name: "__singleton__".into(),
        nodes,
        wires,
        output_node: target_idx,
        return_type: "f32".into(),
    }
}

fn data_type_for(v: &Value) -> String {
    use rce_vm_backend_cpu::ScalarValue;
    match v {
        Value::Scalar(ScalarValue::F32(_)) => "f32".into(),
        Value::Scalar(ScalarValue::F64(_)) => "f64".into(),
        Value::Scalar(ScalarValue::I32(_)) => "i32".into(),
        Value::Scalar(ScalarValue::I64(_)) => "i64".into(),
        Value::Scalar(ScalarValue::Bool(_)) => "bool".into(),
        Value::Scalar(ScalarValue::String(_)) => "string".into(),
        Value::Tensor(t) => t.dtype.name().into(),
        Value::DeviceTensor(d) => d.dtype.name().into(),
    }
}

fn config_view<'a>(backends: &'a Backends) -> InterpretConfig<'a> {
    let leaves: Vec<&dyn LeafDispatch> =
        backends.leaves.iter().map(|a| a.as_ref() as _).collect();
    let ops: Vec<&dyn OpDispatch> = backends.ops.iter().map(|a| a.as_ref() as _).collect();
    // Leak the Vec into a 'static slice for the config — the slice lives
    // for the duration of this function call, but interpret_with_config
    // takes a borrow that escapes the call. We need stable backing.
    //
    // Simpler: just construct the config with leaked references that
    // outlive the interpret call. Since this function is called from a
    // blocking task that owns `backends` for the duration, the references
    // we hand out are valid.
    //
    // SAFETY note: we use Box::leak on a fresh Vec of references; the
    // Box backing only needs to outlive the interpret_with_config call,
    // which it does because we never return.
    let leaves_static: &'a [&'a dyn LeafDispatch] = Box::leak(leaves.into_boxed_slice());
    let ops_static: &'a [&'a dyn OpDispatch] = Box::leak(ops.into_boxed_slice());
    InterpretConfig {
        leaves: leaves_static,
        ops: ops_static,
    }
}

fn has_cycle(graph: &FunctionGraph) -> bool {
    let n = graph.nodes.len();
    let mut indeg = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for w in &graph.wires {
        adj[w.from].push(w.to);
        indeg[w.to] += 1;
    }
    let mut queue: std::collections::VecDeque<usize> = (0..n).filter(|i| indeg[*i] == 0).collect();
    let mut visited = 0usize;
    while let Some(u) = queue.pop_front() {
        visited += 1;
        for &v in &adj[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                queue.push_back(v);
            }
        }
    }
    visited != n
}

#[cfg(test)]
mod tests {
    use super::*;
    use rce_vm_backend_cpu::ScalarValue;
    use rce_graph::graph::{LiteralValue, OpKind, WireKind};

    fn add_graph() -> FunctionGraph {
        FunctionGraph {
            name: "add".into(),
            nodes: vec![
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Param(Param {
                    name: "y".into(),
                    data_type: "f32".into(),
                }),
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "f32".into(),
        }
    }

    #[test]
    fn scheduler_runs_simple_add() {
        let g = add_graph();
        let mut inputs = Inputs::new();
        inputs.insert("x".into(), ScalarValue::F32(3.0).into());
        inputs.insert("y".into(), ScalarValue::F32(4.0).into());
        let v = run(&g, inputs, Backends::default()).unwrap();
        match v {
            Value::Scalar(ScalarValue::F32(x)) => assert_eq!(x, 7.0),
            other => panic!("expected F32(7.0), got {other:?}"),
        }
    }

    /// Y-shaped graph: two independent literals + the same input, summed.
    /// Demonstrates parallel branch execution; correctness is the test.
    #[test]
    fn scheduler_runs_parallel_branches() {
        let g = FunctionGraph {
            name: "parallel".into(),
            nodes: vec![
                // 0: x
                GraphNode::Param(Param {
                    name: "x".into(),
                    data_type: "f32".into(),
                }),
                // 1: 10.0
                GraphNode::Literal {
                    value: LiteralValue::F32(10.0),
                    data_type: "f32".into(),
                },
                // 2: 100.0
                GraphNode::Literal {
                    value: LiteralValue::F32(100.0),
                    data_type: "f32".into(),
                },
                // 3: x + 10
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
                // 4: x + 100
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
                // 5: branch_a + branch_b
                GraphNode::Apply {
                    op: OpKind::Add,
                    output_type: "f32".into(),
                },
            ],
            wires: vec![
                Wire { from: 0, to: 3, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 3, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 0, to: 4, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 2, to: 4, port_in_idx: 1, wire_kind: WireKind::Move },
                Wire { from: 3, to: 5, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 4, to: 5, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 5,
            return_type: "f32".into(),
        };

        let mut inputs = Inputs::new();
        inputs.insert("x".into(), ScalarValue::F32(1.0).into());
        let v = run(&g, inputs, Backends::default()).unwrap();
        // (1+10) + (1+100) = 11 + 101 = 112
        match v {
            Value::Scalar(ScalarValue::F32(x)) => assert_eq!(x, 112.0),
            other => panic!("expected F32(112.0), got {other:?}"),
        }
    }

    #[test]
    fn scheduler_detects_cycles() {
        let g = FunctionGraph {
            name: "cyc".into(),
            nodes: vec![
                GraphNode::Apply { op: OpKind::Add, output_type: "i32".into() },
                GraphNode::Apply { op: OpKind::Add, output_type: "i32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 1, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 0, port_in_idx: 0, wire_kind: WireKind::Move },
            ],
            output_node: 1,
            return_type: "i32".into(),
        };
        let res = run(&g, Inputs::new(), Backends::default());
        assert!(matches!(res, Err(ScheduleError::Cyclic)));
    }

    /// Stage 4b: a priced backend that lists `"add"` in `handles()`
    /// intercepts `OpKind::Add` nodes — the same routing the existing
    /// demo uses for Custom("matmul") / Custom("silu") now works for
    /// built-in arithmetic ops too. Backward compatibility: with no
    /// priced backend claiming "add", the legacy interpreter handles
    /// `OpKind::Add` exactly as before (proven by the
    /// `scheduler_runs_simple_add` test above being unchanged).
    #[test]
    fn priced_backend_intercepts_built_in_add_when_handles_lists_it() {
        use crate::placement::PricedBackend;
        use rce_vm_backend_cpu::HostTensor;
        use std::sync::{Arc, Mutex};

        struct AddRecorder(Arc<Mutex<Vec<&'static str>>>);
        impl PricedBackend for AddRecorder {
            fn id(&self) -> &str { "test:add-recorder" }
            fn handles(&self) -> &[&'static str] { &["add"] }
            fn cost(&self, _op: &str, _shapes: &[&[usize]]) -> (f64, rce_vm_core::EnergyBasis) {
                (1.0, rce_vm_core::EnergyBasis::Reported)
            }
            fn execute(&self, op: &str, inputs: &[Value]) -> std::result::Result<Value, String> {
                self.0.lock().unwrap().push("add");
                // Sentinel output (one f32 element = 42.0) so the
                // assertion can prove this code path ran.
                let _ = (op, inputs);
                let bytes = (42.0f32).to_le_bytes().to_vec();
                HostTensor::new(vec![1], rce_vm_backend_cpu::DType::F32, bytes)
                    .map(Value::Tensor)
                    .map_err(|e| format!("{e:?}"))
            }
        }

        // Add graph that consumes two tensor params (so the intercept's
        // `has_tensor` check fires) and runs OpKind::Add.
        let g = FunctionGraph {
            name: "add-tensors".into(),
            nodes: vec![
                GraphNode::Param(Param { name: "x".into(), data_type: "f32".into() }),
                GraphNode::Param(Param { name: "y".into(), data_type: "f32".into() }),
                GraphNode::Apply { op: OpKind::Add, output_type: "f32".into() },
            ],
            wires: vec![
                Wire { from: 0, to: 2, port_in_idx: 0, wire_kind: WireKind::Move },
                Wire { from: 1, to: 2, port_in_idx: 1, wire_kind: WireKind::Move },
            ],
            output_node: 2,
            return_type: "f32".into(),
        };

        let calls = Arc::new(Mutex::new(Vec::new()));
        let backend: Arc<dyn PricedBackend> = Arc::new(AddRecorder(calls.clone()));

        let mut inputs = Inputs::new();
        let make_tensor = |v: f32| {
            Value::Tensor(HostTensor::new(
                vec![1],
                rce_vm_backend_cpu::DType::F32,
                v.to_le_bytes().to_vec(),
            ).unwrap())
        };
        inputs.insert("x".into(), make_tensor(3.0));
        inputs.insert("y".into(), make_tensor(4.0));

        let backends = Backends {
            leaves: vec![],
            ops: vec![],
            placement: vec![backend],
        };
        let v = run(&g, inputs, backends).unwrap();
        // Priced backend's sentinel reached the output ⇒ it intercepted.
        let recorded = calls.lock().unwrap().clone();
        assert_eq!(recorded, vec!["add"],
            "priced backend listing `add` in handles() must intercept OpKind::Add");
        match v {
            Value::Tensor(t) => {
                let out = f32::from_le_bytes([t.data[0], t.data[1], t.data[2], t.data[3]]);
                assert_eq!(out, 42.0, "got the recorder's sentinel output, not the interpreter's");
            }
            other => panic!("expected Tensor sentinel, got {other:?}"),
        }
    }
}
