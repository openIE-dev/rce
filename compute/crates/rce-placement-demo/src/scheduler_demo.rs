//! RCE placement-driven scheduler demo — multi-op + DeviceTensor preservation.
//!
//! Three things proven in one run:
//!
//! 1. **Per-op + per-size routing**: matmul/silu × small/large each route to
//!    the lowest-joule backend independently, inside `rce_vm_scheduler::run`.
//!
//! 2. **Cost-driven dispatch replaces first-claim-wins** for any Custom op
//!    claimed by a priced backend.
//!
//! 3. **DeviceTensor preservation across consecutive same-device ops**: a
//!    silu→silu chain on GPU runs the second silu against the *first silu's
//!    `DeviceTensor` output* — zero host roundtrip on the chain link. The
//!    audit log records per-input residency so the preservation is visible.
//!
//! This is the largest hidden cost in the pre-refactor path: every
//! PricedBackend::execute returned `HostTensor`, so a chain of GPU ops
//! demoted-uploaded once per node. The new `Value`-in / `Value`-out signature
//! lets MetalLeaf's `DeviceTensor` output flow directly into the next
//! MetalLeaf dispatch without leaving GPU memory.

use std::sync::{Arc, Mutex};

use rce_op_metal::MetalLeaf;
use rce_op_mps::MpsMatmul;
use rce_op_nn_prims::{GeluLeaf, RmsNormLeaf, SiluLeaf};
use rce_placement::{
    BackendId, CostTable, LinearCoeffs, elementwise_flops_bytes, matmul_flops_bytes,
};
use rce_vm_backend_cpu::{DType, HostTensor, Inputs, OpDispatch, ScalarValue, Value};
use rce_vm_scheduler::{Backends, PricedBackend, run};
use rce_graph::graph::{
    FunctionGraph, GraphNode, OpKind, Param, Wire, WireKind,
};

// ── Audit log ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct CallRecord {
    op: String,
    backend: String,
    input_shapes: Vec<Vec<usize>>,
    input_residency: Vec<&'static str>, // "host" | "device" | "scalar"
    output_residency: &'static str,
}

#[derive(Clone)]
struct AuditLog(Arc<Mutex<Vec<CallRecord>>>);
impl AuditLog {
    fn new() -> Self { Self(Arc::new(Mutex::new(Vec::new()))) }
    fn push(&self, r: CallRecord) { self.0.lock().unwrap().push(r); }
    fn snapshot(&self) -> Vec<CallRecord> { self.0.lock().unwrap().clone() }
}

fn residency_of(v: &Value) -> &'static str {
    match v {
        Value::Tensor(_) => "host",
        Value::DeviceTensor(_) => "device",
        Value::Scalar(_) => "scalar",
    }
}

fn shape_of(v: &Value) -> Vec<usize> {
    match v {
        Value::Tensor(t) => t.shape.clone(),
        Value::DeviceTensor(d) => d.shape.clone(),
        _ => vec![],
    }
}

// ── PricedBackend impls ───────────────────────────────────────────

/// Generic FLOPs/bytes estimator used by both CPU and GPU priced backends.
/// `shapes` is the tensor-shape slice the scheduler passes (scalar args
/// were filtered out — see scheduler.rs intercept), so `shapes[0]` is the
/// first tensor input regardless of where scalars sit in the full args.
fn op_flops_bytes(op: &str, shapes: &[&[usize]]) -> (u64, u64) {
    match op {
        "matmul" => matmul_flops_bytes(
            shapes[0][0] as u64, shapes[0][1] as u64, shapes[1][1] as u64, 4),
        "silu" | "gelu" => {
            let n: usize = shapes[0].iter().product();
            elementwise_flops_bytes(n as u64, 4)
        }
        "rms_norm" => {
            // input + weight, both elementwise; ~3 FLOPs/elem (square, add,
            // mul) on the reduction + 2 FLOPs/elem (mul, mul) on the scale.
            let n: usize = shapes[0].iter().product();
            let flops = 5 * n as u64;
            let bytes = 3 * 4 * n as u64; // x + w in, out out
            (flops, bytes)
        }
        _ => (0, 0),
    }
}

/// CPU backend. Demotes any device-resident input first (CPU can't read GPU
/// memory directly); always returns host-resident output.
struct CpuPriced {
    id: String,
    handles: Vec<&'static str>,
    table: Arc<CostTable>,
    audit: AuditLog,
    silu: SiluLeaf,
    gelu: GeluLeaf,
    rms_norm: RmsNormLeaf,
    metal_for_demote: Arc<MetalLeaf>,
}

impl CpuPriced {
    fn demote(&self, v: &Value) -> Result<Value, String> {
        match v {
            Value::Tensor(_) | Value::Scalar(_) => Ok(v.clone()),
            Value::DeviceTensor(d) => self.metal_for_demote
                .demote_to_host(d)
                .ok_or_else(|| "demote_to_host returned None".to_string()),
        }
    }
}

impl PricedBackend for CpuPriced {
    fn id(&self) -> &str { &self.id }
    fn handles(&self) -> &[&'static str] { &self.handles }

    fn cost(&self, op: &str, shapes: &[&[usize]]) -> (f64, bool) {
        let (flops, bytes) = op_flops_bytes(op, shapes);
        let est = self.table.estimate(op, &BackendId::new(self.id.clone()), flops, bytes);
        (est.picojoules, est.measured)
    }

    fn execute(&self, op: &str, inputs: &[Value]) -> Result<Value, String> {
        // Audit only tensor inputs (scalars don't have a shape).
        let input_residency: Vec<&'static str> = inputs.iter()
            .filter(|v| !matches!(v, Value::Scalar(_)))
            .map(|v| residency_of(v))
            .collect();
        let input_shapes: Vec<Vec<usize>> = inputs.iter()
            .filter(|v| !matches!(v, Value::Scalar(_)))
            .map(|v| shape_of(v))
            .collect();

        // Demote device-resident inputs to host before CPU dispatch.
        let host_inputs: Vec<Value> = inputs.iter()
            .map(|v| self.demote(v))
            .collect::<Result<_, _>>()?;

        let out = match op {
            "matmul" => {
                let (a, b) = match (&host_inputs[0], &host_inputs[1]) {
                    (Value::Tensor(a), Value::Tensor(b)) => (a, b),
                    _ => return Err("matmul requires tensor inputs".into()),
                };
                let m = a.shape[0]; let k = a.shape[1]; let n = b.shape[1];
                let av = a.as_f32().map_err(|e| format!("a as_f32: {e:?}"))?;
                let bv = b.as_f32().map_err(|e| format!("b as_f32: {e:?}"))?;
                let mut c = vec![0.0f32; m * n];
                rce_op_blas::sgemm_into(&av, &bv, &mut c, m, k, n);
                let bytes: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
                Value::Tensor(HostTensor::new(vec![m, n], DType::F32, bytes)
                    .map_err(|e| format!("{e:?}"))?)
            }
            "silu" => self.silu.execute(0, "silu", &host_inputs)
                .map_err(|e| format!("SiluLeaf: {e:?}"))?,
            "gelu" => self.gelu.execute(0, "gelu", &host_inputs)
                .map_err(|e| format!("GeluLeaf: {e:?}"))?,
            "rms_norm" => self.rms_norm.execute(0, "rms_norm", &host_inputs)
                .map_err(|e| format!("RmsNormLeaf: {e:?}"))?,
            _ => return Err(format!("CpuPriced: unhandled op {op}")),
        };

        self.audit.push(CallRecord {
            op: op.to_string(),
            backend: self.id.clone(),
            input_shapes,
            input_residency,
            output_residency: residency_of(&out),
        });
        Ok(out)
    }
}

/// GPU backend. Accepts device-resident inputs without re-uploading them.
/// `silu` (via MetalLeaf) returns `DeviceTensor` so chained GPU ops stay
/// resident. `matmul` (via MpsMatmul) now accepts `Value::DeviceTensor`
/// inputs directly via `rce_op_metal::metal_buffer_of` — a silu→matmul
/// chain on GPU does NOT demote between dispatches.
struct GpuPriced {
    id: String,
    handles: Vec<&'static str>,
    table: Arc<CostTable>,
    audit: AuditLog,
    mps: Arc<MpsMatmul>,
    metal: Arc<MetalLeaf>,
}

impl PricedBackend for GpuPriced {
    fn id(&self) -> &str { &self.id }
    fn handles(&self) -> &[&'static str] { &self.handles }

    fn cost(&self, op: &str, shapes: &[&[usize]]) -> (f64, bool) {
        let (flops, bytes) = op_flops_bytes(op, shapes);
        let est = self.table.estimate(op, &BackendId::new(self.id.clone()), flops, bytes);
        (est.picojoules, est.measured)
    }

    fn execute(&self, op: &str, inputs: &[Value]) -> Result<Value, String> {
        let input_residency: Vec<&'static str> = inputs.iter()
            .filter(|v| !matches!(v, Value::Scalar(_)))
            .map(|v| residency_of(v))
            .collect();
        let input_shapes: Vec<Vec<usize>> = inputs.iter()
            .filter(|v| !matches!(v, Value::Scalar(_)))
            .map(|v| shape_of(v))
            .collect();

        let out = match op {
            "matmul" => {
                // MpsMatmul now accepts Value::DeviceTensor inputs directly
                // (borrowed MTLBuffer via rce-op-metal::metal_buffer_of).
                // No demote on input; the silu→matmul GPU chain skips one
                // device→host copy per device-resident input.
                self.mps.execute(0, "matmul", inputs)
                    .map_err(|e| format!("MPS dispatch: {e:?}"))?
            }
            "silu" | "gelu" | "rms_norm" => {
                // MetalLeaf preserves input residency (HOST in → HOST out,
                // DEVICE in → DEVICE out). Promote tensor inputs to device
                // on entry so MetalLeaf returns a `DeviceTensor`; the next
                // GPU op in the chain then skips both copies.
                // For rms_norm, scalar args (ngroups, eps) pass through
                // unchanged — `promote_to_device` is only attempted on
                // tensor-valued args.
                let promoted: Vec<Value> = inputs.iter().map(|v| match v {
                    Value::DeviceTensor(_) | Value::Scalar(_) => v.clone(),
                    Value::Tensor(t) => self.metal
                        .promote_to_device(t)
                        .unwrap_or_else(|| v.clone()),
                }).collect();
                self.metal.execute(0, op, &promoted)
                    .map_err(|e| format!("MetalLeaf {op}: {e:?}"))?
            }
            _ => return Err(format!("GpuPriced: unhandled op {op}")),
        };

        self.audit.push(CallRecord {
            op: op.to_string(),
            backend: self.id.clone(),
            input_shapes,
            input_residency,
            output_residency: residency_of(&out),
        });
        Ok(out)
    }
}

// ── Cost table ────────────────────────────────────────────────────

fn build_table() -> Arc<CostTable> {
    let mut t = CostTable::new(LinearCoeffs {
        a_pj_per_flop: 1.0, b_pj_per_byte: 0.0, c_pj_overhead: 0.0,
    });
    t.insert("matmul", &BackendId::new("cpu:op-blas"),
        LinearCoeffs { a_pj_per_flop: 5.0, b_pj_per_byte: 0.0, c_pj_overhead: 1_000_000.0 },
        true);
    t.insert("matmul", &BackendId::new("gpu:op-mps"),
        LinearCoeffs { a_pj_per_flop: 0.29, b_pj_per_byte: 0.0, c_pj_overhead: 5_000_000_000.0 },
        true);
    t.insert("silu", &BackendId::new("cpu:op-nn-prims"),
        LinearCoeffs { a_pj_per_flop: 50.0, b_pj_per_byte: 0.2, c_pj_overhead: 100_000.0 },
        false);
    t.insert("silu", &BackendId::new("gpu:op-metal"),
        LinearCoeffs { a_pj_per_flop: 2.0, b_pj_per_byte: 0.02, c_pj_overhead: 500_000_000.0 },
        false);
    // gelu — same FLOPs/elem class as silu; coefficients mirror it.
    t.insert("gelu", &BackendId::new("cpu:op-nn-prims"),
        LinearCoeffs { a_pj_per_flop: 50.0, b_pj_per_byte: 0.2, c_pj_overhead: 100_000.0 },
        false);
    t.insert("gelu", &BackendId::new("gpu:op-metal"),
        LinearCoeffs { a_pj_per_flop: 2.0, b_pj_per_byte: 0.02, c_pj_overhead: 500_000_000.0 },
        false);
    // rms_norm — bandwidth-bound (x + w in, out out → 3× elementwise
    // bytes-touched at moderate FLOPs/elem). GPU wins at large N because
    // the per-byte coefficient dominates.
    t.insert("rms_norm", &BackendId::new("cpu:op-nn-prims"),
        LinearCoeffs { a_pj_per_flop: 30.0, b_pj_per_byte: 0.3, c_pj_overhead: 100_000.0 },
        false);
    t.insert("rms_norm", &BackendId::new("gpu:op-metal"),
        LinearCoeffs { a_pj_per_flop: 1.5, b_pj_per_byte: 0.03, c_pj_overhead: 500_000_000.0 },
        false);
    Arc::new(t)
}

// ── Graph & inputs ────────────────────────────────────────────────

fn rand_f32(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
    }).collect()
}

fn host_tensor_value(data: Vec<f32>, shape: Vec<usize>) -> Value {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Value::Tensor(HostTensor::new(shape, DType::F32, bytes).expect("host tensor"))
}

/// 7 Apply nodes spanning four ops (matmul, silu, gelu, rms_norm) and three
/// backends, with a silu→rms_norm→matmul GPU chain that exercises both
/// MetalLeaf and MpsMatmul on `Value::DeviceTensor` inputs, *plus* a
/// scalar-arg op (rms_norm takes ngroups + eps scalars alongside its two
/// tensor inputs — proving the scheduler intercept now passes scalars
/// through cleanly).
fn build_demo_graph() -> FunctionGraph {
    let param = |name: &str| GraphNode::Param(Param {
        name: name.into(), data_type: "f32".into(),
    });
    let scalar_i32_param = |name: &str| GraphNode::Param(Param {
        name: name.into(), data_type: "i32".into(),
    });
    let scalar_f32_param = |name: &str| GraphNode::Param(Param {
        name: name.into(), data_type: "f32".into(),
    });
    let app = |op: &str| GraphNode::Apply {
        op: OpKind::Custom(op.into()),
        output_type: "f32".into(),
    };
    let w = |from: usize, to: usize, port: i32| Wire {
        from, to, port_in_idx: port, wire_kind: WireKind::Move,
    };
    //   0: Param  a_small        [128,128]
    //   1: Param  b_small        [128,128]
    //   2: Apply  matmul(0,1)                              ← matmul-128 → CPU
    //   3: Param  x_silu_small   [64,64]
    //   4: Apply  silu(3)                                  ← silu-small → CPU
    //   5: Param  x_gelu_small   [64,64]
    //   6: Apply  gelu(5)                                  ← gelu-small → CPU
    //   7: Param  x_silu_chain   [2048,2048]
    //   8: Apply  silu(7)                                  ← silu-chain  → GPU (host in, device out)
    //   9: Param  rms_weight     [2048,2048]
    //  10: Param  rms_ngroups    (i32)
    //  11: Param  rms_eps        (f32)
    //  12: Apply  rms_norm(8, 9, 10, 11)                   ← rms_norm-chain → GPU (device in, device out)
    //  13: Param  w_chain        [2048,2048]
    //  14: Apply  matmul(12, 13)                           ← matmul-2048 → GPU (device in, host out)  ← output
    FunctionGraph {
        name: "multi_op_with_scalar_args".into(),
        nodes: vec![
            param("a_small"), param("b_small"), app("matmul"),
            param("x_silu_small"), app("silu"),
            param("x_gelu_small"), app("gelu"),
            param("x_silu_chain"), app("silu"),
            param("rms_weight"),
            scalar_i32_param("rms_ngroups"),
            scalar_f32_param("rms_eps"),
            app("rms_norm"),
            param("w_chain"), app("matmul"),
        ],
        wires: vec![
            w(0, 2, 0), w(1, 2, 1),
            w(3, 4, 0),
            w(5, 6, 0),
            w(7, 8, 0),
            w(8, 12, 0),                          // silu → rms_norm (device input)
            w(9, 12, 1),                          // rms_weight
            w(10, 12, 2),                         // ngroups scalar
            w(11, 12, 3),                         // eps scalar
            w(12, 14, 0),                         // rms_norm → matmul (device input!)
            w(13, 14, 1),                         // w_chain
        ],
        output_node: 14,
        return_type: "f32".into(),
    }
}

// ── Demo ───────────────────────────────────────────────────────────

fn main() {
    println!("# RCE scheduler demo — multi-op routing + DeviceTensor preservation");
    println!();

    let table = build_table();
    let audit = AuditLog::new();

    let mps_inner = match MpsMatmul::with_min_flops(0) {
        Some(m) => Arc::new(m),
        None => { eprintln!("MPS unavailable; demo needs Apple Silicon."); std::process::exit(1); }
    };
    let metal_inner = match MetalLeaf::new() {
        Some(m) => Arc::new(m),
        None => { eprintln!("MetalLeaf unavailable; demo needs Apple Silicon."); std::process::exit(1); }
    };

    let cpu_mm = Arc::new(CpuPriced {
        id: "cpu:op-blas".into(),
        handles: vec!["matmul"],
        table: table.clone(),
        audit: audit.clone(),
        silu: SiluLeaf,
        gelu: GeluLeaf,
        rms_norm: RmsNormLeaf,
        metal_for_demote: metal_inner.clone(),
    });
    let cpu_nn = Arc::new(CpuPriced {
        id: "cpu:op-nn-prims".into(),
        handles: vec!["silu", "gelu", "rms_norm"],
        table: table.clone(),
        audit: audit.clone(),
        silu: SiluLeaf,
        gelu: GeluLeaf,
        rms_norm: RmsNormLeaf,
        metal_for_demote: metal_inner.clone(),
    });
    let gpu_mm = Arc::new(GpuPriced {
        id: "gpu:op-mps".into(),
        handles: vec!["matmul"],
        table: table.clone(),
        audit: audit.clone(),
        mps: mps_inner.clone(),
        metal: metal_inner.clone(),
    });
    let gpu_nn = Arc::new(GpuPriced {
        id: "gpu:op-metal".into(),
        handles: vec!["silu", "gelu", "rms_norm"],
        table: table.clone(),
        audit: audit.clone(),
        mps: mps_inner.clone(),
        metal: metal_inner.clone(),
    });

    let backends = Backends {
        leaves: vec![],
        ops: vec![],
        placement: vec![cpu_mm.clone(), gpu_mm.clone(), cpu_nn.clone(), gpu_nn.clone()],
    };

    let graph = build_demo_graph();

    let mut seed = 0xdeadbeefcafebabeu64;
    let mut inputs: Inputs = Inputs::new();
    inputs.insert("a_small".into(),      host_tensor_value(rand_f32(128*128,   &mut seed), vec![128, 128]));
    inputs.insert("b_small".into(),      host_tensor_value(rand_f32(128*128,   &mut seed), vec![128, 128]));
    inputs.insert("x_silu_small".into(), host_tensor_value(rand_f32(64*64,     &mut seed), vec![64, 64]));
    inputs.insert("x_gelu_small".into(), host_tensor_value(rand_f32(64*64,     &mut seed), vec![64, 64]));
    inputs.insert("x_silu_chain".into(), host_tensor_value(rand_f32(2048*2048, &mut seed), vec![2048, 2048]));
    inputs.insert("rms_weight".into(),   host_tensor_value(rand_f32(2048*2048, &mut seed), vec![2048, 2048]));
    inputs.insert("rms_ngroups".into(),  Value::Scalar(ScalarValue::I32(2048))); // per-row norm
    inputs.insert("rms_eps".into(),      Value::Scalar(ScalarValue::F32(1e-6)));
    inputs.insert("w_chain".into(),      host_tensor_value(rand_f32(2048*2048, &mut seed), vec![2048, 2048]));

    println!("running 7-Apply graph through rce-vm-scheduler ...");
    let out = run(&graph, inputs, backends).expect("scheduler run");
    println!("output_node returned: {} (residency: {})",
        match &out {
            Value::Tensor(t) => format!("tensor f32 shape={:?}", t.shape),
            Value::DeviceTensor(d) => format!("device-tensor shape={:?} on {}", d.shape, d.device.name()),
            other => format!("{other:?}"),
        },
        residency_of(&out),
    );
    println!();

    let log = audit.snapshot();
    println!("# audit log — per-node routing + I/O residency:");
    for (i, r) in log.iter().enumerate() {
        let ins = r.input_shapes.iter()
            .zip(r.input_residency.iter())
            .map(|(s, res)| format!("{res}:{s:?}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  call {}: {:<7} [{}] → {} (out: {})",
            i + 1, r.op, ins, r.backend, r.output_residency);
    }
    println!();

    // ── Verification ──
    let mut all_ok = true;

    let route = |op: &str, predicate: &dyn Fn(&Vec<Vec<usize>>) -> bool| -> Option<&CallRecord> {
        log.iter().find(|r| r.op == op && predicate(&r.input_shapes))
    };
    let expectations: Vec<(&str, Option<&CallRecord>, &str)> = vec![
        ("matmul-128",     route("matmul",   &|s| s[0][0] == 128),  "cpu:op-blas"),
        ("matmul-2048",    route("matmul",   &|s| s[0][0] == 2048), "gpu:op-mps"),
        ("silu-64",        route("silu",     &|s| s[0][0] == 64),   "cpu:op-nn-prims"),
        ("silu-2048",      route("silu",     &|s| s[0][0] == 2048), "gpu:op-metal"),
        ("gelu-64",        route("gelu",     &|s| s[0][0] == 64),   "cpu:op-nn-prims"),
        ("rms_norm-2048",  route("rms_norm", &|s| s[0][0] == 2048), "gpu:op-metal"),
    ];
    println!("# routing checks:");
    for (label, got, expected) in &expectations {
        let actual = got.map(|r| r.backend.as_str()).unwrap_or("MISSING");
        let ok = actual == *expected;
        if !ok { all_ok = false; }
        println!("  {:<18} expected {:<18} got {:<18} {}",
            label, expected, actual, if ok { "OK" } else { "WRONG" });
    }

    // Chain residency: silu(8) → rms_norm(12) → matmul(14) — both
    // downstream consumers got DEVICE input thanks to MetalLeaf/MpsMatmul
    // accepting Value::DeviceTensor end-to-end.
    let rms_call = log.iter().find(|r| r.op == "rms_norm" && r.backend == "gpu:op-metal");
    if let Some(r) = rms_call {
        let arg0_device = r.input_residency.first().copied() == Some("device");
        if !arg0_device { all_ok = false; }
        println!("  {:<18} expected arg0 input    device  got {:<6}              {}",
            "rms_norm-from-silu", r.input_residency.first().copied().unwrap_or("?"),
            if arg0_device { "OK (MetalLeaf rms_norm got DeviceTensor)" } else { "WRONG" });
    } else {
        all_ok = false;
        println!("  rms_norm-from-silu MetalLeaf rms_norm call missing                            WRONG");
    }
    let mps_call = log.iter().find(|r| r.op == "matmul" && r.backend == "gpu:op-mps");
    if let Some(r) = mps_call {
        let arg0_device = r.input_residency.first().copied() == Some("device");
        if !arg0_device { all_ok = false; }
        println!("  {:<18} expected arg0 input    device  got {:<6}              {}",
            "matmul-from-rmsnorm", r.input_residency.first().copied().unwrap_or("?"),
            if arg0_device { "OK (MPS accepted DeviceTensor)" } else { "WRONG (forced demote!)" });
        // NEW (slice 3): MPS preserves residency — device input ⇒ device
        // output. The chain tail now stays GPU-resident *including* the
        // matmul output; no more forced demote at chain end.
        let out_device = r.output_residency == "device";
        if !out_device { all_ok = false; }
        println!("  {:<18} expected matmul output device  got {:<6}              {}",
            "matmul-output", r.output_residency,
            if out_device { "OK (MpsMatmul preserves residency)" } else { "WRONG (chain-end demote!)" });
    } else {
        all_ok = false;
        println!("  matmul-from-rmsnorm MPS call missing                                          WRONG");
    }

    // Final scheduler output is now the matmul-from-rmsnorm DeviceTensor
    // (no demote at the chain end).
    let final_output_device = matches!(out, Value::DeviceTensor(_));
    if !final_output_device { all_ok = false; }
    println!("  {:<18} expected scheduler out device  got {:<6}              {}",
        "scheduler-output", residency_of(&out),
        if final_output_device { "OK (chain end stays device-resident)" } else { "WRONG" });

    println!();
    if all_ok {
        println!("# all checks passed:");
        println!("#  - 4 ops × per-size cost-driven dispatch (matmul, silu, gelu, rms_norm);");
        println!("#  - scalar args (rms_norm's ngroups + eps) pass through the placement intercept;");
        println!("#  - DeviceTensor preserved silu→rms_norm→matmul, INCLUDING the matmul output;");
        println!("#  - the entire chain tail flows from device to device with no host roundtrip.");
    } else {
        eprintln!("# one or more checks failed.");
        std::process::exit(2);
    }
}
