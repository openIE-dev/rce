//! RCE placement demo — Stage 5 wiring proof.
//!
//! End-to-end demonstration that **cost-driven placement actually executes**:
//! `pick_backend` chooses the lowest-joule backend from a cost-priced HAL,
//! and `dispatch_f32` then runs the matmul on the chosen leaf. CPU and GPU
//! results are cross-checked for correctness within IEEE-754 tolerance.
//!
//! This is what was missing from `rce-placement`'s tests (the routing
//! decision was proved via `pick_backend(&slice, &backends)` returning the
//! expected `BackendId`, but `Backend::dispatch` was stubbed). Here the same
//! decision is followed by a real `op-blas::sgemm_into` or
//! `MpsMatmul::execute` call — closing the loop.

use std::sync::Arc;

use rce_op_mps::MpsMatmul;
use rce_placement::{
    Backend, BackendId, BackendKind, Capabilities, Cost, CostTable, DispatchError, GraphSlice,
    LinearCoeffs, Plan, TensorHandle, Value, matmul_flops_bytes, matmul_slice, pick_backend,
    unpack_matmul,
};
use rce_vm_backend_cpu::{DType, HostTensor, OpDispatch, Value as CpuValue};

// ── Concrete priced backends with real dispatch ───────────────────

struct BlasMatmulBackend {
    id: BackendId,
    caps: Capabilities,
    table: Arc<CostTable>,
}

impl BlasMatmulBackend {
    fn new(table: Arc<CostTable>) -> Self {
        Self {
            id: BackendId::new("cpu:op-blas"),
            caps: Capabilities {
                kind: Some(BackendKind::Cpu),
                supports_f32: true,
                ..Default::default()
            },
            table,
        }
    }

    /// Real Stage-5 dispatch: run sgemm on the CPU and return the result.
    /// `Backend::dispatch` takes `&[TensorHandle]` (opaque u64s) which would
    /// need a value store; the demo bypasses that with a typed entry point.
    fn dispatch_f32(&self, plan: &Plan, a: &[f32], b: &[f32]) -> Vec<f32> {
        let (m, k, n) = unpack_matmul(&plan.artifact).expect("matmul slice");
        let (m, k, n) = (m as usize, k as usize, n as usize);
        assert_eq!(a.len(), m * k, "A shape mismatch");
        assert_eq!(b.len(), k * n, "B shape mismatch");
        let mut c = vec![0.0f32; m * n];
        rce_op_blas::sgemm_into(a, b, &mut c, m, k, n);
        c
    }
}

impl Backend for BlasMatmulBackend {
    fn id(&self) -> &BackendId { &self.id }
    fn capabilities(&self) -> Capabilities { self.caps.clone() }
    fn compile(&self, slice: &GraphSlice) -> Result<Plan, DispatchError> {
        unpack_matmul(&slice.graph_cbor).ok_or_else(|| {
            DispatchError::CapabilityDenied("op-blas backend handles only matmul slices".into())
        })?;
        Ok(Plan {
            backend: self.id.clone(),
            artifact: slice.graph_cbor.clone(),
            placements: vec![],
            determinism_strict: slice.strict_deterministic,
        })
    }
    fn dispatch(
        &self,
        _plan: &Plan,
        _inputs: &[TensorHandle],
    ) -> Result<Vec<Value>, DispatchError> {
        Err(DispatchError::Driver(
            "use dispatch_f32 (the TensorHandle path needs a value store, Stage 5+)".into(),
        ))
    }
    fn cost(&self, plan: &Plan) -> Cost {
        let (m, k, n) = unpack_matmul(&plan.artifact).unwrap_or((0, 0, 0));
        let (flops, bytes) = matmul_flops_bytes(m, k, n, 4);
        Cost {
            energy: self.table.estimate("matmul", &plan.backend, flops, bytes),
            time_ns: 0,
            peak_bytes: 0,
        }
    }
}

struct MpsMatmulBackend {
    id: BackendId,
    caps: Capabilities,
    table: Arc<CostTable>,
    mps: MpsMatmul,
}

impl MpsMatmulBackend {
    fn new(table: Arc<CostTable>) -> Option<Self> {
        // min_flops=0 ⇒ never fall back to BLAS; always GPU.
        let mps = MpsMatmul::with_min_flops(0)?;
        Some(Self {
            id: BackendId::new("gpu:op-mps"),
            caps: Capabilities {
                kind: Some(BackendKind::Metal),
                supports_f32: true,
                device_memory: true,
                ..Default::default()
            },
            table,
            mps,
        })
    }

    fn dispatch_f32(&self, plan: &Plan, a: &[f32], b: &[f32]) -> Vec<f32> {
        let (m, k, n) = unpack_matmul(&plan.artifact).expect("matmul slice");
        let (m, k, n) = (m as usize, k as usize, n as usize);
        let av = host_tensor_f32(a, vec![m, k]);
        let bv = host_tensor_f32(b, vec![k, n]);
        let out = self
            .mps
            .execute(0, "matmul", &[av, bv])
            .expect("MPS dispatch");
        match out {
            CpuValue::Tensor(t) => t
                .as_f32()
                .expect("f32 result")
                .to_vec(),
            other => panic!("expected Value::Tensor, got {other:?}"),
        }
    }
}

impl Backend for MpsMatmulBackend {
    fn id(&self) -> &BackendId { &self.id }
    fn capabilities(&self) -> Capabilities { self.caps.clone() }
    fn compile(&self, slice: &GraphSlice) -> Result<Plan, DispatchError> {
        unpack_matmul(&slice.graph_cbor).ok_or_else(|| {
            DispatchError::CapabilityDenied("op-mps backend handles only matmul slices".into())
        })?;
        Ok(Plan {
            backend: self.id.clone(),
            artifact: slice.graph_cbor.clone(),
            placements: vec![],
            determinism_strict: slice.strict_deterministic,
        })
    }
    fn dispatch(
        &self,
        _plan: &Plan,
        _inputs: &[TensorHandle],
    ) -> Result<Vec<Value>, DispatchError> {
        Err(DispatchError::Driver(
            "use dispatch_f32 (the TensorHandle path needs a value store, Stage 5+)".into(),
        ))
    }
    fn cost(&self, plan: &Plan) -> Cost {
        let (m, k, n) = unpack_matmul(&plan.artifact).unwrap_or((0, 0, 0));
        let (flops, bytes) = matmul_flops_bytes(m, k, n, 4);
        Cost {
            energy: self.table.estimate("matmul", &plan.backend, flops, bytes),
            time_ns: 0,
            peak_bytes: 0,
        }
    }
}

fn host_tensor_f32(data: &[f32], shape: Vec<usize>) -> CpuValue {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    CpuValue::Tensor(HostTensor::new(shape, DType::F32, bytes).expect("host tensor"))
}

// ── Cost table (calibrated coefficients, M3 Ultra) ────────────────

fn build_table() -> Arc<CostTable> {
    // Illustrative coefficients, not a reading (the 86 mJ and 5 mJ notes
    // of commit 8f1e69e63 are not a meter record kept here). Basis is
    // `reported` when a counter calibration is loaded, `est` for a model.
    let mut t = CostTable::new(LinearCoeffs {
        a_pj_per_flop: 1.0,
        b_pj_per_byte: 0.0,
        c_pj_overhead: 0.0,
    });
    t.insert(
        "matmul",
        &BackendId::new("cpu:op-blas"),
        LinearCoeffs { a_pj_per_flop: 5.0, b_pj_per_byte: 0.0, c_pj_overhead: 1_000_000.0 },
        rce_placement::EnergyBasis::Est,
    );
    t.insert(
        "matmul",
        &BackendId::new("gpu:op-mps"),
        LinearCoeffs { a_pj_per_flop: 0.29, b_pj_per_byte: 0.0, c_pj_overhead: 5_000_000_000.0 },
        rce_placement::EnergyBasis::Est,
    );
    Arc::new(t)
}

// ── Demo: pick → dispatch → cross-check ───────────────────────────

fn rand_f32(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn rel_max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-6);
            ((x - y).abs() / denom)
        })
        .fold(0.0f32, f32::max)
}

fn main() {
    println!("# RCE placement demo — Stage 5 wiring: pick_backend → real dispatch");
    println!("# coefficients: CPU(op-blas) a=5.0 pJ/F, c=1µJ ; GPU(op-mps) a=0.29 pJ/F, c=5mJ");
    println!();

    let table = build_table();
    let blas = Arc::new(BlasMatmulBackend::new(table.clone()));
    let mps = match MpsMatmulBackend::new(table.clone()) {
        Some(b) => Arc::new(b),
        None => {
            eprintln!("MPS unavailable; this demo needs Apple Silicon.");
            std::process::exit(1);
        }
    };
    let backends: Vec<Arc<dyn Backend>> = vec![blas.clone(), mps.clone()];

    println!(
        "{:>6}  {:>14}  {:>14}  {:>16}  {:>10}",
        "N", "chosen", "predicted pJ", "result[0]", "rel_max_diff",
    );
    println!("{}", "-".repeat(72));

    let mut seed = 0xdeadbeefcafebabeu64;
    let mut overall_max_diff = 0.0f32;
    for &n in &[64usize, 128, 256, 512, 1024, 2048] {
        let slice = matmul_slice(n as u64, n as u64, n as u64, false);
        let (chosen_id, est) = pick_backend(&slice, &backends).expect("at least one backend");

        let a = rand_f32(n * n, &mut seed);
        let b = rand_f32(n * n, &mut seed);

        // Always compute the CPU baseline (cheap-ish, gives us ground truth).
        let cpu_plan = blas.compile(&slice).unwrap();
        let cpu_result = blas.dispatch_f32(&cpu_plan, &a, &b);

        // Dispatch on whichever backend pick_backend chose. The architecture
        // proof: the routing decision and the execution are wired together.
        let (chosen_result, label) = match chosen_id.0.as_str() {
            "cpu:op-blas" => (cpu_result.clone(), "CPU".to_string()),
            "gpu:op-mps" => {
                let plan = mps.compile(&slice).unwrap();
                (mps.dispatch_f32(&plan, &a, &b), "GPU".to_string())
            }
            other => panic!("unexpected backend {other}"),
        };

        let diff = rel_max_diff(&cpu_result, &chosen_result);
        overall_max_diff = overall_max_diff.max(diff);

        println!(
            "{:>6}  {:>14}  {:>14.2e}  {:>16.6}  {:>10.2e}",
            n,
            label,
            est.picojoules,
            chosen_result.first().copied().unwrap_or(0.0),
            diff
        );
    }

    println!();
    println!("# overall worst CPU↔chosen relative difference: {:.2e}", overall_max_diff);
    if overall_max_diff < 5e-4 {
        println!("# correctness OK — the GPU/CPU paths agree within tolerance.");
        println!("# routing decision flips by shape via min-joule cost; both paths produce the");
        println!("# same matmul result. The energy-routing thesis runs end-to-end on real");
        println!("# hardware.");
    } else {
        eprintln!("# correctness FAIL — CPU and chosen backend disagree (>{:.0e}).", 5e-4);
        std::process::exit(2);
    }
}
