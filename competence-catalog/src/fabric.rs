//! Heterogeneous compute by design — one Form, lowered to **every** compute fabric.
//!
//! RCE is not a CPU compiler: the Form lowers to CPU (x86/ARM/RISC-V), GPU (NVIDIA PTX, AMD
//! CDNA, Apple Metal), tensor (AMX/ANE/TPU/Ascend/Cambricon), dataflow/spatial (Tenstorrent
//! Tensix, Groq, Cerebras, Graphcore, CGRA), WASM/WebGPU, and the ML IRs (MLIR/StableHLO/Triton/
//! ONNX). Each fabric is a row in a capability+cost+template table (data, like the emission
//! LUTs); lowering is a deterministic lookup; placement = argmin joules across the whole
//! heterogeneous set (flowg's OpDispatch generalized). Adding a fabric is adding a row.
//!
//! Cost coefficients are a modelled (est) Horowitz-form prior; measured calibration replaces it.

use crate::placement::{CostModel, Prim};

/// The fabric family (the shape of its primitive floor).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Cpu,
    Gpu,
    Tensor,
    Dataflow,
    Wasm,
    MlIr,
}

/// A compute fabric: capability (lookup), cost model, and lowering templates (data).
pub struct Fabric {
    pub name: &'static str,
    pub kind: Kind,
    pub has_fma: bool,
    pub has_tensor: bool,
    pub model: CostModel,
    /// The fused `a*b+c` lowering (used iff the capability says FMA/tensor-fma exists).
    pub fused: &'static str,
    /// The split `mul+add` lowering (fallback).
    pub split: &'static str,
}

/// The global heterogeneous fabric table. Adding a fabric = adding a row.
pub fn builtin_fabrics() -> Vec<Fabric> {
    let cpu = CostModel { a: 1.0, b: 100.0, c: 1.0 };
    let gpu = CostModel { a: 0.1, b: 10.0, c: 0.5 };
    let tensor = CostModel { a: 0.02, b: 2.0, c: 0.3 };
    let dataflow = CostModel { a: 0.03, b: 1.0, c: 0.2 };
    let wasm = CostModel { a: 0.8, b: 80.0, c: 0.8 };
    vec![
        Fabric { name: "cpu-x86", kind: Kind::Cpu, has_fma: true, has_tensor: false, model: cpu,
            fused: "vfmadd213ss xmm0, xmm1, xmm2", split: "mulss xmm0, xmm1\naddss xmm0, xmm2" },
        Fabric { name: "cpu-arm64", kind: Kind::Cpu, has_fma: true, has_tensor: false, model: cpu,
            fused: "fmadd s0, s0, s1, s2", split: "fmul s0, s0, s1\nfadd s0, s0, s2" },
        Fabric { name: "cpu-riscv", kind: Kind::Cpu, has_fma: true, has_tensor: false, model: cpu,
            fused: "fmadd.s fa0, fa0, fa1, fa2", split: "fmul.s fa0, fa0, fa1\nfadd.s fa0, fa0, fa2" },
        Fabric { name: "gpu-nvidia-ptx", kind: Kind::Gpu, has_fma: true, has_tensor: true, model: gpu,
            fused: "fma.rn.f32 %f3, %f1, %f2, %f3;", split: "mul.f32 %f3, %f1, %f2;\nadd.f32 %f3, %f3, %f4;" },
        Fabric { name: "gpu-amd-cdna", kind: Kind::Gpu, has_fma: true, has_tensor: true, model: gpu,
            fused: "v_fmac_f32 v0, v1, v2", split: "v_mul_f32 v0, v1, v2\nv_add_f32 v0, v0, v3" },
        Fabric { name: "gpu-apple-metal", kind: Kind::Gpu, has_fma: true, has_tensor: false, model: gpu,
            fused: "fma(a, b, c)", split: "(a * b) + c" },
        Fabric { name: "webgpu-wgsl", kind: Kind::Gpu, has_fma: false, has_tensor: false, model: gpu,
            fused: "fma(a, b, c)", split: "(a * b) + c" },
        Fabric { name: "apple-ane", kind: Kind::Tensor, has_fma: true, has_tensor: true, model: tensor,
            fused: "ane_fma_tile(A, B, C)", split: "ane_mul_tile(A, B)\nane_add_tile(., C)" },
        Fabric { name: "intel-amx", kind: Kind::Tensor, has_fma: true, has_tensor: true, model: tensor,
            fused: "tile_fma(TMM0, TMM1, TMM2)", split: "tile_mul(TMM0, TMM1)\ntile_add(TMM0, TMM2)" },
        Fabric { name: "google-tpu", kind: Kind::Tensor, has_fma: true, has_tensor: true, model: tensor,
            fused: "mxu_fma(A, B, C)", split: "mxu_mul(A, B)\nvector_add(., C)" },
        Fabric { name: "ascend-davinci", kind: Kind::Tensor, has_fma: true, has_tensor: true, model: tensor,
            fused: "cube_fma(A, B, C)", split: "cube_mul(A, B)\nvector_add(., C)" },
        Fabric { name: "cambricon", kind: Kind::Tensor, has_fma: true, has_tensor: true, model: tensor,
            fused: "nram_fma(A, B, C)", split: "nram_mul(A, B)\nnram_add(., C)" },
        Fabric { name: "tenstorrent-tensix", kind: Kind::Dataflow, has_fma: true, has_tensor: true, model: dataflow,
            fused: "n0 = fma(a, b, c)", split: "n0 = mul(a, b)\nn1 = add(n0, c)" },
        Fabric { name: "groq-lpu", kind: Kind::Dataflow, has_fma: true, has_tensor: true, model: dataflow,
            fused: "n0 = fma(a, b, c)", split: "n0 = mul(a, b)\nn1 = add(n0, c)" },
        Fabric { name: "cerebras-wse", kind: Kind::Dataflow, has_fma: true, has_tensor: true, model: dataflow,
            fused: "pe_fma(a, b, c)", split: "pe_mul(a, b)\npe_add(., c)" },
        Fabric { name: "graphcore-ipu", kind: Kind::Dataflow, has_fma: true, has_tensor: true, model: dataflow,
            fused: "tile_fma(a, b, c)", split: "tile_mul(a, b)\ntile_add(., c)" },
        Fabric { name: "cgra", kind: Kind::Dataflow, has_fma: true, has_tensor: false, model: dataflow,
            fused: "pe[0].fma(a, b, c)", split: "pe[0].mul(a, b)\npe[1].add(., c)" },
        Fabric { name: "wasm", kind: Kind::Wasm, has_fma: false, has_tensor: false, model: wasm,
            fused: "f32.fma", split: "local.get $a\nlocal.get $b\nf32.mul\nlocal.get $c\nf32.add" },
        Fabric { name: "mlir", kind: Kind::MlIr, has_fma: true, has_tensor: false, model: cpu,
            fused: "%2 = arith.mulf %0, %1\n%3 = arith.addf %2, %c", split: "%2 = arith.mulf %0, %1\n%3 = arith.addf %2, %c" },
        Fabric { name: "stablehlo", kind: Kind::MlIr, has_fma: true, has_tensor: false, model: cpu,
            fused: "%2 = stablehlo.multiply %0, %1\n%3 = stablehlo.add %2, %c", split: "%2 = stablehlo.multiply %0, %1\n%3 = stablehlo.add %2, %c" },
        Fabric { name: "triton", kind: Kind::MlIr, has_fma: true, has_tensor: true, model: gpu,
            fused: "a * b + c", split: "a * b + c" },
        Fabric { name: "onnx", kind: Kind::MlIr, has_fma: false, has_tensor: false, model: cpu,
            fused: "Mul(a, b) -> Add(., c)", split: "Mul(a, b) -> Add(., c)" },
    ]
}

/// Lower `a*b+c` to a fabric: the fused primitive iff capability says it exists, else split.
/// Deterministic lookup — the carve-out is derived, not flagged.
pub fn lower_muladd(f: &Fabric) -> &'static str {
    if f.has_fma { f.fused } else { f.split }
}

/// Estimated joules for `a*b+c` on a fabric (fused vs split per capability).
pub fn energy_muladd(f: &Fabric) -> f64 {
    if f.has_fma {
        f.model.energy(Prim::Fma)
    } else {
        f.model.energy(Prim::Mul) + f.model.energy(Prim::Add)
    }
}

/// Heterogeneous placement: pick the fabric minimizing the derived cost — one Form, many
/// silicons, cheapest correct wins. Re-runnable as capability/cost changes (living).
pub fn place(fabrics: &[Fabric]) -> Option<&Fabric> {
    fabrics.iter().min_by(|x, y| energy_muladd(x).partial_cmp(&energy_muladd(y)).unwrap_or(std::cmp::Ordering::Equal))
}

/// Resident memory capacity per fabric class, KiB — a modelled datasheet-class prior
/// (Thor 128 GB unified vs HW3-class vs Groq SRAM). Adding a fabric = adding a row.
pub fn memory_kib(name: &str) -> u64 {
    match name {
        "cpu-x86" => 134_217_728,
        "cpu-arm64" => 16_777_216,
        "cpu-riscv" => 4_194_304,
        "gpu-nvidia-ptx" => 83_886_080,
        "gpu-amd-cdna" => 201_326_592,
        "gpu-apple-metal" => 134_217_728,
        "webgpu-wgsl" => 4_194_304,
        "apple-ane" => 16_777_216,
        "intel-amx" => 134_217_728,
        "google-tpu" => 33_554_432,
        "ascend-davinci" => 67_108_864,
        "cambricon" => 16_777_216,
        "tenstorrent-tensix" => 12_582_912,
        "groq-lpu" => 235_520,
        "cerebras-wse" => 46_137_344,
        "graphcore-ipu" => 921_600,
        "cgra" => 1_024,
        "wasm" => 4_194_304,
        _ => 16_777_216,
    }
}

/// Real capability lookup on THIS host (living optimization): which fabrics are present.
/// File/framework existence + cfg, not tribal flag knowledge. Used to place work on what
/// the machine actually has.
pub fn local_capabilities() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::new();
    if cfg!(target_arch = "aarch64") {
        v.push("cpu-arm64");
    } else {
        v.push("cpu-x86");
    }
    v.push("wasm");
    if std::path::Path::new("/System/Library/Frameworks/Metal.framework").exists() {
        v.push("gpu-apple-metal");
    }
    if std::path::Path::new("/System/Library/Frameworks/Accelerate.framework").exists() {
        v.push("apple-amx");
    }
    v
}

/// Memory-aware placement: argmin joules among fabric classes that can hold the act
/// (`need_kib`) and exist inside the device budget (`cap_kib`). The FSD-Lite inversion:
/// a memory-bound device gets the cheapest mind that FITS, not a crippled tier.
pub fn place_within(fabrics: &[Fabric], need_kib: u64, cap_kib: u64) -> Option<&Fabric> {
    fabrics
        .iter()
        .filter(|f| memory_kib(f.name) >= need_kib && memory_kib(f.name) <= cap_kib)
        .min_by(|x, y| energy_muladd(x).partial_cmp(&energy_muladd(y)).unwrap_or(std::cmp::Ordering::Equal))
}