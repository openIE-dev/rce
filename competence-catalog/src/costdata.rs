//! Frontier 1 — calibrated per-instruction cost/energy (uops.info / Agner Fog / Horowitz).
//!
//! The living-optimization cost model was a modelled Horowitz prior. The obscure-but-knowable
//! calibration sources are public: **Horowitz ISSCC 2014** (pJ per op; data movement dominates),
//! **Agner Fog instruction_tables** and **uops.info** (measured latency/throughput/µops per x86
//! microarchitecture). This module encodes that prior as a per-primitive table and derives
//! placement cost models from it. Values are an order-of-magnitude **est** prior (Horowitz 45nm +
//! Agner relative latencies), labelled as such; a `measured_j` calibration replaces it when
//! available. The point: `b ≫ a` (moving a byte costs far more than a flop), and a fused primitive
//! beats mul+add where the fabric has it.

use crate::placement::{CostModel, Prim};

/// A per-instruction cost entry: latency (cycles), µops, and energy (pJ).
pub struct InstrCost {
    pub prim: Prim,
    pub latency_cycles: u32,
    pub uops: u32,
    pub energy_pj: f64,
}

/// Horowitz-form energy prior + Agner/uops relative latency (est, 45nm order-of-magnitude).
pub fn horowitz_table() -> Vec<InstrCost> {
    vec![
        InstrCost { prim: Prim::Add, latency_cycles: 1, uops: 1, energy_pj: 0.1 },
        InstrCost { prim: Prim::Mul, latency_cycles: 3, uops: 1, energy_pj: 3.1 },
        InstrCost { prim: Prim::Fma, latency_cycles: 4, uops: 1, energy_pj: 2.5 },
        InstrCost { prim: Prim::Load, latency_cycles: 4, uops: 1, energy_pj: 5.0 },
        InstrCost { prim: Prim::Store, latency_cycles: 4, uops: 1, energy_pj: 5.0 },
    ]
}

/// An 8-byte DRAM access costs ~640 pJ (Horowitz) — the `b ≫ a` fact that makes data movement the
/// dominant term and locality the first optimization.
pub fn dram_energy_pj() -> f64 {
    640.0
}

/// Energy of one primitive from the table.
pub fn instruction_energy(p: Prim) -> f64 {
    horowitz_table().iter().find(|c| c.prim == p).map(|c| c.energy_pj).unwrap_or(0.0)
}

/// The calibrated saving from fusing mul+add into one FMA (energy prior).
pub fn fma_saving_calibrated() -> f64 {
    instruction_energy(Prim::Mul) + instruction_energy(Prim::Add) - instruction_energy(Prim::Fma)
}

/// Derive placement cost models per fabric family from the calibrated prior.
/// `a` = energy/flop, `b` = energy/byte (DRAM-dominated), `c` = dispatch.
pub fn cpu_model() -> CostModel {
    CostModel { a: 0.1, b: dram_energy_pj() / 8.0, c: 0.5 }
}
pub fn gpu_model() -> CostModel {
    // Higher flop throughput, cheaper on-die movement, but host transfer still DRAM-bound.
    CostModel { a: 0.02, b: 20.0, c: 0.3 }
}
pub fn tensor_model() -> CostModel {
    // Matrix units amortize flops; movement within the tile is cheap.
    CostModel { a: 0.005, b: 2.0, c: 0.2 }
}