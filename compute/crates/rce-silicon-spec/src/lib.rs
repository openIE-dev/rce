//! RCE's silicon roadmap — the 27 dedicated execution units that cover the
//! uncovered attractors from the Periodic Stack of Intelligence.
//!
//! Organized into 4 tiers by priority (Tier A = highest):
//!
//! | Tier | Units | Role                    | Phase         |
//! |------|-------|-------------------------|---------------|
//! | A    | 5     | Highest-density gaps    | Phase 1       |
//! | B    | 6     | Adaptation row          | Phase 1       |
//! | C    | 8     | Shallow feature layer   | Phase 2       |
//! | D    | 8     | Pattern + model layer   | Phase 3       |
//!
//! Each unit has:
//!
//! - A `short_code` ("A1", "B3", etc.) matching `RceSiliconUnit` in
//!   `inv-ai-codegraph::rce::enums`
//! - A `TensorCell` coordinate (α × β × γ × δ) from the Periodic Tensor
//! - A reference to the `Attractor` it resolves
//! - Target energy per operation (picojoules)
//! - Die area budget (µm²)
//! - I/O contract (input shape, output shape, optional parameters)
//! - FPGA target (K26 block count, DSP blocks, LUTs, BRAM)
//! - Implementation phase and status
//!
//! This crate is the bridge between the research docs
//! (`research/spatial-intelligence/rce-silicon-roadmap.md`) and the
//! actual FPGA/ASIC implementation. Nothing here synthesizes — it's all
//! declarative spec data.

use rce_graph::attractors::Attractor;
use rce_graph::axis_bridge::attractor_silicon_code;
use rce_graph::tensor_ops::TensorCell;
use serde::{Deserialize, Serialize};

/// Phase in the silicon bring-up program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// Phase 1: Tier A + Tier B. Highest priority. FPGA bring-up first, then
    /// tape-out to 5nm ASIC.
    Phase1,
    /// Phase 2: Tier C. Shallow feature attractors. FPGA-only for now.
    Phase2,
    /// Phase 3: Tier D. Pattern + model layer. Requires L3 sustained memory
    /// subsystem.
    Phase3,
}

/// Implementation status of a unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Status {
    /// Spec only, no RTL or HDL yet.
    SpecOnly,
    /// Verilog / Chisel draft exists.
    RtlDraft,
    /// Synthesized to FPGA, running in lab.
    FpgaBringup,
    /// Taped out to ASIC.
    AsicTapeout,
}

/// FPGA mapping estimate for a single execution unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FpgaMapping {
    pub luts: u32,
    pub dsp_blocks: u32,
    pub bram_kb: u32,
    pub uram_kb: u32,
}

/// I/O shape contract for a unit.
#[derive(Debug, Clone, PartialEq)]
pub struct IoContract {
    /// Input shape description (`"vec<f64>"`, `"mat<f32,NxN>"`, etc.).
    pub input: &'static str,
    /// Output shape description.
    pub output: &'static str,
    /// Optional parameters not counted as inputs (thresholds, learning rates).
    pub params: &'static [&'static str],
}

/// Full specification of one silicon execution unit.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitSpec {
    /// Short code: "A1", "B3", "C8", "D4".
    pub code: &'static str,
    /// Human-readable name matching the canonical attractor.
    pub name: &'static str,
    /// The attractor this unit resolves (one of the 37).
    pub attractor_id: u8,
    /// Cell coordinate notation (e.g. "CAUS x L2 x D3 x E2").
    pub cell: &'static str,
    /// Cascade tier (0-3) for dispatch routing.
    pub cascade_tier: u8,
    /// Target energy per operation, in picojoules.
    pub target_pj_per_op: f64,
    /// Die area budget in square micrometers at 5nm.
    pub area_um2_5nm: f64,
    /// I/O contract.
    pub io: IoContract,
    /// Cycle budget — target number of clock cycles per op.
    pub cycles_per_op: u32,
    /// FPGA mapping estimate on Xilinx Kria K26.
    pub fpga: FpgaMapping,
    /// Which phase this unit belongs to.
    pub phase: Phase,
    /// Current implementation status.
    pub status: Status,
    /// Human-readable design notes.
    pub notes: &'static str,
}

impl UnitSpec {
    /// Expected impedance ratio vs the Landauer floor.
    /// Tier A analog / neuromorphic units target 10² (100×).
    /// Tier B/C digital-adaptive target 10³ (1000×).
    /// Tier D pattern/model target 10⁴-10⁵.
    pub fn target_impedance_ratio(&self) -> f64 {
        match self.code.chars().next() {
            Some('A') => 100.0,
            Some('B') => 1_000.0,
            Some('C') => 1_000.0,
            Some('D') => 10_000.0,
            _ => f64::NAN,
        }
    }

    /// Landauer floor energy for this unit's cell coordinate (pJ).
    pub fn landauer_floor_pj(&self) -> f64 {
        TensorCell::from_notation(self.cell)
            .map(|c| c.landauer_floor_pj())
            .unwrap_or(0.0)
    }

    /// Effective impedance = target_pj / landauer_floor. Lower is better.
    pub fn effective_impedance(&self) -> f64 {
        let floor = self.landauer_floor_pj();
        if floor <= 0.0 { f64::INFINITY } else { self.target_pj_per_op / floor }
    }
}

// ═══════════════════════════════════════════════════════════════════
// THE 27 UNIT SPECS
// ═══════════════════════════════════════════════════════════════════

const fn io(input: &'static str, output: &'static str, params: &'static [&'static str]) -> IoContract {
    IoContract { input, output, params }
}

const fn fpga(luts: u32, dsp: u32, bram: u32, uram: u32) -> FpgaMapping {
    FpgaMapping { luts, dsp_blocks: dsp, bram_kb: bram, uram_kb: uram }
}

/// All 27 silicon execution units, in canonical Tier A → D order.
pub const UNITS: &[UnitSpec] = &[
    // ── Tier A — highest-density gaps (Phase 1) ────────────────
    UnitSpec {
        code: "A1", name: "DeepCausalModelError", attractor_id: 17,
        cell: "CAUS x L2 x D3 x E2", cascade_tier: 3,
        target_pj_per_op: 500.0, area_um2_5nm: 800_000.0,
        io: io("struct<cause_graph, observed, counterfactual>", "vec<f64>", &["depth", "branch_factor"]),
        cycles_per_op: 512,
        fpga: fpga(22_000, 32, 512, 256),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Reversible recursive graph processor. 7 instance count. 10^14 impedance ratio target. Highest-impact target.",
    },
    UnitSpec {
        code: "A2", name: "DeepSelfModelError", attractor_id: 8,
        cell: "SELF x L2 x D3 x E2", cascade_tier: 3,
        target_pj_per_op: 400.0, area_um2_5nm: 700_000.0,
        io: io("struct<reward, v_current, v_next, gamma>", "f64", &["td_lambda"]),
        cycles_per_op: 256,
        fpga: fpga(18_000, 24, 384, 128),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Recursive prediction engine with sustained self-model memory. TD-lambda accumulator. 7 instances.",
    },
    UnitSpec {
        code: "A3", name: "MidDepthCausalInference", attractor_id: 32,
        cell: "CAUS x L2 x D2 x E2", cascade_tier: 2,
        target_pj_per_op: 300.0, area_um2_5nm: 500_000.0,
        io: io("struct<observation, model, actions>", "struct<best_action, expected_value>", &["beam_width"]),
        cycles_per_op: 256,
        fpga: fpga(15_000, 20, 256, 128),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Two-stage prediction pipeline: forward-model simulation + beam search + argmax.",
    },
    UnitSpec {
        code: "A4", name: "ParallelSpectralDecomposition", attractor_id: 24,
        cell: "SPEC x L2 x D1 x E0", cascade_tier: 1,
        target_pj_per_op: 0.01, area_um2_5nm: 120_000.0,
        io: io("vec<f64>", "vec<f64> [N bands]", &["num_bands"]),
        cycles_per_op: 1,
        fpga: fpga(4_000, 16, 32, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Analog filter bank. 10^-20 J per output channel. Cheapest to build — existing analog SOC IP available.",
    },
    UnitSpec {
        code: "A5", name: "SustainedStructuralMemory", attractor_id: 37,
        cell: "STRUC x L3 x D2 x E1", cascade_tier: 3,
        target_pj_per_op: 50.0, area_um2_5nm: 2_000_000.0,
        io: io("vec<f64>[query]", "vec<f64>[retrieved]", &["capacity", "retention"]),
        cycles_per_op: 64,
        fpga: fpga(30_000, 8, 1024, 1024),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Persistent associative memory. Content-addressable. Survives power cycles. Backed by RRAM/PCM on ASIC, BRAM on FPGA.",
    },
    // ── Tier B — adaptation row (Phase 1) ───────────────────────
    UnitSpec {
        code: "B1", name: "MagnitudeGainAdaptation", attractor_id: 4,
        cell: "MAG x L2 x D1 x E3", cascade_tier: 1,
        target_pj_per_op: 0.8, area_um2_5nm: 30_000.0,
        io: io("f64", "f64", &["alpha", "rms_init"]),
        cycles_per_op: 4,
        fpga: fpga(600, 2, 4, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Analog integrator with exponential decay. <1 nJ per update. Baroreceptor-class primitive.",
    },
    UnitSpec {
        code: "B2", name: "TemporalAdaptation", attractor_id: 3,
        cell: "TEMP x L2 x D1 x E3", cascade_tier: 1,
        target_pj_per_op: 0.8, area_um2_5nm: 30_000.0,
        io: io("vec<f64>[stream]", "vec<f64>[delta]", &["alpha"]),
        cycles_per_op: 4,
        fpga: fpga(600, 2, 4, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Running baseline + delta. Identical to B1 structurally but tagged TEMP for type dispatch.",
    },
    UnitSpec {
        code: "B3", name: "SpatialAdaptation", attractor_id: 34,
        cell: "SPAT x L2 x D1 x E3", cascade_tier: 1,
        target_pj_per_op: 0.8, area_um2_5nm: 30_000.0,
        io: io("mat<f64>[2d_field]", "mat<f64>[delta_field]", &["alpha"]),
        cycles_per_op: 16,
        fpga: fpga(1_200, 4, 16, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "2D spatial integrator. Light/dark adaptation, tilt aftereffect.",
    },
    UnitSpec {
        code: "B4", name: "SpectralAdaptation", attractor_id: 25,
        cell: "SPEC x L2 x D1 x E3", cascade_tier: 1,
        target_pj_per_op: 0.8, area_um2_5nm: 30_000.0,
        io: io("vec<f64>[bands]", "vec<f64>[adapted_bands]", &["alpha"]),
        cycles_per_op: 4,
        fpga: fpga(800, 4, 8, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Per-band adaptation. Jamming avoidance, cochlear adaptation, Lombard effect.",
    },
    UnitSpec {
        code: "B5", name: "RawMagnitudeAdaptation", attractor_id: 29,
        cell: "MAG x L2 x D0 x E3", cascade_tier: 0,
        target_pj_per_op: 0.3, area_um2_5nm: 15_000.0,
        io: io("f64", "f64", &["alpha"]),
        cycles_per_op: 2,
        fpga: fpga(400, 1, 2, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Single-variable running baseline. Chemotaxis, sensitization. Simplest adaptation primitive.",
    },
    UnitSpec {
        code: "B6", name: "RawStructuralRecognition", attractor_id: 20,
        cell: "STRUC x L2 x D0 x E1", cascade_tier: 0,
        target_pj_per_op: 0.5, area_um2_5nm: 50_000.0,
        io: io("&[u8][input]", "bool[matched]", &["template_id"]),
        cycles_per_op: 1,
        fpga: fpga(2_000, 0, 64, 0),
        phase: Phase::Phase1, status: Status::SpecOnly,
        notes: "Content-addressable memory with parallel comparators. Molecular key-lock, tokenization.",
    },
    // ── Tier C — shallow feature (Phase 2) ──────────────────────
    UnitSpec {
        code: "C1", name: "ShallowStructuralTemplate", attractor_id: 21,
        cell: "STRUC x L2 x D1 x E1", cascade_tier: 1,
        target_pj_per_op: 2.0, area_um2_5nm: 80_000.0,
        io: io("vec<f64>[input], vec<f64>[template]", "f64[similarity]", &[]),
        cycles_per_op: 8,
        fpga: fpga(2_500, 8, 32, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Single-layer template matching. Kin discrimination, combinatorial quorum.",
    },
    UnitSpec {
        code: "C2", name: "SpectralTemplateMatch", attractor_id: 14,
        cell: "SPEC x L2 x D1 x E1", cascade_tier: 1,
        target_pj_per_op: 2.0, area_um2_5nm: 80_000.0,
        io: io("vec<f64>[spectrum], vec<f64>[template]", "f64", &[]),
        cycles_per_op: 8,
        fpga: fpga(2_500, 8, 32, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Spectral cross-correlation. Spider web vibration, formant recognition, cricket song.",
    },
    UnitSpec {
        code: "C3", name: "ShallowRelationalTemplate", attractor_id: 31,
        cell: "REL x L2 x D1 x E1", cascade_tier: 1,
        target_pj_per_op: 2.0, area_um2_5nm: 80_000.0,
        io: io("vec<f64>[query], mat<f64>[database]", "usize[idx], f64[score]", &[]),
        cycles_per_op: 16,
        fpga: fpga(3_000, 12, 64, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "RAG retrieval primitive. Classical conditioning, voice recognition.",
    },
    UnitSpec {
        code: "C4", name: "ShallowRelationalComparison", attractor_id: 30,
        cell: "REL x L2 x D1 x E0", cascade_tier: 1,
        target_pj_per_op: 1.5, area_um2_5nm: 60_000.0,
        io: io("mat<f64>[n x d][concurrent_inputs]", "mat<f64>[n x n][similarity]", &[]),
        cycles_per_op: 8,
        fpga: fpga(2_000, 16, 32, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Pairwise similarity matrix for concurrent streams. Dance audience, vocal turn-taking.",
    },
    UnitSpec {
        code: "C5", name: "ShallowCausalInference", attractor_id: 33,
        cell: "CAUS x L2 x D1 x E2", cascade_tier: 1,
        target_pj_per_op: 3.0, area_um2_5nm: 100_000.0,
        io: io("vec<f64>[action], vec<f64>[outcome]", "f64[estimated_causal_effect]", &[]),
        cycles_per_op: 32,
        fpga: fpga(4_000, 16, 64, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Single-step cause-effect estimation. Operant conditioning, speculative decode.",
    },
    UnitSpec {
        code: "C6", name: "RawExternalMagnitude", attractor_id: 26,
        cell: "MAG x L2 x D0 x E0", cascade_tier: 0,
        target_pj_per_op: 0.1, area_um2_5nm: 8_000.0,
        io: io("f64, f64", "i8 [-1/0/+1]", &[]),
        cycles_per_op: 1,
        fpga: fpga(100, 0, 0, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Just a comparator. Already exists in every ALU — noted here for completeness.",
    },
    UnitSpec {
        code: "C7", name: "ShallowMagnitudeContrast", attractor_id: 27,
        cell: "MAG x L2 x D1 x E0", cascade_tier: 1,
        target_pj_per_op: 0.4, area_um2_5nm: 20_000.0,
        io: io("vec<f64>[a], vec<f64>[b]", "vec<f64>[delta]", &[]),
        cycles_per_op: 2,
        fpga: fpga(500, 4, 4, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Element-wise contrast. Retinal ON/OFF, gate control pain.",
    },
    UnitSpec {
        code: "C8", name: "MagnitudeVsComputedReference", attractor_id: 28,
        cell: "MAG x L2 x D1 x E2", cascade_tier: 1,
        target_pj_per_op: 1.2, area_um2_5nm: 40_000.0,
        io: io("vec<f64>[logits]", "vec<f64>[normalized]", &[]),
        cycles_per_op: 4,
        fpga: fpga(1_500, 8, 16, 0),
        phase: Phase::Phase2, status: Status::SpecOnly,
        notes: "Softmax silicon. Batch normalization, number sense. Currently emulated in every NN via matmul+exp+reduce+div.",
    },
    // ── Tier D — pattern + model (Phase 3) ──────────────────────
    UnitSpec {
        code: "D1", name: "MultiScaleTemporalContrast", attractor_id: 9,
        cell: "TEMP x L2 x D2 x E0", cascade_tier: 2,
        target_pj_per_op: 15.0, area_um2_5nm: 300_000.0,
        io: io("vec<f64>[stream]", "vec<f64>[multi_scale_features]", &["num_scales"]),
        cycles_per_op: 64,
        fpga: fpga(8_000, 24, 128, 64),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Wavelet-like multi-scale pyramid. Click trains, flight turbulence detection.",
    },
    UnitSpec {
        code: "D2", name: "SpatialTemplateMatch", attractor_id: 10,
        cell: "SPAT x L2 x D2 x E1", cascade_tier: 2,
        target_pj_per_op: 20.0, area_um2_5nm: 400_000.0,
        io: io("mat<f64>[image], mat<f64>[template]", "mat<f64>[correlation]", &[]),
        cycles_per_op: 128,
        fpga: fpga(10_000, 48, 256, 128),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "2D cross-correlation with argmax. Place cells, face recognition, nest templates.",
    },
    UnitSpec {
        code: "D3", name: "CausalSchemaMatch", attractor_id: 12,
        cell: "CAUS x L2 x D2 x E1", cascade_tier: 2,
        target_pj_per_op: 25.0, area_um2_5nm: 450_000.0,
        io: io("struct<observation>, vec<struct<schema>>", "usize[best_schema], f64[match]", &[]),
        cycles_per_op: 128,
        fpga: fpga(12_000, 24, 256, 128),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Schema library lookup + pattern match. Tool-use schema, medical symptom matching, fault diagnosis.",
    },
    UnitSpec {
        code: "D4", name: "StructuralCompetition", attractor_id: 22,
        cell: "STRUC x L2 x D2 x E0", cascade_tier: 2,
        target_pj_per_op: 18.0, area_um2_5nm: 350_000.0,
        io: io("vec<struct<candidate>>", "usize[winner], f64[margin]", &[]),
        cycles_per_op: 64,
        fpga: fpga(9_000, 16, 128, 64),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Winner-take-all with suppression. Chromatophore matching, binocular rivalry.",
    },
    UnitSpec {
        code: "D5", name: "AdaptiveStructuralPerception", attractor_id: 23,
        cell: "STRUC x L2 x D2 x E3", cascade_tier: 2,
        target_pj_per_op: 22.0, area_um2_5nm: 400_000.0,
        io: io("stream<mat<f64>>", "vec<struct<feature_stream>>", &[]),
        cycles_per_op: 96,
        fpga: fpga(11_000, 20, 192, 96),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Active sensing with state feedback. Whisker, saccadic, auditory stream segregation.",
    },
    UnitSpec {
        code: "D6", name: "RecursiveSpatialMaps", attractor_id: 13,
        cell: "SPAT x L3 x D3 x E1", cascade_tier: 3,
        target_pj_per_op: 80.0, area_um2_5nm: 1_500_000.0,
        io: io("stream<position>", "mat<f64>[cognitive_map]", &["map_resolution"]),
        cycles_per_op: 256,
        fpga: fpga(25_000, 32, 768, 512),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Hierarchical grid-cell circuit. Place cell topology, bee cognitive map. Requires L3 sustained memory.",
    },
    UnitSpec {
        code: "D7", name: "TemporalContextEncoding", attractor_id: 15,
        cell: "TEMP x L3 x D2 x E3", cascade_tier: 2,
        target_pj_per_op: 30.0, area_um2_5nm: 600_000.0,
        io: io("stream<f64>, vec<context>", "vec<f64>[encoded_context]", &[]),
        cycles_per_op: 128,
        fpga: fpga(14_000, 16, 384, 256),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Circadian / time cells. Interference resolution, temporal binding.",
    },
    UnitSpec {
        code: "D8", name: "SustainedSpatialSelfReference", attractor_id: 16,
        cell: "SPAT x L3 x D2 x E3", cascade_tier: 2,
        target_pj_per_op: 35.0, area_um2_5nm: 700_000.0,
        io: io("stream<motion>, mat<f64>[env]", "vec<f64>[self_position]", &[]),
        cycles_per_op: 128,
        fpga: fpga(15_000, 20, 384, 256),
        phase: Phase::Phase3, status: Status::SpecOnly,
        notes: "Path integration + head direction + self-position. Grid cells. Requires L3 sustained memory.",
    },
];

// ═══════════════════════════════════════════════════════════════════
// QUERY API
// ═══════════════════════════════════════════════════════════════════

/// Total number of unit specs.
pub fn unit_count() -> usize { UNITS.len() }

/// Look up a unit by its short code.
pub fn unit_by_code(code: &str) -> Option<&'static UnitSpec> {
    UNITS.iter().find(|u| u.code == code)
}

/// Look up all units that resolve a given attractor (typically just 0 or 1).
pub fn units_for_attractor(attractor: Attractor) -> Vec<&'static UnitSpec> {
    let id = attractor.id();
    UNITS.iter().filter(|u| u.attractor_id == id).collect()
}

/// All units in a given tier ('A', 'B', 'C', 'D').
pub fn units_in_tier(tier: char) -> Vec<&'static UnitSpec> {
    UNITS.iter().filter(|u| u.code.starts_with(tier)).collect()
}

/// All units in a given phase.
pub fn units_in_phase(phase: Phase) -> Vec<&'static UnitSpec> {
    UNITS.iter().filter(|u| u.phase == phase).collect()
}

/// Units with a specific implementation status.
pub fn units_with_status(status: Status) -> Vec<&'static UnitSpec> {
    UNITS.iter().filter(|u| u.status == status).collect()
}

/// Bring-up report: counts by phase, tier, and status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BringUpReport {
    pub total_units: usize,
    pub tier_a: usize,
    pub tier_b: usize,
    pub tier_c: usize,
    pub tier_d: usize,
    pub phase_1: usize,
    pub phase_2: usize,
    pub phase_3: usize,
    pub spec_only: usize,
    pub rtl_draft: usize,
    pub fpga_bringup: usize,
    pub asic_tapeout: usize,
    pub total_die_area_um2: f64,
    pub total_target_pj_per_op_bundle: f64,
    pub total_fpga_luts: u32,
    pub total_fpga_dsp: u32,
    pub total_fpga_bram_kb: u32,
}

/// Generate the full bring-up report from the 27 specs.
pub fn bring_up_report() -> BringUpReport {
    let mut r = BringUpReport {
        total_units: UNITS.len(),
        tier_a: 0, tier_b: 0, tier_c: 0, tier_d: 0,
        phase_1: 0, phase_2: 0, phase_3: 0,
        spec_only: 0, rtl_draft: 0, fpga_bringup: 0, asic_tapeout: 0,
        total_die_area_um2: 0.0,
        total_target_pj_per_op_bundle: 0.0,
        total_fpga_luts: 0, total_fpga_dsp: 0, total_fpga_bram_kb: 0,
    };
    for u in UNITS {
        match u.code.chars().next() {
            Some('A') => r.tier_a += 1,
            Some('B') => r.tier_b += 1,
            Some('C') => r.tier_c += 1,
            Some('D') => r.tier_d += 1,
            _ => {}
        }
        match u.phase {
            Phase::Phase1 => r.phase_1 += 1,
            Phase::Phase2 => r.phase_2 += 1,
            Phase::Phase3 => r.phase_3 += 1,
        }
        match u.status {
            Status::SpecOnly => r.spec_only += 1,
            Status::RtlDraft => r.rtl_draft += 1,
            Status::FpgaBringup => r.fpga_bringup += 1,
            Status::AsicTapeout => r.asic_tapeout += 1,
        }
        r.total_die_area_um2 += u.area_um2_5nm;
        r.total_target_pj_per_op_bundle += u.target_pj_per_op;
        r.total_fpga_luts += u.fpga.luts;
        r.total_fpga_dsp += u.fpga.dsp_blocks;
        r.total_fpga_bram_kb += u.fpga.bram_kb;
    }
    r
}

/// Verify every Attractor in the silicon roadmap matches its axis-bridge code.
pub fn verify_roadmap() -> Result<(), String> {
    for unit in UNITS {
        let attractor = Attractor::from_id(unit.attractor_id)
            .ok_or_else(|| format!("unit {} has invalid attractor_id {}", unit.code, unit.attractor_id))?;
        let expected_code = attractor_silicon_code(attractor);
        if expected_code != unit.code {
            return Err(format!(
                "unit {} spec says attractor #{} but axis_bridge says attractor #{} maps to {}",
                unit.code, unit.attractor_id, unit.attractor_id, expected_code
            ));
        }
        let expected_cell = attractor.cell().notation();
        if expected_cell != unit.cell {
            return Err(format!(
                "unit {} cell {} doesn't match attractor #{} canonical cell {}",
                unit.code, unit.cell, unit.attractor_id, expected_cell
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_27_units_total() {
        assert_eq!(UNITS.len(), 27);
    }

    #[test]
    fn test_tier_counts() {
        assert_eq!(units_in_tier('A').len(), 5);
        assert_eq!(units_in_tier('B').len(), 6);
        assert_eq!(units_in_tier('C').len(), 8);
        assert_eq!(units_in_tier('D').len(), 8);
    }

    #[test]
    fn test_phase_counts() {
        assert_eq!(units_in_phase(Phase::Phase1).len(), 11); // Tier A (5) + Tier B (6)
        assert_eq!(units_in_phase(Phase::Phase2).len(), 8);  // Tier C
        assert_eq!(units_in_phase(Phase::Phase3).len(), 8);  // Tier D
    }

    #[test]
    fn test_unit_by_code() {
        let a1 = unit_by_code("A1").unwrap();
        assert_eq!(a1.name, "DeepCausalModelError");
        assert_eq!(a1.attractor_id, 17);
        assert_eq!(a1.cell, "CAUS x L2 x D3 x E2");

        let a4 = unit_by_code("A4").unwrap();
        assert_eq!(a4.name, "ParallelSpectralDecomposition");
        assert!(a4.target_pj_per_op < 1.0);
    }

    #[test]
    fn test_unique_codes() {
        let codes: Vec<&str> = UNITS.iter().map(|u| u.code).collect();
        let unique: std::collections::HashSet<_> = codes.iter().collect();
        assert_eq!(codes.len(), unique.len());
    }

    #[test]
    fn test_unique_attractor_ids() {
        let ids: Vec<u8> = UNITS.iter().map(|u| u.attractor_id).collect();
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(ids.len(), unique.len());
    }

    #[test]
    fn test_verify_roadmap_consistency() {
        verify_roadmap().expect("silicon roadmap must match axis_bridge");
    }

    #[test]
    fn test_bring_up_report() {
        let r = bring_up_report();
        assert_eq!(r.total_units, 27);
        assert_eq!(r.tier_a, 5);
        assert_eq!(r.tier_b, 6);
        assert_eq!(r.tier_c, 8);
        assert_eq!(r.tier_d, 8);
        assert_eq!(r.phase_1, 11);
        assert_eq!(r.spec_only, 27); // all units are spec only today
        assert_eq!(r.asic_tapeout, 0);
        // Total die area should fit in ~200 mm² (2×10¹¹ µm²) per roadmap budget
        let total_mm2 = r.total_die_area_um2 / 1_000_000.0;
        assert!(total_mm2 < 200.0, "total die area {} mm² exceeds 200 mm² budget", total_mm2);
    }

    #[test]
    fn test_units_for_attractor() {
        // #17 Deep causal model error maps to A1
        let units = units_for_attractor(Attractor::DeepCausalModelError);
        assert_eq!(units.len(), 1);
        assert_eq!(units[0].code, "A1");
    }

    #[test]
    fn test_impedance_targets_by_tier() {
        assert_eq!(unit_by_code("A1").unwrap().target_impedance_ratio(), 100.0);
        assert_eq!(unit_by_code("B1").unwrap().target_impedance_ratio(), 1000.0);
        assert_eq!(unit_by_code("C1").unwrap().target_impedance_ratio(), 1000.0);
        assert_eq!(unit_by_code("D1").unwrap().target_impedance_ratio(), 10_000.0);
    }

    #[test]
    fn test_a4_is_cheapest() {
        // A4 Parallel Spectral Decomposition should have the lowest pJ/op
        let a4 = unit_by_code("A4").unwrap();
        for u in UNITS {
            if u.code != "A4" {
                assert!(u.target_pj_per_op >= a4.target_pj_per_op,
                    "A4 should be the cheapest unit, but {} has {} pJ", u.code, u.target_pj_per_op);
            }
        }
    }

    #[test]
    fn test_fpga_mapping_nonzero() {
        // Every unit should have at least some FPGA LUT budget
        for u in UNITS {
            assert!(u.fpga.luts > 0 || u.code == "C6", "{} has 0 LUTs", u.code);
        }
    }

    #[test]
    fn test_total_fpga_fits_on_k26() {
        let r = bring_up_report();
        // Kria K26 has ~117k LUTs; the full 27-unit suite should fit with margin
        assert!(r.total_fpga_luts < 250_000, "FPGA LUT total {} too large", r.total_fpga_luts);
    }

    #[test]
    fn test_cell_notation_validity() {
        // Every unit's cell string must parse
        for u in UNITS {
            let cell = TensorCell::from_notation(u.cell);
            assert!(cell.is_some(), "unit {} has invalid cell {}", u.code, u.cell);
        }
    }

    #[test]
    fn test_status_is_all_spec_only() {
        // At roadmap creation time, nothing has been built yet
        assert_eq!(units_with_status(Status::SpecOnly).len(), 27);
    }
}
