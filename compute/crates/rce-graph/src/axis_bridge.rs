//! Bridge between `rce_graph::tensor_ops` axis types and the canonical
//! types of the axis enums of the external inv-ai-codegraph crate.
//!
//! Both crates represent the same Periodic Tensor axes but with independent
//! enum definitions because they live in different workspace crates with
//! no compile-time dependency on each other. This module provides
//! conversion traits + a dispatch-ready `AxisCoordinate` type that carries
//! enough information to route any attractor call through either system.
//!
//! # Why not merge the two?
//!
//! `inv-ai-codegraph` is a ~13K LOC crate with thousands of tests built on
//! its own local enums. `rce-graph` is the JMax runtime. Forcing either to
//! depend on the other would entangle them. Instead we keep both and
//! bridge at the type boundary: the language side uses `TensorCell`, the
//! codegraph/silicon side uses `AttractorOp + RceSiliconUnit`, and this
//! bridge maps between them losslessly.
//!
//! # Correspondence
//!
//! | rce_graph::tensor_ops  | inv-ai-codegraph axis enums     |
//! |------------------------|--------------------------------|
//! | `ContrastType`         | implicit via `AttractorOp` α tag |
//! | `EnergyCost`           | `EnergyClass` (if present)        |
//! | `IntegrationDepth`     | `CascadeTier` (γ → tier promotion) |
//! | `ReferenceType` E0..E3 | `ReferenceType` External/Stored/Computed/SelfState |
//! | `TensorCell`           | (cell, tier, silicon_unit) triple  |
//!
//! # What this module is NOT
//!
//! This is not the MCP server, not the interpreter, and not a full-cell
//! database. It is a pure translation layer. Call it from either side to
//! bridge without coupling.

use crate::tensor_ops::{
    ContrastType, EnergyCost, IntegrationDepth, ReferenceType, TensorCell,
};

/// An axis-agnostic coordinate usable from either side of the bridge.
///
/// Unlike `TensorCell` (which lives purely in `rce-graph`) or
/// `(AttractorOp, CascadeTier, RceSiliconUnit)` (which lives purely in
/// `inv-ai-codegraph`), this carries:
///
/// - The 4 canonical axis values as short codes
/// - The attractor ID (#1-#37) if this coordinate resolves to a known attractor
/// - The cascade tier (0-3)
/// - The silicon unit tag as a short code ("A1", "B3", "C8", "D4", or "fallback")
///
/// This is the over-the-wire format that the MCP server, the interpreter,
/// and the graph nodes all share.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AxisCoordinate {
    pub alpha: &'static str,
    pub beta: &'static str,
    pub gamma: &'static str,
    pub delta: &'static str,
    pub attractor_id: Option<u8>,
    pub cascade_tier: u8,
    pub silicon_unit: &'static str,
}

impl AxisCoordinate {
    /// Notation form: `"MAG x L2 x D0 x E1"`.
    pub fn notation(&self) -> String {
        format!("{} x {} x {} x {}", self.alpha, self.beta, self.gamma, self.delta)
    }

    /// Whether this coordinate has an assigned silicon unit (not fallback).
    pub fn has_silicon(&self) -> bool {
        !self.silicon_unit.is_empty() && self.silicon_unit != "fallback"
    }
}

/// Convert a `TensorCell` into an `AxisCoordinate`, filling in the
/// attractor/silicon lookup from the known attractor table.
impl From<TensorCell> for AxisCoordinate {
    fn from(cell: TensorCell) -> Self {
        // Look up the attractor that owns this cell (if any)
        let attractor = crate::attractors::Attractor::all()
            .into_iter()
            .find(|a| a.cell() == cell);
        let (attractor_id, silicon_unit, cascade_tier) = match attractor {
            Some(a) => (
                Some(a.id()),
                attractor_silicon_code(a),
                a.cascade_tier(),
            ),
            None => (None, "fallback", cell.cascade_tier()),
        };
        Self {
            alpha: cell.alpha.short_code(),
            beta: cell.beta.short_code(),
            gamma: cell.gamma.short_code(),
            delta: cell.delta.short_code(),
            attractor_id,
            cascade_tier,
            silicon_unit,
        }
    }
}

/// Try to parse an `AxisCoordinate` from a cell notation string.
/// Returns `None` if the notation is malformed.
pub fn coordinate_from_notation(s: &str) -> Option<AxisCoordinate> {
    TensorCell::from_notation(s).map(AxisCoordinate::from)
}

/// Map an attractor to its silicon unit code from the rce roadmap.
///
/// These codes match the silicon-unit enum of the inv-ai-codegraph crate.
/// Tier A (5 units), Tier B (6 units), Tier C (8 units), Tier D (8 units) = 27 total.
pub fn attractor_silicon_code(a: crate::attractors::Attractor) -> &'static str {
    use crate::attractors::Attractor::*;
    match a {
        // Tier A — highest priority (5 units)
        DeepCausalModelError => "A1",
        DeepSelfModelError => "A2",
        MidDepthCausalInference => "A3",
        ParallelSpectralDecomposition => "A4",
        SustainedStructuralMemory => "A5",

        // Tier B — adaptation row (6 units)
        MagnitudeGainAdaptation => "B1",
        TemporalAdaptation => "B2",
        SpatialAdaptation => "B3",
        SpectralAdaptation => "B4",
        RawMagnitudeAdaptation => "B5",
        RawStructuralRecognition => "B6",

        // Tier C — shallow feature (8 units)
        ShallowStructuralTemplate => "C1",
        SpectralTemplateMatch => "C2",
        ShallowRelationalTemplate => "C3",
        ShallowRelationalComparison => "C4",
        ShallowCausalInference => "C5",
        RawExternalMagnitude => "C6",
        ShallowMagnitudeContrast => "C7",
        MagnitudeVsComputedReference => "C8",

        // Tier D — pattern + model (8 units)
        MultiScaleTemporalContrast => "D1",
        SpatialTemplateMatch => "D2",
        CausalSchemaMatch => "D3",
        StructuralCompetition => "D4",
        AdaptiveStructuralPerception => "D5",
        RecursiveSpatialMaps => "D6",
        TemporalContextEncoding => "D7",
        SustainedSpatialSelfReference => "D8",

        // Attractors that lower to subgraphs of existing operations
        // (threshold crossing, spatial gradient, deep structural template,
        // competitive comparison, prediction error, deep self model error,
        // deep relational template, metacognition, long-range predictive temporal,
        // deep temporal memory, relational prediction error) — these are either
        // already covered by generic matmul/softmax silicon or pending roadmap assignment.
        ThresholdCrossing
        | SpatialGradient
        | DeepStructuralTemplate
        | CompetitiveComparison
        | ForwardModelPredictionError
        | DeepRelationalTemplate
        | Metacognition
        | LongRangePredictiveTemporal
        | DeepTemporalMemory
        | RelationalPredictionError => "fallback",
    }
}

/// Convert an axis short code ("MAG", "L2", "D1", "E0") back to its
/// strongly-typed equivalent. Returns `None` if the code is malformed.
pub fn contrast_from_code(s: &str) -> Option<ContrastType> { ContrastType::from_short_code(s) }
pub fn energy_from_code(s: &str) -> Option<EnergyCost> { EnergyCost::from_short_code(s) }
pub fn depth_from_code(s: &str) -> Option<IntegrationDepth> { IntegrationDepth::from_short_code(s) }
pub fn reference_from_code(s: &str) -> Option<ReferenceType> { ReferenceType::from_short_code(s) }

/// Build a TensorCell from four short codes (MAG/L2/D0/E1 style).
pub fn build_cell(
    alpha: &str,
    beta: &str,
    gamma: &str,
    delta: &str,
) -> Option<TensorCell> {
    Some(TensorCell::new(
        contrast_from_code(alpha)?,
        energy_from_code(beta)?,
        depth_from_code(gamma)?,
        reference_from_code(delta)?,
    ))
}

// ═══════════════════════════════════════════════════════════════════
// COVERAGE SUMMARY (bridge-level)
// ═══════════════════════════════════════════════════════════════════

/// Count how many of the 27 silicon units have an attractor assigned.
///
/// Returns `(tier_a, tier_b, tier_c, tier_d)` — expected to be (5, 6, 8, 8).
pub fn silicon_tier_counts() -> (usize, usize, usize, usize) {
    let mut a = 0;
    let mut b = 0;
    let mut c = 0;
    let mut d = 0;
    for attractor in crate::attractors::Attractor::all() {
        let code = attractor_silicon_code(attractor);
        match code.chars().next() {
            Some('A') => a += 1,
            Some('B') => b += 1,
            Some('C') => c += 1,
            Some('D') => d += 1,
            _ => {}
        }
    }
    (a, b, c, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attractors::Attractor;

    #[test]
    fn test_tensor_cell_to_axis_coordinate() {
        let cell = TensorCell::from_notation("STRUC x L2 x D2 x E1").unwrap();
        let coord: AxisCoordinate = cell.into();
        assert_eq!(coord.alpha, "STRUC");
        assert_eq!(coord.gamma, "D2");
        assert_eq!(coord.attractor_id, Some(5)); // deep_structural_template
        assert_eq!(coord.cascade_tier, 2);
    }

    #[test]
    fn test_axis_coordinate_notation() {
        let coord = AxisCoordinate::from(
            TensorCell::from_notation("MAG x L2 x D0 x E1").unwrap()
        );
        assert_eq!(coord.notation(), "MAG x L2 x D0 x E1");
    }

    #[test]
    fn test_coordinate_from_notation() {
        let coord = coordinate_from_notation("SPAT x L2 x D1 x E0").unwrap();
        assert_eq!(coord.attractor_id, Some(2)); // spatial_gradient
        assert_eq!(coord.alpha, "SPAT");
    }

    #[test]
    fn test_unknown_cell_yields_no_attractor() {
        // A coordinate with no matching attractor
        let cell = TensorCell::from_notation("SPEC x L0 x D0 x E0").unwrap();
        let coord: AxisCoordinate = cell.into();
        assert_eq!(coord.attractor_id, None);
        assert_eq!(coord.silicon_unit, "fallback");
    }

    #[test]
    fn test_silicon_code_tier_a() {
        assert_eq!(attractor_silicon_code(Attractor::DeepCausalModelError), "A1");
        assert_eq!(attractor_silicon_code(Attractor::DeepSelfModelError), "A2");
        assert_eq!(attractor_silicon_code(Attractor::SustainedStructuralMemory), "A5");
    }

    #[test]
    fn test_silicon_code_tier_b() {
        assert_eq!(attractor_silicon_code(Attractor::MagnitudeGainAdaptation), "B1");
        assert_eq!(attractor_silicon_code(Attractor::RawStructuralRecognition), "B6");
    }

    #[test]
    fn test_silicon_code_tier_c() {
        assert_eq!(attractor_silicon_code(Attractor::ShallowStructuralTemplate), "C1");
        assert_eq!(attractor_silicon_code(Attractor::MagnitudeVsComputedReference), "C8");
    }

    #[test]
    fn test_silicon_code_tier_d() {
        assert_eq!(attractor_silicon_code(Attractor::MultiScaleTemporalContrast), "D1");
        assert_eq!(attractor_silicon_code(Attractor::SustainedSpatialSelfReference), "D8");
    }

    #[test]
    fn test_silicon_tier_counts() {
        let (a, b, c, d) = silicon_tier_counts();
        assert_eq!(a, 5, "Tier A should have 5 units");
        assert_eq!(b, 6, "Tier B should have 6 units");
        assert_eq!(c, 8, "Tier C should have 8 units");
        assert_eq!(d, 8, "Tier D should have 8 units");
        assert_eq!(a + b + c + d, 27, "total silicon roadmap = 27");
    }

    #[test]
    fn test_axis_code_roundtrip() {
        assert_eq!(contrast_from_code("STRUC"), Some(ContrastType::Structural));
        assert_eq!(energy_from_code("L0"), Some(EnergyCost::L0Free));
        assert_eq!(depth_from_code("D3"), Some(IntegrationDepth::D3Model));
        assert_eq!(reference_from_code("E2"), Some(ReferenceType::E2Computed));
    }

    #[test]
    fn test_build_cell_from_codes() {
        let cell = build_cell("SPAT", "L2", "D1", "E0").unwrap();
        assert_eq!(cell.notation(), "SPAT x L2 x D1 x E0");
    }

    #[test]
    fn test_build_cell_invalid() {
        assert_eq!(build_cell("NOT", "L2", "D1", "E0"), None);
        assert_eq!(build_cell("SPAT", "L9", "D1", "E0"), None);
    }

    #[test]
    fn test_has_silicon() {
        let fallback_cell = TensorCell::from_notation("SPEC x L0 x D0 x E0").unwrap();
        let coord: AxisCoordinate = fallback_cell.into();
        assert!(!coord.has_silicon());

        let native_cell = TensorCell::from_notation("CAUS x L2 x D3 x E2").unwrap();
        let coord: AxisCoordinate = native_cell.into();
        assert!(coord.has_silicon()); // deep_causal_model_error → A1
    }
}
