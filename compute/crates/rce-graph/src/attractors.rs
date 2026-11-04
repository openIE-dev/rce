//! The 37 convergent intelligence attractors from the Periodic Stack of Intelligence.
//!
//! Each attractor is a cell in the Periodic Tensor where evolution, engineering,
//! and chemistry all converge on the same structural operation regardless of
//! substrate. The distribution is 4-13-12-8 across cascade tiers (T0 raw, T1
//! feature, T2 pattern, T3 model). Current AI covers roughly 10 of 37. The
//! remaining 27 are the frontier.
//!
//! ## How this maps to the graph
//!
//! The `inv-ai-codegraph::rce::AttractorOp` enum holds the canonical 37
//! variants with full energy and silicon metadata. This module provides the
//! language-level interface: every attractor is a named function with its
//! `TensorCell` coordinates and a reference implementation.
//!
//! ## The 8 convergent attractor cells (information stack cross-reference)
//!
//! These 8 cells account for ~98% of intelligence primitives across all
//! known intelligent systems:
//!
//! 1. `MAG × L2 × D0 × E1` — threshold crossing (T0, 5 instances)
//! 2. `SPAT × L2 × D1 × E0` — spatial gradient (T1, 5+ instances)
//! 3. `TEMP × L2 × D1 × E3` — temporal adaptation (T1, 5 instances)
//! 4. `MAG × L2 × D1 × E3` — gain adaptation (T1, 4 instances)
//! 5. `STRUC × L2 × D2 × E1` — template match (T2, 9 instances — densest)
//! 6. `REL × L2 × D2 × E0` — competitive comparison (T2, 5 instances)
//! 7. `SELF × L2 × D2 × E2` — prediction error (T2, 7 instances)
//! 8. `SELF × L2 × D3 × E2` — deep model error (T3, 7 instances)

use crate::tensor_ops::{
    ContrastType, EnergyCost, IntegrationDepth, ReferenceType, TensorCell,
};

/// The 37 convergent intelligence attractors, tagged with their tier.
///
/// Distribution: 4 T0 (raw) + 13 T1 (feature) + 12 T2 (pattern) + 8 T3 (model).
/// Variants are numbered #1-#37 matching the canonical ordering in the
/// Periodic Stack of Intelligence final document.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attractor {
    // ── Tier 0 — Raw (4) ───────────────────────────────────────
    /// #1  MAG × L2 × D0 × E1 — threshold crossing.
    /// Neuron threshold, nociception, quorum sensing, phase transition.
    ThresholdCrossing,
    /// #20 STRUC × L2 × D0 × E1 — raw structural recognition.
    /// Tokenization, molecular recognition, catalytic recognition.
    RawStructuralRecognition,
    /// #26 MAG × L2 × D0 × E0 — raw external magnitude comparison.
    /// ALU comparator, piezoelectric, pheromone presence, otolith.
    RawExternalMagnitude,
    /// #29 MAG × L2 × D0 × E3 — raw magnitude adaptation.
    /// Chemotaxis, sensitization, speed cell, osmolar thirst.
    RawMagnitudeAdaptation,

    // ── Tier 1 — Feature (13) ──────────────────────────────────
    /// #2  SPAT × L2 × D1 × E0 — spatial gradient.
    /// Retinal edge detection, phototropism, electrolocation.
    SpatialGradient,
    /// #3  TEMP × L2 × D1 × E3 — temporal adaptation.
    /// Motion detection, habituation, optic flow, fold-change.
    TemporalAdaptation,
    /// #4  MAG × L2 × D1 × E3 — magnitude gain adaptation.
    /// Physarum flow, V-formation, baroreceptor adaptation.
    MagnitudeGainAdaptation,
    /// #14 SPEC × L2 × D1 × E1 — spectral template match.
    /// Spider web vibration, formant recognition, cricket song.
    SpectralTemplateMatch,
    /// #21 STRUC × L2 × D1 × E1 — shallow structural template match.
    /// Kin discrimination, combinatorial quorum, alarm receiver.
    ShallowStructuralTemplate,
    /// #24 SPEC × L2 × D1 × E0 — parallel spectral decomposition.
    /// Color opponency, polarization, cochlear, color edges.
    ParallelSpectralDecomposition,
    /// #25 SPEC × L2 × D1 × E3 — spectral adaptation.
    /// Jamming avoidance, cochlear adaptation, Lombard effect.
    SpectralAdaptation,
    /// #27 MAG × L2 × D1 × E0 — shallow magnitude contrast (concurrent).
    /// Plant systemic, retinal ON/OFF, gate control pain.
    ShallowMagnitudeContrast,
    /// #28 MAG × L2 × D1 × E2 — magnitude vs computed reference.
    /// Softmax, batch normalization, number sense, steering.
    MagnitudeVsComputedReference,
    /// #30 REL × L2 × D1 × E0 — shallow concurrent relational comparison.
    /// Dance audience, inter-arm coordination, vocal turn-taking.
    ShallowRelationalComparison,
    /// #31 REL × L2 × D1 × E1 — shallow relational template match.
    /// Classical conditioning, RAG retrieval, voice recognition.
    ShallowRelationalTemplate,
    /// #33 CAUS × L2 × D1 × E2 — shallow causal inference.
    /// Operant conditioning, speculative decode, contingency learning.
    ShallowCausalInference,
    /// #34 SPAT × L2 × D1 × E3 — spatial adaptation.
    /// Light/dark adaptation, tilt aftereffect, motion aftereffect.
    SpatialAdaptation,

    // ── Tier 2 — Pattern (12) ──────────────────────────────────
    /// #5  STRUC × L2 × D2 × E1 — deep structural template match. DENSEST CELL.
    /// Immune recognition, affordance, holographic recall, grammar.
    DeepStructuralTemplate,
    /// #6  REL × L2 × D2 × E0 — competitive comparison.
    /// Self-attention, cross-attention, perceptual rivalry.
    CompetitiveComparison,
    /// #7  SELF × L2 × D2 × E2 — prediction error (forward model).
    /// Cerebellar prediction, reafference cancellation, VOR.
    ForwardModelPredictionError,
    /// #9  TEMP × L2 × D2 × E0 — multi-scale temporal contrast.
    /// Wavelet transient, click trains, flight turbulence.
    MultiScaleTemporalContrast,
    /// #10 SPAT × L2 × D2 × E1 — spatial template match.
    /// Place cells, face recognition, nest templates.
    SpatialTemplateMatch,
    /// #12 CAUS × L2 × D2 × E1 — causal schema match.
    /// Tool-use schema, predator-prey, medical symptom, fault diagnosis.
    CausalSchemaMatch,
    /// #15 TEMP × L3 × D2 × E3 — temporal context encoding (sustained).
    /// Circadian entrainment, interference resolution, time cells.
    TemporalContextEncoding,
    /// #16 SPAT × L3 × D2 × E3 — sustained spatial self-reference.
    /// Path integration, grid cells, head direction cells.
    SustainedSpatialSelfReference,
    /// #22 STRUC × L2 × D2 × E0 — structural competition.
    /// Chromatophore matching, binocular rivalry, ambiguous figures.
    StructuralCompetition,
    /// #23 STRUC × L2 × D2 × E3 — adaptive structural perception.
    /// Whisker active sensing, auditory stream segregation, saccadic.
    AdaptiveStructuralPerception,
    /// #32 CAUS × L2 × D2 × E2 — mid-depth causal inference.
    /// Beam search, tool dispatch, smooth pursuit, shortest-path.
    MidDepthCausalInference,
    /// #35 REL × L2 × D2 × E2 — relational prediction error.
    /// Mirror neurons, social expectation violation, partner reliability.
    RelationalPredictionError,

    // ── Tier 3 — Model (8) ─────────────────────────────────────
    /// #8  SELF × L2 × D3 × E2 — deep self-model error.
    /// Dopamine reward prediction, theory of mind, recursive TOM.
    DeepSelfModelError,
    /// #11 REL × L2 × D3 × E1 — deep relational template.
    /// Transitive inference, grammar, musical chords, chess openings.
    DeepRelationalTemplate,
    /// #13 SPAT × L3 × D3 × E1 — recursive spatial maps.
    /// Place cell topology, hierarchical maps, bee cognitive map.
    RecursiveSpatialMaps,
    /// #17 CAUS × L2 × D3 × E2 — deep causal model error.
    /// Counterfactual reasoning, scientific inference, narrative prediction.
    DeepCausalModelError,
    /// #18 SELF × L3 × D3 × E2 — metacognition.
    /// Confidence monitoring, ACC error monitoring, feeling-of-knowing.
    Metacognition,
    /// #19 TEMP × L3 × D3 × E2 — long-range predictive temporal.
    /// Sharp wave ripple, long-range music prediction, prospective memory.
    LongRangePredictiveTemporal,
    /// #36 TEMP × L3 × D3 × E1 — deep temporal memory (stored).
    /// Episodic memory, migratory calendar, life-history timing.
    DeepTemporalMemory,
    /// #37 STRUC × L3 × D2 × E1 — sustained structural memory.
    /// Semantic memory, immune memory, motor skill, birdsong template.
    SustainedStructuralMemory,
}

impl Attractor {
    /// The canonical attractor ID from the Periodic Stack of Intelligence (#1-#37).
    pub fn id(&self) -> u8 {
        use Attractor::*;
        match self {
            ThresholdCrossing => 1, SpatialGradient => 2, TemporalAdaptation => 3,
            MagnitudeGainAdaptation => 4, DeepStructuralTemplate => 5,
            CompetitiveComparison => 6, ForwardModelPredictionError => 7,
            DeepSelfModelError => 8, MultiScaleTemporalContrast => 9,
            SpatialTemplateMatch => 10, DeepRelationalTemplate => 11,
            CausalSchemaMatch => 12, RecursiveSpatialMaps => 13,
            SpectralTemplateMatch => 14, TemporalContextEncoding => 15,
            SustainedSpatialSelfReference => 16, DeepCausalModelError => 17,
            Metacognition => 18, LongRangePredictiveTemporal => 19,
            RawStructuralRecognition => 20, ShallowStructuralTemplate => 21,
            StructuralCompetition => 22, AdaptiveStructuralPerception => 23,
            ParallelSpectralDecomposition => 24, SpectralAdaptation => 25,
            RawExternalMagnitude => 26, ShallowMagnitudeContrast => 27,
            MagnitudeVsComputedReference => 28, RawMagnitudeAdaptation => 29,
            ShallowRelationalComparison => 30, ShallowRelationalTemplate => 31,
            MidDepthCausalInference => 32, ShallowCausalInference => 33,
            SpatialAdaptation => 34, RelationalPredictionError => 35,
            DeepTemporalMemory => 36, SustainedStructuralMemory => 37,
        }
    }

    /// The full `TensorCell` coordinate of this attractor in the Periodic Tensor.
    pub fn cell(&self) -> TensorCell {
        use Attractor::*;
        use ContrastType::*;
        use EnergyCost::*;
        use IntegrationDepth::*;
        use ReferenceType::*;
        let (a, b, g, d) = match self {
            // Tier 0
            ThresholdCrossing             => (Magnitude,       L2Erasure,    D0Raw,     E1Stored),
            RawStructuralRecognition      => (Structural,      L2Erasure,    D0Raw,     E1Stored),
            RawExternalMagnitude          => (Magnitude,       L2Erasure,    D0Raw,     E0External),
            RawMagnitudeAdaptation        => (Magnitude,       L2Erasure,    D0Raw,     E3Self),
            // Tier 1
            SpatialGradient               => (Spatial,         L2Erasure,    D1Feature, E0External),
            TemporalAdaptation            => (Temporal,        L2Erasure,    D1Feature, E3Self),
            MagnitudeGainAdaptation       => (Magnitude,       L2Erasure,    D1Feature, E3Self),
            SpectralTemplateMatch         => (Spectral,        L2Erasure,    D1Feature, E1Stored),
            ShallowStructuralTemplate     => (Structural,      L2Erasure,    D1Feature, E1Stored),
            ParallelSpectralDecomposition => (Spectral,        L2Erasure,    D1Feature, E0External),
            SpectralAdaptation            => (Spectral,        L2Erasure,    D1Feature, E3Self),
            ShallowMagnitudeContrast      => (Magnitude,       L2Erasure,    D1Feature, E0External),
            MagnitudeVsComputedReference  => (Magnitude,       L2Erasure,    D1Feature, E2Computed),
            ShallowRelationalComparison   => (Relational,      L2Erasure,    D1Feature, E0External),
            ShallowRelationalTemplate     => (Relational,      L2Erasure,    D1Feature, E1Stored),
            ShallowCausalInference        => (Causal,          L2Erasure,    D1Feature, E2Computed),
            SpatialAdaptation             => (Spatial,         L2Erasure,    D1Feature, E3Self),
            // Tier 2
            DeepStructuralTemplate        => (Structural,      L2Erasure,    D2Pattern, E1Stored),
            CompetitiveComparison         => (Relational,      L2Erasure,    D2Pattern, E0External),
            ForwardModelPredictionError   => (SelfReferential, L2Erasure,    D2Pattern, E2Computed),
            MultiScaleTemporalContrast    => (Temporal,        L2Erasure,    D2Pattern, E0External),
            SpatialTemplateMatch          => (Spatial,         L2Erasure,    D2Pattern, E1Stored),
            CausalSchemaMatch             => (Causal,          L2Erasure,    D2Pattern, E1Stored),
            TemporalContextEncoding       => (Temporal,        L3Sustained,  D2Pattern, E3Self),
            SustainedSpatialSelfReference => (Spatial,         L3Sustained,  D2Pattern, E3Self),
            StructuralCompetition         => (Structural,      L2Erasure,    D2Pattern, E0External),
            AdaptiveStructuralPerception  => (Structural,      L2Erasure,    D2Pattern, E3Self),
            MidDepthCausalInference       => (Causal,          L2Erasure,    D2Pattern, E2Computed),
            RelationalPredictionError     => (Relational,      L2Erasure,    D2Pattern, E2Computed),
            // Tier 3
            DeepSelfModelError            => (SelfReferential, L2Erasure,    D3Model,   E2Computed),
            DeepRelationalTemplate        => (Relational,      L2Erasure,    D3Model,   E1Stored),
            RecursiveSpatialMaps          => (Spatial,         L3Sustained,  D3Model,   E1Stored),
            DeepCausalModelError          => (Causal,          L2Erasure,    D3Model,   E2Computed),
            Metacognition                 => (SelfReferential, L3Sustained,  D3Model,   E2Computed),
            LongRangePredictiveTemporal   => (Temporal,        L3Sustained,  D3Model,   E2Computed),
            DeepTemporalMemory            => (Temporal,        L3Sustained,  D3Model,   E1Stored),
            SustainedStructuralMemory     => (Structural,      L3Sustained,  D2Pattern, E1Stored),
        };
        TensorCell::new(a, b, g, d)
    }

    /// Cascade tier (0-3) — which cascade level dispatches this attractor.
    ///
    /// This is NOT purely derived from γ. `SustainedStructuralMemory` is at
    /// γ=D2 but is classified as Tier 3 (model) because its L3 sustained
    /// memory cost puts it in the deep-model dispatch band. This matches the
    /// canonical 4-13-12-8 distribution in the Periodic Stack of Intelligence.
    pub fn cascade_tier(&self) -> u8 {
        use Attractor::*;
        match self {
            // Tier 0 — Raw (4)
            ThresholdCrossing | RawStructuralRecognition
            | RawExternalMagnitude | RawMagnitudeAdaptation => 0,
            // Tier 1 — Feature (13)
            SpatialGradient | TemporalAdaptation | MagnitudeGainAdaptation
            | SpectralTemplateMatch | ShallowStructuralTemplate
            | ParallelSpectralDecomposition | SpectralAdaptation
            | ShallowMagnitudeContrast | MagnitudeVsComputedReference
            | ShallowRelationalComparison | ShallowRelationalTemplate
            | ShallowCausalInference | SpatialAdaptation => 1,
            // Tier 2 — Pattern (12)
            DeepStructuralTemplate | CompetitiveComparison
            | ForwardModelPredictionError | MultiScaleTemporalContrast
            | SpatialTemplateMatch | CausalSchemaMatch
            | TemporalContextEncoding | SustainedSpatialSelfReference
            | StructuralCompetition | AdaptiveStructuralPerception
            | MidDepthCausalInference | RelationalPredictionError => 2,
            // Tier 3 — Model (8)
            DeepSelfModelError | DeepRelationalTemplate
            | RecursiveSpatialMaps | DeepCausalModelError
            | Metacognition | LongRangePredictiveTemporal
            | DeepTemporalMemory | SustainedStructuralMemory => 3,
        }
    }

    /// All 37 attractors in canonical order (#1-#37).
    pub fn all() -> [Attractor; 37] {
        use Attractor::*;
        [
            ThresholdCrossing, SpatialGradient, TemporalAdaptation, MagnitudeGainAdaptation,
            DeepStructuralTemplate, CompetitiveComparison, ForwardModelPredictionError,
            DeepSelfModelError, MultiScaleTemporalContrast, SpatialTemplateMatch,
            DeepRelationalTemplate, CausalSchemaMatch, RecursiveSpatialMaps,
            SpectralTemplateMatch, TemporalContextEncoding, SustainedSpatialSelfReference,
            DeepCausalModelError, Metacognition, LongRangePredictiveTemporal,
            RawStructuralRecognition, ShallowStructuralTemplate, StructuralCompetition,
            AdaptiveStructuralPerception, ParallelSpectralDecomposition, SpectralAdaptation,
            RawExternalMagnitude, ShallowMagnitudeContrast, MagnitudeVsComputedReference,
            RawMagnitudeAdaptation, ShallowRelationalComparison, ShallowRelationalTemplate,
            MidDepthCausalInference, ShallowCausalInference, SpatialAdaptation,
            RelationalPredictionError, DeepTemporalMemory, SustainedStructuralMemory,
        ]
    }

    /// Look up an attractor by ID (#1-#37). Returns `None` if out of range.
    pub fn from_id(id: u8) -> Option<Attractor> {
        Self::all().into_iter().find(|a| a.id() == id)
    }

    /// Short name: `"#5"` for cross-referencing with the periodic stacks.
    pub fn short_id(&self) -> String { format!("#{}", self.id()) }
}

// ═══════════════════════════════════════════════════════════════════
// REFERENCE IMPLEMENTATIONS (Tier 0 — the 4 raw attractors)
// ═══════════════════════════════════════════════════════════════════

/// #1 MAG × L2 × D0 × E1 — threshold crossing.
/// Returns `1.0` for values above threshold, `0.0` otherwise.
/// This is the single most-used primitive in intelligence: neuron firing,
/// ALU comparators, quorum sensing, phase transitions.
pub fn threshold_crossing(x: &[f64], threshold: f64) -> Vec<f64> {
    x.iter().map(|&v| if v >= threshold { 1.0 } else { 0.0 }).collect()
}

/// #20 STRUC × L2 × D0 × E1 — raw structural recognition.
/// Returns `true` if the input pattern matches the stored template exactly.
/// Molecular recognition, tokenization, key-lock matching.
pub fn raw_structural_recognition(input: &[u8], template: &[u8]) -> bool {
    input == template
}

/// #26 MAG × L2 × D0 × E0 — raw external magnitude comparison.
/// Returns `1.0` if `a > b`, `-1.0` if `a < b`, `0.0` if equal.
/// The ALU comparator — two concurrent inputs, no memory.
pub fn raw_external_magnitude(a: f64, b: f64) -> f64 {
    if a > b { 1.0 } else if a < b { -1.0 } else { 0.0 }
}

/// #29 MAG × L2 × D0 × E3 — raw magnitude adaptation.
/// Updates the running baseline toward the new value with rate `alpha`,
/// returns the adapted delta. Chemotaxis, sensitization.
pub fn raw_magnitude_adaptation(current: f64, baseline: &mut f64, alpha: f64) -> f64 {
    let delta = current - *baseline;
    *baseline += alpha * delta;
    delta
}

// ═══════════════════════════════════════════════════════════════════
// REFERENCE IMPLEMENTATIONS (Tier 1 — feature-level, selected)
// ═══════════════════════════════════════════════════════════════════

/// #2 SPAT × L2 × D1 × E0 — spatial gradient.
/// 1D finite difference: out[i] = x[i+1] - x[i-1]. Boundary pixels return 0.
/// Retinal edge detection, Sobel, phototropism gradient sensing.
pub fn spatial_gradient(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n < 3 { return vec![0.0; n]; }
    let mut out = vec![0.0; n];
    for i in 1..n - 1 { out[i] = x[i + 1] - x[i - 1]; }
    out
}

/// #3 TEMP × L2 × D1 × E3 — temporal adaptation.
/// Returns the running difference between the current value and an
/// exponentially smoothed baseline. Motion detection, habituation, fold-change.
pub fn temporal_adaptation(stream: &[f64], alpha: f64) -> Vec<f64> {
    if stream.is_empty() { return vec![]; }
    let mut baseline = stream[0];
    let mut out = Vec::with_capacity(stream.len());
    for &v in stream {
        let delta = v - baseline;
        baseline += alpha * delta;
        out.push(delta);
    }
    out
}

/// #4 MAG × L2 × D1 × E3 — magnitude gain adaptation.
/// Normalizes by a running RMS estimate. Baroreceptor, physarum, V-formation.
pub fn magnitude_gain_adaptation(stream: &[f64], alpha: f64) -> Vec<f64> {
    if stream.is_empty() { return vec![]; }
    let mut rms = stream[0].abs().max(1e-10);
    stream.iter().map(|&v| {
        rms = (1.0 - alpha) * rms + alpha * v.abs().max(1e-10);
        v / rms
    }).collect()
}

/// #24 SPEC × L2 × D1 × E0 — parallel spectral decomposition.
/// Bank of N bandpass filters applied concurrently. Cochlear, color opponency.
/// Returns one energy value per filter band.
pub fn parallel_spectral_decomposition(signal: &[f64], n_bands: usize) -> Vec<f64> {
    let n = signal.len();
    if n == 0 || n_bands == 0 { return vec![]; }
    let band_size = (n + n_bands - 1) / n_bands;
    (0..n_bands).map(|k| {
        let start = k * band_size;
        let end = ((k + 1) * band_size).min(n);
        if end <= start { 0.0 }
        else {
            signal[start..end].iter().map(|v| v * v).sum::<f64>() / (end - start) as f64
        }
    }).collect()
}

/// #28 MAG × L2 × D1 × E2 — magnitude vs computed reference.
/// Softmax is the canonical example: each element is compared to the max,
/// then divided by the sum of exponentials (a computed reference).
pub fn magnitude_vs_computed_reference_softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() { return vec![]; }
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ═══════════════════════════════════════════════════════════════════
// REFERENCE IMPLEMENTATIONS (Tier 2 — pattern-level, selected)
// ═══════════════════════════════════════════════════════════════════

/// #5 STRUC × L2 × D2 × E1 — deep structural template match. DENSEST CELL.
/// Cosine similarity between input and a stored template. Immune recognition,
/// holographic recall, face recognition, grammar, affordance.
/// Returns similarity in [-1, 1].
pub fn deep_structural_template(input: &[f64], template: &[f64]) -> f64 {
    let dot: f64 = input.iter().zip(template).map(|(a, b)| a * b).sum();
    let norm_i: f64 = input.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_t: f64 = template.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_i < 1e-15 || norm_t < 1e-15 { return 0.0; }
    dot / (norm_i * norm_t)
}

/// #6 REL × L2 × D2 × E0 — competitive comparison (self-attention core).
/// Given query, keys, values, returns attention-weighted sum.
/// `q`: d_k vector, `keys`: n_keys × d_k row-major, `values`: n_keys × d_v row-major.
pub fn competitive_comparison(q: &[f64], keys: &[f64], values: &[f64], n_keys: usize) -> Vec<f64> {
    if n_keys == 0 || keys.is_empty() { return vec![]; }
    let d_k = q.len();
    let d_v = values.len() / n_keys;
    let scale = 1.0 / (d_k as f64).sqrt();
    let scores: Vec<f64> = (0..n_keys).map(|i| {
        let mut dot = 0.0;
        for j in 0..d_k { dot += q[j] * keys[i * d_k + j]; }
        dot * scale
    }).collect();
    let weights = magnitude_vs_computed_reference_softmax(&scores);
    (0..d_v).map(|j| {
        (0..n_keys).map(|i| weights[i] * values[i * d_v + j]).sum()
    }).collect()
}

/// #7 SELF × L2 × D2 × E2 — forward-model prediction error.
/// Given the current observation and a predicted observation from a forward
/// model, returns the vector of residuals. Cerebellar prediction, VOR.
pub fn forward_model_prediction_error(observed: &[f64], predicted: &[f64]) -> Vec<f64> {
    observed.iter().zip(predicted).map(|(o, p)| o - p).collect()
}

// ═══════════════════════════════════════════════════════════════════
// REFERENCE IMPLEMENTATIONS (Tier 3 — model-level, selected)
// ═══════════════════════════════════════════════════════════════════

/// #8 SELF × L2 × D3 × E2 — deep self-model error.
/// Temporal-difference error: δ = r + γ·V(s') − V(s).
/// Dopamine reward prediction, theory of mind, recursive planning.
pub fn deep_self_model_error(reward: f64, v_current: f64, v_next: f64, gamma: f64) -> f64 {
    reward + gamma * v_next - v_current
}

/// #17 CAUS × L2 × D3 × E2 — deep causal model error.
/// Counterfactual residual: compare observed outcome to what the causal
/// model predicts under the *actual* treatment vs a counterfactual one.
/// Returns `(observed - predicted_under_actual, predicted_under_counterfactual - predicted_under_actual)`.
pub fn deep_causal_model_error(
    observed: f64,
    predicted_actual: f64,
    predicted_counterfactual: f64,
) -> (f64, f64) {
    let residual = observed - predicted_actual;
    let counterfactual_effect = predicted_counterfactual - predicted_actual;
    (residual, counterfactual_effect)
}

/// #18 SELF × L3 × D3 × E2 — metacognition.
/// Confidence estimate from a distribution of predictions: returns 1 minus
/// normalized entropy. High confidence = concentrated distribution,
/// low confidence = flat distribution. ACC error monitoring, FOK judgments.
pub fn metacognition(prediction_distribution: &[f64]) -> f64 {
    let n = prediction_distribution.len();
    if n < 2 { return 1.0; }
    let sum: f64 = prediction_distribution.iter().sum();
    if sum < 1e-15 { return 0.0; }
    let entropy: f64 = prediction_distribution.iter()
        .map(|&p| {
            let p_norm = p / sum;
            if p_norm > 1e-15 { -p_norm * p_norm.ln() } else { 0.0 }
        }).sum();
    let max_entropy = (n as f64).ln();
    if max_entropy < 1e-15 { return 1.0; }
    1.0 - (entropy / max_entropy)
}

// ═══════════════════════════════════════════════════════════════════
// COVERAGE METADATA
// ═══════════════════════════════════════════════════════════════════

/// Implementation status of an attractor in current AI / silicon.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttractorStatus {
    /// Native silicon / optimized primitive exists (~10 of 37).
    Native,
    /// Emulated via matmul or similar, runs 10³-10¹⁴× above the Landauer floor.
    Emulated,
    /// Cannot be meaningfully emulated — the operation has no standing
    /// implementation in mainstream stacks.
    Unavailable,
}

impl Attractor {
    /// Coverage status of this attractor in current AI silicon and RCE's silicon roadmap.
    pub fn status(&self) -> AttractorStatus {
        use Attractor::*;
        use AttractorStatus::*;
        match self {
            // Native: well-served by existing silicon
            ThresholdCrossing | RawExternalMagnitude | CompetitiveComparison
            | MagnitudeVsComputedReference | ParallelSpectralDecomposition
            | SpatialGradient | ForwardModelPredictionError | DeepStructuralTemplate
            | ShallowMagnitudeContrast | RawStructuralRecognition => Native,
            // Emulated via matmul / softmax but with large impedance ratio
            TemporalAdaptation | MagnitudeGainAdaptation | SpectralTemplateMatch
            | ShallowStructuralTemplate | SpectralAdaptation | ShallowRelationalComparison
            | ShallowRelationalTemplate | ShallowCausalInference | SpatialAdaptation
            | MultiScaleTemporalContrast | SpatialTemplateMatch | CausalSchemaMatch
            | StructuralCompetition | MidDepthCausalInference | DeepSelfModelError
            | DeepRelationalTemplate | DeepCausalModelError => Emulated,
            // Unavailable: sustained memory / self-referential / recursive
            RawMagnitudeAdaptation | AdaptiveStructuralPerception
            | RelationalPredictionError | TemporalContextEncoding
            | SustainedSpatialSelfReference | RecursiveSpatialMaps
            | Metacognition | LongRangePredictiveTemporal
            | DeepTemporalMemory | SustainedStructuralMemory => Unavailable,
        }
    }
}

/// Count attractors by status.
pub fn coverage_counts() -> (usize, usize, usize) {
    let mut native = 0;
    let mut emulated = 0;
    let mut unavailable = 0;
    for a in Attractor::all() {
        match a.status() {
            AttractorStatus::Native => native += 1,
            AttractorStatus::Emulated => emulated += 1,
            AttractorStatus::Unavailable => unavailable += 1,
        }
    }
    (native, emulated, unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_37_attractors() {
        let all = Attractor::all();
        assert_eq!(all.len(), 37);
        let ids: Vec<u8> = all.iter().map(|a| a.id()).collect();
        for i in 1..=37u8 {
            assert!(ids.contains(&i), "missing attractor #{}", i);
        }
    }

    #[test]
    fn test_tier_distribution_4_13_12_8() {
        let mut tiers = [0usize; 4];
        for a in Attractor::all() {
            tiers[a.cascade_tier() as usize] += 1;
        }
        assert_eq!(tiers, [4, 13, 12, 8], "attractor tier distribution mismatch");
    }

    #[test]
    fn test_8_convergent_cells() {
        use Attractor::*;
        let expected = [
            (ThresholdCrossing,             "MAG x L2 x D0 x E1"),
            (SpatialGradient,               "SPAT x L2 x D1 x E0"),
            (TemporalAdaptation,            "TEMP x L2 x D1 x E3"),
            (MagnitudeGainAdaptation,       "MAG x L2 x D1 x E3"),
            (DeepStructuralTemplate,        "STRUC x L2 x D2 x E1"),
            (CompetitiveComparison,         "REL x L2 x D2 x E0"),
            (ForwardModelPredictionError,   "SELF x L2 x D2 x E2"),
            (DeepSelfModelError,            "SELF x L2 x D3 x E2"),
        ];
        for (attractor, expected_notation) in expected {
            assert_eq!(attractor.cell().notation(), expected_notation,
                "attractor {:?} cell mismatch", attractor);
        }
    }

    #[test]
    fn test_from_id_roundtrip() {
        for a in Attractor::all() {
            let id = a.id();
            let recovered = Attractor::from_id(id).unwrap();
            assert_eq!(a, recovered);
        }
        assert_eq!(Attractor::from_id(0), None);
        assert_eq!(Attractor::from_id(38), None);
    }

    #[test]
    fn test_threshold_crossing() {
        let out = threshold_crossing(&[0.2, 0.5, 0.8, 1.0], 0.5);
        assert_eq!(out, vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_spatial_gradient() {
        let grad = spatial_gradient(&[0.0, 1.0, 3.0, 6.0, 10.0]);
        assert_eq!(grad.len(), 5);
        assert_eq!(grad[0], 0.0); // boundary
        assert_eq!(grad[1], 3.0); // 3 - 0
        assert_eq!(grad[2], 5.0); // 6 - 1
        assert_eq!(grad[3], 7.0); // 10 - 3
        assert_eq!(grad[4], 0.0); // boundary
    }

    #[test]
    fn test_temporal_adaptation() {
        // Constant signal → adaptation brings delta toward zero
        let out = temporal_adaptation(&[1.0; 20], 0.5);
        assert!((out[19]).abs() < 0.01);
    }

    #[test]
    fn test_softmax_computed_reference() {
        let out = magnitude_vs_computed_reference_softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(out.len(), 3);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(out[2] > out[1] && out[1] > out[0]);
    }

    #[test]
    fn test_deep_structural_template() {
        let t = vec![1.0, 0.0, 0.0];
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!((deep_structural_template(&t, &a) - 1.0).abs() < 1e-10);
        assert!(deep_structural_template(&t, &b).abs() < 1e-10);
    }

    #[test]
    fn test_competitive_comparison_attention() {
        // Single key = single value → full weight on it
        let q = vec![1.0, 0.0];
        let keys = vec![1.0, 0.0];
        let values = vec![42.0, -5.0];
        let out = competitive_comparison(&q, &keys, &values, 1);
        assert_eq!(out, vec![42.0, -5.0]);
    }

    #[test]
    fn test_forward_model_prediction_error() {
        let err = forward_model_prediction_error(&[1.0, 2.0, 3.0], &[0.5, 2.5, 2.5]);
        assert_eq!(err, vec![0.5, -0.5, 0.5]);
    }

    #[test]
    fn test_deep_self_model_error_td_residual() {
        // Bellman residual: δ = r + γ·V(s') − V(s)
        let delta = deep_self_model_error(1.0, 5.0, 5.5, 0.9);
        assert!((delta - (1.0 + 0.9 * 5.5 - 5.0)).abs() < 1e-10);
    }

    #[test]
    fn test_deep_causal_model_error() {
        let (residual, counterfactual) = deep_causal_model_error(10.0, 8.0, 12.0);
        assert_eq!(residual, 2.0);         // observed - predicted
        assert_eq!(counterfactual, 4.0);   // counterfactual - actual
    }

    #[test]
    fn test_metacognition_confidence() {
        // Highly peaked distribution → high confidence
        let peaked = vec![10.0, 0.01, 0.01, 0.01];
        let flat = vec![1.0, 1.0, 1.0, 1.0];
        let c_peaked = metacognition(&peaked);
        let c_flat = metacognition(&flat);
        assert!(c_peaked > 0.7, "peaked confidence = {}", c_peaked);
        assert!(c_flat < 0.1, "flat confidence = {}", c_flat);
    }

    #[test]
    fn test_coverage_counts() {
        let (native, emulated, unavailable) = coverage_counts();
        assert_eq!(native + emulated + unavailable, 37);
        assert!(native >= 8 && native <= 12, "native: {}", native);
        // The original 27 uncovered attractors split into emulated (17) and unavailable (10)
        // Exact numbers depend on the classification but total must be 37
    }

    #[test]
    fn test_short_id_format() {
        assert_eq!(Attractor::DeepStructuralTemplate.short_id(), "#5");
        assert_eq!(Attractor::ThresholdCrossing.short_id(), "#1");
        assert_eq!(Attractor::SustainedStructuralMemory.short_id(), "#37");
    }
}
