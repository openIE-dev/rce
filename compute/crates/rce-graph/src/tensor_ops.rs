//! Tensor operations: reshape, transpose, broadcasting, einsum-like,
//! element-wise ops, reductions, slicing, concatenation, stacking.

// ═══════════════════════════════════════════════════════════════════
// AXIS LABELS
// ═══════════════════════════════════════════════════════════════════

/// Semantic label for a tensor axis.
///
/// Tells the compiler (rce_graph → FGIR → backend) *what* each dimension
/// represents, enabling:
///
/// - **Backend layout choice**: `{Batch, Height, Width, Channel}` → NHWC on
///   ARM, NCHW on GPU. `{X, Y, Z}` → Morton/Hilbert layout for spatial.
/// - **Correctness checking**: can't `MatMul` two `{X,Y,Z}` tensors naively.
/// - **Spatial intelligence**: the cascade distinguishes "3D dense" from
///   "3D sparse" cost based on axis labels. An `{X,Y,Z}` tensor sampled
///   by an octree is L4Communication; the same shape indexed linearly is
///   just FloatMul.
/// - **Energy refinement**: a `{Batch}` axis parallelizes embarrassingly,
///   a `{Time}` axis is inherently sequential — the energy model uses this
///   to calibrate per-op costs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AxisKind {
    /// Batch dimension (embarrassingly parallel, no cross-element deps).
    Batch,
    /// Feature / channel dimension (e.g. RGB, conv filters).
    Channel,
    /// Spatial X axis (width).
    X,
    /// Spatial Y axis (height).
    Y,
    /// Spatial Z axis (depth — 3D spatial intelligence axis).
    Z,
    /// Temporal axis (sequential dependency between elements).
    Time,
    /// Camera / view index (multi-view perception, NeRF, stereo).
    Camera,
    /// View index (when Camera is the physical device and View is a render pass).
    View,
    /// Sequence dimension (NLP tokens, audio frames — like Time but may be
    /// shuffleable via attention).
    Sequence,
    /// Unnamed / unknown dimension (the default — no semantic constraint).
    Unknown,
}

impl AxisKind {
    /// Whether this axis allows embarrassingly-parallel execution (no
    /// cross-element dependencies).
    pub fn is_parallel(&self) -> bool {
        matches!(self, Self::Batch | Self::Channel | Self::Camera | Self::View)
    }

    /// Whether this axis implies sequential dependency.
    pub fn is_sequential(&self) -> bool {
        matches!(self, Self::Time)
    }

    /// Whether this axis represents a spatial dimension ({X, Y, Z}).
    pub fn is_spatial(&self) -> bool {
        matches!(self, Self::X | Self::Y | Self::Z)
    }
}

// ═══════════════════════════════════════════════════════════════════
// CONTRAST TYPE (v10 — Periodic Stack of Intelligence α axis)
// ═══════════════════════════════════════════════════════════════════

/// Semantic contrast type for a whole tensor — the α axis from the
/// Periodic Stack of Intelligence.
///
/// While `AxisKind` labels *individual dimensions* (spatial X/Y/Z,
/// temporal, channel, etc.), `ContrastType` labels *the whole tensor*
/// with the **kind of difference** its values represent. A tensor of
/// pixel intensity values is SPAT. A tensor of spectrogram magnitudes
/// is SPEC. A tensor of temporal deltas is TEMP. A tensor of object
/// prediction errors is SELF.
///
/// The 8 contrast types are the top-level classification of the
/// intelligence stack — they match the α axis in both the intelligence
/// stack and the information stack. Every intelligent operation produces
/// a bit whose contrast type lives in exactly one of these 8 values.
///
/// ## Why this matters for the compiler
///
/// 1. **Attractor dispatch**: when the compiler sees a tensor with a
///    specific contrast type, it can route operations on that tensor to
///    the right attractor primitive. A SPAT tensor → spatial gradient
///    (#2) or spatial template (#10). A SPEC tensor → parallel spectral
///    decomposition (#24). A SELF tensor → prediction error (#7) or
///    deep self-model error (#8).
///
/// 2. **Type safety**: you can't directly combine tensors with
///    incompatible contrast types. A MAG tensor × a SPAT tensor is a
///    structural error — the result has no well-defined contrast type.
///    The compiler can catch this before runtime.
///
/// 3. **Silicon routing**: combined with cascade_tier, contrast type
///    picks the specific silicon unit of RCE's roadmap. SPEC × D1 × E0 → the
///    A4 ParallelSpectralDecomposition unit. SPAT × D2 × E1 → the
///    D2 SpatialTemplateMatch unit.
///
/// 4. **Information stack consistency**: every tensor that enters an
///    operation must have a contrast type that matches the operation's
///    input contrast type. This is the structural version of the
///    information stack's cross-validation (§7.12) at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContrastType {
    /// MAG — Scalar difference: more/less, above/below threshold.
    /// Examples: thermostat readings, neuron firing rates, brightness.
    Magnitude,
    /// TEMP — Temporal difference: change over time, rate, rhythm.
    /// Examples: motion vectors, velocity, heartbeat, audio frames.
    Temporal,
    /// SPAT — Spatial difference: gradient, edge, topology, position.
    /// Examples: pixel coordinates, point clouds, place cell maps, 3D geometry.
    Spatial,
    /// SPEC — Spectral difference: frequency, pitch, color, polarization.
    /// Examples: FFT outputs, cochlear channels, color opponency, bird song.
    Spectral,
    /// STRUC — Structural difference: pattern, arrangement, graph, syntax.
    /// Examples: token embeddings, molecular shapes, immune templates, grammar.
    Structural,
    /// REL — Relational difference: similarity, ordering, analogy, competition.
    /// Examples: attention scores, embeddings, social hierarchies, analogies.
    Relational,
    /// CAUS — Causal difference: if-then, counterfactual, intervention.
    /// Examples: reward predictions, causal graphs, physical dynamics.
    Causal,
    /// SELF — Self-referential difference: prediction error, surprise, self-state.
    /// Examples: loss gradients, cerebellar errors, metacognitive confidence.
    SelfReferential,
    /// No contrast type specified — the default for raw numeric tensors.
    /// Compiler treats it as compatible with everything (no type safety).
    Unspecified,
}

impl ContrastType {
    /// Whether this contrast type is sensory (comes from the environment).
    /// Sensory types are MAG, TEMP, SPAT, SPEC, STRUC.
    pub fn is_sensory(&self) -> bool {
        matches!(
            self,
            Self::Magnitude
                | Self::Temporal
                | Self::Spatial
                | Self::Spectral
                | Self::Structural
        )
    }

    /// Whether this contrast type is cognitive (requires integration).
    /// Cognitive types are REL, CAUS, SELF.
    pub fn is_cognitive(&self) -> bool {
        matches!(
            self,
            Self::Relational | Self::Causal | Self::SelfReferential
        )
    }

    /// Whether two contrast types are directly compatible for an operation
    /// that combines them (e.g. element-wise add, matmul).
    ///
    /// Rules:
    /// - Any type is compatible with itself.
    /// - `Unspecified` is compatible with any type (default fallback).
    /// - Otherwise, two different specified types are incompatible.
    ///
    /// This is conservative — it prevents silent type errors. Operations
    /// that legitimately cross contrast types (e.g. reward prediction error:
    /// MAG → SELF) must explicitly cast via `with_contrast_type`.
    pub fn is_compatible_with(&self, other: Self) -> bool {
        *self == other || *self == Self::Unspecified || other == Self::Unspecified
    }

    /// The short code used in the intelligence stack cell notation
    /// (MAG, TEMP, SPAT, SPEC, STRUC, REL, CAUS, SELF).
    pub fn short_code(&self) -> &'static str {
        match self {
            Self::Magnitude => "MAG",
            Self::Temporal => "TEMP",
            Self::Spatial => "SPAT",
            Self::Spectral => "SPEC",
            Self::Structural => "STRUC",
            Self::Relational => "REL",
            Self::Causal => "CAUS",
            Self::SelfReferential => "SELF",
            Self::Unspecified => "?",
        }
    }

    /// Parse a contrast type from its short code.
    pub fn from_short_code(code: &str) -> Option<Self> {
        match code {
            "MAG" => Some(Self::Magnitude),
            "TEMP" => Some(Self::Temporal),
            "SPAT" => Some(Self::Spatial),
            "SPEC" => Some(Self::Spectral),
            "STRUC" => Some(Self::Structural),
            "REL" => Some(Self::Relational),
            "CAUS" => Some(Self::Causal),
            "SELF" => Some(Self::SelfReferential),
            "?" | "" => Some(Self::Unspecified),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// ENERGY COST (β axis — Periodic Stack thermodynamic cost)
// ═══════════════════════════════════════════════════════════════════

/// Thermodynamic cost level for an operation — the β axis from the
/// Periodic Tensor. This is NOT an estimate; it is the *structural class*
/// of the operation's energy profile.
///
/// The Landauer floor (kT ln 2 ≈ 2.87 × 10⁻²¹ J at 300K per bit erased)
/// determines the theoretical minimum. The actual energy is the floor
/// times the impedance ratio (10² for analog, 10⁶ for GPU, 10¹⁴ for video).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EnergyCost {
    /// L0 — Thermodynamically free. Bijective, zero information erasure.
    /// FFT, wavelets, scatter/gather, HD bind/unbind, orthogonal matmul,
    /// optical SVD, qudit gates, adiabatic logic. 39 known primitives.
    L0Free,
    /// L1 — Measurement. Requires entropy source but doesn't sustain state.
    /// Quantum measurement, probabilistic bit, sampling.
    L1Measurement,
    /// L2 — Erasure. Destroys information: kT ln 2 per bit erased.
    /// Comparison, thresholding, ReLU, pooling, hashing.
    L2Erasure,
    /// L3 — Sustained. Maintains state across time: continuous energy drain.
    /// Memory, attention KV cache, persistent associative memory.
    L3Sustained,
    /// L4 — Communication. Moves bits across space: energy ∝ distance × bits.
    /// Network I/O, PCIe, inter-chip links, 3D spatial sampling.
    L4Communication,
}

impl EnergyCost {
    pub fn short_code(&self) -> &'static str {
        match self { Self::L0Free => "L0", Self::L1Measurement => "L1", Self::L2Erasure => "L2", Self::L3Sustained => "L3", Self::L4Communication => "L4" }
    }
    pub fn from_short_code(code: &str) -> Option<Self> {
        match code { "L0" => Some(Self::L0Free), "L1" => Some(Self::L1Measurement), "L2" => Some(Self::L2Erasure), "L3" => Some(Self::L3Sustained), "L4" => Some(Self::L4Communication), _ => None }
    }
    pub fn is_free(&self) -> bool { *self == Self::L0Free }
    pub fn landauer_floor_pj(&self) -> f64 {
        match self { Self::L0Free => 0.0, Self::L1Measurement => 0.003, Self::L2Erasure => 0.003, Self::L3Sustained => 0.003, Self::L4Communication => 1.0 }
    }
}

// ═══════════════════════════════════════════════════════════════════
// INTEGRATION DEPTH (γ axis — cascade tier / intelligence depth)
// ═══════════════════════════════════════════════════════════════════

/// Integration depth — the γ axis from the Periodic Tensor.
/// Maps directly to cascade dispatch tiers: the cascade walks D0 → D3,
/// stopping at the first tier that resolves the query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum IntegrationDepth {
    /// D0 — Raw: single comparison, no integration. Threshold crossing.
    /// Cascade tier T0. Cache lookup, pattern-lang match.
    D0Raw,
    /// D1 — Feature: single-step integration over a local neighborhood.
    /// Spatial gradient, temporal adaptation, gain control.
    /// Cascade tier T1. Eigenbasis, spatial 3D.
    D1Feature,
    /// D2 — Pattern: multi-step integration, competitive dynamics.
    /// Template match, attention, prediction error.
    /// Cascade tier T2. Holographic recall, flowR reasoning.
    D2Pattern,
    /// D3 — Model: recursive, self-referential, sustained.
    /// Deep causal models, metacognition, long-range memory.
    /// Cascade tier T3. Neural/LLM tier.
    D3Model,
}

impl IntegrationDepth {
    pub fn short_code(&self) -> &'static str {
        match self { Self::D0Raw => "D0", Self::D1Feature => "D1", Self::D2Pattern => "D2", Self::D3Model => "D3" }
    }
    pub fn from_short_code(code: &str) -> Option<Self> {
        match code { "D0" => Some(Self::D0Raw), "D1" => Some(Self::D1Feature), "D2" => Some(Self::D2Pattern), "D3" => Some(Self::D3Model), _ => None }
    }
    pub fn cascade_tier(&self) -> u8 {
        match self { Self::D0Raw => 0, Self::D1Feature => 1, Self::D2Pattern => 2, Self::D3Model => 3 }
    }
}

// ═══════════════════════════════════════════════════════════════════
// REFERENCE TYPE (δ axis — novel axis from the Information Stack)
// ═══════════════════════════════════════════════════════════════════

/// Reference type — the δ axis from the Periodic Stack of Information.
/// This is the novel contribution: what the contrast is measured *against*
/// determines storage cost, capacity regimes, and the structure of coding theory.
///
/// Shannon's framework assumes the reference is given (mutual information
/// compares two present variables). The δ axis shows that the reference
/// source is a load-bearing structural distinction that partitions both
/// channel coding and rate-distortion into four regimes each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceType {
    /// E0 — External: contrast measured against environment.
    /// Raw sensor input, direct observation. The reference is outside the system.
    E0External,
    /// E1 — Stored: contrast measured against internal memory.
    /// Template match, cache lookup, recognition. The reference was previously acquired.
    E1Stored,
    /// E2 — Computed: contrast measured against a derived quantity.
    /// Prediction error, model residual. The reference is generated by computation.
    E2Computed,
    /// E3 — Self-referential: contrast measured against the system's own state.
    /// Metacognition, self-model error, homeostasis. The observer IS the reference.
    E3Self,
}

impl ReferenceType {
    pub fn short_code(&self) -> &'static str {
        match self { Self::E0External => "E0", Self::E1Stored => "E1", Self::E2Computed => "E2", Self::E3Self => "E3" }
    }
    pub fn from_short_code(code: &str) -> Option<Self> {
        match code { "E0" => Some(Self::E0External), "E1" => Some(Self::E1Stored), "E2" => Some(Self::E2Computed), "E3" => Some(Self::E3Self), _ => None }
    }
    pub fn requires_memory(&self) -> bool { matches!(self, Self::E1Stored | Self::E3Self) }
    pub fn requires_model(&self) -> bool { matches!(self, Self::E2Computed | Self::E3Self) }
}

// ═══════════════════════════════════════════════════════════════════
// TENSOR CELL — full Periodic Tensor coordinate
// ═══════════════════════════════════════════════════════════════════

/// A cell in the Periodic Tensor — the full coordinate of an operation
/// across the α, β, γ, δ axes. This is the intelligence stack's 4D
/// coordinate system: what kind of contrast × how much energy × how deep
/// the integration × what the reference is.
///
/// Every tensor operation in JMax can optionally carry a cell coordinate.
/// When present, the cascade uses it for dispatch: the cell determines
/// which attractor primitive handles the operation and what silicon unit
/// executes it.
///
/// Cell notation: `SPAT × L2 × D1 × E0` (spatial gradient, erasure cost,
/// feature-level integration, external reference).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorCell {
    pub alpha: ContrastType,
    pub beta: EnergyCost,
    pub gamma: IntegrationDepth,
    pub delta: ReferenceType,
}

impl TensorCell {
    pub fn new(alpha: ContrastType, beta: EnergyCost, gamma: IntegrationDepth, delta: ReferenceType) -> Self {
        Self { alpha, beta, gamma, delta }
    }

    /// Parse from cell notation: "SPAT x L2 x D1 x E0".
    pub fn from_notation(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.split(" x ").map(|p| p.trim()).collect();
        if parts.len() != 4 { return None; }
        Some(Self {
            alpha: ContrastType::from_short_code(parts[0])?,
            beta: EnergyCost::from_short_code(parts[1])?,
            gamma: IntegrationDepth::from_short_code(parts[2])?,
            delta: ReferenceType::from_short_code(parts[3])?,
        })
    }

    /// Format as cell notation: "SPAT x L2 x D1 x E0".
    pub fn notation(&self) -> String {
        format!("{} x {} x {} x {}", self.alpha.short_code(), self.beta.short_code(), self.gamma.short_code(), self.delta.short_code())
    }

    /// The cascade tier this cell dispatches to (0-3).
    pub fn cascade_tier(&self) -> u8 { self.gamma.cascade_tier() }

    /// Whether this cell's operation is thermodynamically free (L0).
    pub fn is_free(&self) -> bool { self.beta.is_free() }

    /// Whether this cell requires persistent memory (E1 or E3 reference).
    pub fn requires_memory(&self) -> bool { self.delta.requires_memory() }

    /// Whether this cell requires a computational model (E2 or E3 reference).
    pub fn requires_model(&self) -> bool { self.delta.requires_model() }

    /// Landauer floor energy in picojoules for this cell.
    pub fn landauer_floor_pj(&self) -> f64 { self.beta.landauer_floor_pj() }
}

impl std::fmt::Display for TensorCell {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.notation())
    }
}

// ═══════════════════════════════════════════════════════════════════
// TENSOR REPRESENTATION
// ═══════════════════════════════════════════════════════════════════

/// N-dimensional tensor (row-major contiguous storage).
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    /// Optional semantic labels for each axis. When present, `axes.len()`
    /// must equal `shape.len()`. When absent (`None`), all axes are `Unknown`.
    pub axes: Option<Vec<AxisKind>>,
    /// Optional whole-tensor contrast type — the α axis from the Periodic
    /// Stack of Intelligence. Labels what *kind of contrast* this tensor's
    /// values represent (magnitude, temporal, spatial, spectral, structural,
    /// relational, causal, self-referential).
    ///
    /// When absent (`None`), the tensor is untyped — the compiler treats it
    /// as compatible with every contrast type (no type safety). When present,
    /// operations combining this tensor with another must pass compatibility
    /// via `ContrastType::is_compatible_with`.
    pub contrast_type: Option<ContrastType>,
}

impl Tensor {
    /// Create tensor from flat data and shape (no axis labels, no contrast type).
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        let strides = compute_strides(&shape);
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape, strides, axes: None, contrast_type: None }
    }

    /// Create tensor from flat data, shape, and axis labels.
    pub fn new_labeled(data: Vec<f64>, shape: Vec<usize>, axes: Vec<AxisKind>) -> Self {
        assert_eq!(axes.len(), shape.len(), "axes.len() must match shape.len()");
        let strides = compute_strides(&shape);
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self { data, shape, strides, axes: Some(axes), contrast_type: None }
    }

    /// Create tensor with a specified contrast type — the α axis from the
    /// Periodic Stack of Intelligence. Use this when you know the tensor
    /// represents a specific kind of contrast (spatial gradients, spectral
    /// decomposition, prediction errors, etc.).
    pub fn new_with_contrast(
        data: Vec<f64>,
        shape: Vec<usize>,
        contrast_type: ContrastType,
    ) -> Self {
        let strides = compute_strides(&shape);
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            data,
            shape,
            strides,
            axes: None,
            contrast_type: Some(contrast_type),
        }
    }

    /// Create tensor with both axis labels and contrast type — the fully
    /// annotated form. Use this for tensors the compiler should reason about
    /// structurally (e.g. known-shape inputs to attractor primitives).
    pub fn new_fully_annotated(
        data: Vec<f64>,
        shape: Vec<usize>,
        axes: Vec<AxisKind>,
        contrast_type: ContrastType,
    ) -> Self {
        assert_eq!(axes.len(), shape.len(), "axes.len() must match shape.len()");
        let strides = compute_strides(&shape);
        assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            data,
            shape,
            strides,
            axes: Some(axes),
            contrast_type: Some(contrast_type),
        }
    }

    /// Attach a contrast type to an existing tensor.
    pub fn with_contrast_type(mut self, contrast_type: ContrastType) -> Self {
        self.contrast_type = Some(contrast_type);
        self
    }

    /// Get the contrast type if set, otherwise `Unspecified`.
    pub fn contrast_type(&self) -> ContrastType {
        self.contrast_type.unwrap_or(ContrastType::Unspecified)
    }

    /// Whether this tensor has an explicit contrast type set.
    pub fn has_contrast_type(&self) -> bool {
        self.contrast_type.is_some()
    }

    /// Whether this tensor's contrast type is compatible with another tensor's
    /// for operations that combine them. Use this as a precondition before
    /// element-wise ops, matmul, attention, etc.
    ///
    /// Returns `true` if either tensor is unspecified or the types match.
    /// Returns `false` if the types are both specified and different.
    pub fn is_contrast_compatible_with(&self, other: &Tensor) -> bool {
        self.contrast_type().is_compatible_with(other.contrast_type())
    }

    /// Attach axis labels to an existing tensor. Panics if length mismatch.
    pub fn with_axes(mut self, axes: Vec<AxisKind>) -> Self {
        assert_eq!(axes.len(), self.shape.len());
        self.axes = Some(axes);
        self
    }

    /// Get the axis kind for dimension `i`. Returns `Unknown` if no labels set.
    pub fn axis(&self, i: usize) -> AxisKind {
        self.axes.as_ref()
            .and_then(|a| a.get(i).copied())
            .unwrap_or(AxisKind::Unknown)
    }

    /// Whether any axis is spatial ({X, Y, Z}).
    pub fn has_spatial_axes(&self) -> bool {
        self.axes.as_ref()
            .map(|a| a.iter().any(|k| k.is_spatial()))
            .unwrap_or(false)
    }

    /// How many axes are embarrassingly parallel (Batch, Channel, Camera, View).
    pub fn parallel_axes_count(&self) -> usize {
        self.axes.as_ref()
            .map(|a| a.iter().filter(|k| k.is_parallel()).count())
            .unwrap_or(0)
    }

    /// Create tensor of zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(vec![0.0; n], shape)
    }

    /// Create tensor of ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(vec![1.0; n], shape)
    }

    /// Create tensor filled with value.
    pub fn full(shape: Vec<usize>, value: f64) -> Self {
        let n: usize = shape.iter().product();
        Self::new(vec![value; n], shape)
    }

    /// Create identity matrix as tensor.
    pub fn eye(n: usize) -> Self {
        let mut data = vec![0.0; n * n];
        for i in 0..n { data[i * n + i] = 1.0; }
        Self::new(data, vec![n, n])
    }

    /// Arange: [start, stop) with step.
    pub fn arange(start: f64, stop: f64, step: f64) -> Self {
        let mut data = Vec::new();
        let mut v = start;
        while v < stop { data.push(v); v += step; }
        let n = data.len();
        Self::new(data, vec![n])
    }

    /// Linspace: n evenly spaced values from start to stop (inclusive).
    pub fn linspace(start: f64, stop: f64, n: usize) -> Self {
        if n <= 1 { return Self::new(vec![start], vec![1]); }
        let step = (stop - start) / (n - 1) as f64;
        let data: Vec<f64> = (0..n).map(|i| start + i as f64 * step).collect();
        Self::new(data, vec![n])
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Total number of elements.
    pub fn numel(&self) -> usize { self.data.len() }

    /// Get element at multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f64 {
        let flat = indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum::<usize>();
        self.data[flat]
    }

    /// Set element at multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: f64) {
        let flat = indices.iter().zip(&self.strides).map(|(i, s)| i * s).sum::<usize>();
        self.data[flat] = value;
    }

    /// Reshape (same total elements).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert_eq!(self.numel(), new_shape.iter().product::<usize>());
        Self::new(self.data.clone(), new_shape)
    }

    /// Flatten to 1D.
    pub fn flatten(&self) -> Self {
        Self::new(self.data.clone(), vec![self.numel()])
    }

    /// Transpose (swap last two axes for 2D+).
    pub fn transpose(&self) -> Self {
        if self.ndim() < 2 { return self.clone(); }
        let mut new_shape = self.shape.clone();
        let n = new_shape.len();
        new_shape.swap(n - 2, n - 1);
        let rows = self.shape[n - 2];
        let cols = self.shape[n - 1];
        let batch: usize = self.shape[..n - 2].iter().product();
        let mut data = vec![0.0; self.numel()];
        for b in 0..batch {
            let offset = b * rows * cols;
            for i in 0..rows {
                for j in 0..cols {
                    data[offset + j * rows + i] = self.data[offset + i * cols + j];
                }
            }
        }
        Self::new(data, new_shape)
    }

    /// Squeeze: remove dimensions of size 1.
    pub fn squeeze(&self) -> Self {
        let new_shape: Vec<usize> = self.shape.iter().filter(|&&s| s > 1).copied().collect();
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        Self::new(self.data.clone(), new_shape)
    }

    /// Unsqueeze: add dimension of size 1 at given axis.
    pub fn unsqueeze(&self, axis: usize) -> Self {
        let mut new_shape = self.shape.clone();
        new_shape.insert(axis, 1);
        Self::new(self.data.clone(), new_shape)
    }
}

// ═══════════════════════════════════════════════════════════════════
// ELEMENT-WISE OPERATIONS
// ═══════════════════════════════════════════════════════════════════

/// Element-wise addition.
pub fn tensor_add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Element-wise subtraction.
pub fn tensor_sub(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x - y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Element-wise multiplication (Hadamard).
pub fn tensor_mul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Element-wise division.
pub fn tensor_div(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x / y).collect();
    Tensor::new(data, a.shape.clone())
}

/// Scalar addition.
pub fn tensor_add_scalar(a: &Tensor, s: f64) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x + s).collect(), a.shape.clone())
}

/// Scalar multiplication.
pub fn tensor_mul_scalar(a: &Tensor, s: f64) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x * s).collect(), a.shape.clone())
}

/// Element-wise power.
pub fn tensor_pow(a: &Tensor, p: f64) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.powf(p)).collect(), a.shape.clone())
}

/// Element-wise sqrt.
pub fn tensor_sqrt(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.sqrt()).collect(), a.shape.clone())
}

/// Element-wise abs.
pub fn tensor_abs(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.abs()).collect(), a.shape.clone())
}

/// Element-wise exp.
pub fn tensor_exp(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.exp()).collect(), a.shape.clone())
}

/// Element-wise log.
pub fn tensor_log(a: &Tensor) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.ln()).collect(), a.shape.clone())
}

/// Element-wise max(a, b).
pub fn tensor_maximum(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape, b.shape);
    let data: Vec<f64> = a.data.iter().zip(&b.data).map(|(x, y)| x.max(*y)).collect();
    Tensor::new(data, a.shape.clone())
}

/// Element-wise clamp.
pub fn tensor_clamp(a: &Tensor, min: f64, max: f64) -> Tensor {
    Tensor::new(a.data.iter().map(|x| x.max(min).min(max)).collect(), a.shape.clone())
}

// ═══════════════════════════════════════════════════════════════════
// REDUCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Sum all elements.
pub fn tensor_sum(a: &Tensor) -> f64 { a.data.iter().sum() }

/// Mean of all elements.
pub fn tensor_mean(a: &Tensor) -> f64 { tensor_sum(a) / a.numel() as f64 }

/// Max element.
pub fn tensor_max(a: &Tensor) -> f64 { a.data.iter().copied().fold(f64::NEG_INFINITY, f64::max) }

/// Min element.
pub fn tensor_min(a: &Tensor) -> f64 { a.data.iter().copied().fold(f64::INFINITY, f64::min) }

/// Argmax (flat index).
pub fn tensor_argmax(a: &Tensor) -> usize {
    a.data.iter().enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap().0
}

/// Argmin (flat index).
pub fn tensor_argmin(a: &Tensor) -> usize {
    a.data.iter().enumerate()
        .min_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap().0
}

/// Sum along axis.
pub fn tensor_sum_axis(a: &Tensor, axis: usize) -> Tensor {
    let mut new_shape = a.shape.clone();
    let axis_size = new_shape.remove(axis);
    if new_shape.is_empty() { new_shape.push(1); }
    let new_numel: usize = new_shape.iter().product();
    let mut data = vec![0.0; new_numel];

    let outer: usize = a.shape[..axis].iter().product();
    let inner: usize = a.shape[axis + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            let mut sum = 0.0;
            for k in 0..axis_size {
                sum += a.data[o * axis_size * inner + k * inner + i];
            }
            data[o * inner + i] = sum;
        }
    }
    Tensor::new(data, new_shape)
}

/// Mean along axis.
pub fn tensor_mean_axis(a: &Tensor, axis: usize) -> Tensor {
    let s = tensor_sum_axis(a, axis);
    let n = a.shape[axis] as f64;
    Tensor::new(s.data.iter().map(|x| x / n).collect(), s.shape)
}

// ═══════════════════════════════════════════════════════════════════
// MATRIX OPERATIONS (2D tensors)
// ═══════════════════════════════════════════════════════════════════

/// Matrix multiply (2D tensors).
pub fn tensor_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2);
    assert_eq!(b.ndim(), 2);
    let m = a.shape[0];
    let k = a.shape[1];
    assert_eq!(k, b.shape[0]);
    let n = b.shape[1];
    let mut data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k { sum += a.data[i * k + l] * b.data[l * n + j]; }
            data[i * n + j] = sum;
        }
    }
    Tensor::new(data, vec![m, n])
}

/// Dot product (1D tensors).
pub fn tensor_dot(a: &Tensor, b: &Tensor) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| x * y).sum()
}

// ═══════════════════════════════════════════════════════════════════
// CONCATENATION & STACKING
// ═══════════════════════════════════════════════════════════════════

/// Concatenate tensors along axis.
pub fn tensor_concat(tensors: &[&Tensor], axis: usize) -> Tensor {
    let _ndim = tensors[0].ndim();
    let mut new_shape = tensors[0].shape.clone();
    new_shape[axis] = tensors.iter().map(|t| t.shape[axis]).sum();

    let outer: usize = new_shape[..axis].iter().product();
    let inner: usize = new_shape[axis + 1..].iter().product();
    let mut data = vec![0.0; new_shape.iter().product::<usize>()];

    let mut offset = 0;
    for t in tensors {
        let ax = t.shape[axis];
        for o in 0..outer {
            for k in 0..ax {
                for i in 0..inner {
                    data[o * new_shape[axis] * inner + (offset + k) * inner + i] =
                        t.data[o * ax * inner + k * inner + i];
                }
            }
        }
        offset += ax;
    }
    Tensor::new(data, new_shape)
}

/// Stack tensors along new axis.
pub fn tensor_stack(tensors: &[&Tensor], axis: usize) -> Tensor {
    let expanded: Vec<Tensor> = tensors.iter().map(|t| t.unsqueeze(axis)).collect();
    let refs: Vec<&Tensor> = expanded.iter().collect();
    tensor_concat(&refs, axis)
}

/// Where (conditional select): result[i] = if cond[i] then a[i] else b[i].
pub fn tensor_where(cond: &[bool], a: &Tensor, b: &Tensor) -> Tensor {
    let data: Vec<f64> = cond.iter().zip(a.data.iter().zip(&b.data))
        .map(|(&c, (x, y))| if c { *x } else { *y })
        .collect();
    Tensor::new(data, a.shape.clone())
}

// ═══════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_create() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::arange(0.0, 12.0, 1.0);
        let r = t.reshape(vec![3, 4]);
        assert_eq!(r.shape, vec![3, 4]);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tt = t.transpose();
        assert_eq!(tt.shape, vec![3, 2]);
        assert_eq!(tt.get(&[0, 0]), 1.0);
        assert_eq!(tt.get(&[0, 1]), 4.0);
    }

    #[test]
    fn test_elementwise() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = tensor_add(&a, &b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_axis() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s = tensor_sum_axis(&t, 1); // sum along columns
        assert_eq!(s.data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::eye(2);
        let c = tensor_matmul(&a, &b);
        assert_eq!(c.data, a.data);
    }

    #[test]
    fn test_concat() {
        let a = Tensor::new(vec![1.0, 2.0], vec![2]);
        let b = Tensor::new(vec![3.0, 4.0], vec![2]);
        let c = tensor_concat(&[&a, &b], 0);
        assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_linspace() {
        let t = Tensor::linspace(0.0, 1.0, 5);
        assert_eq!(t.data, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1]);
        let s = t.squeeze();
        assert_eq!(s.shape, vec![3]);
        let u = s.unsqueeze(0);
        assert_eq!(u.shape, vec![1, 3]);
    }

    // ── Axis label tests ─────────────────────────────────────────

    #[test]
    fn test_axis_kind_properties() {
        assert!(AxisKind::Batch.is_parallel());
        assert!(AxisKind::Channel.is_parallel());
        assert!(AxisKind::Camera.is_parallel());
        assert!(!AxisKind::Time.is_parallel());
        assert!(AxisKind::Time.is_sequential());
        assert!(!AxisKind::Batch.is_sequential());
        assert!(AxisKind::X.is_spatial());
        assert!(AxisKind::Y.is_spatial());
        assert!(AxisKind::Z.is_spatial());
        assert!(!AxisKind::Batch.is_spatial());
    }

    #[test]
    fn test_tensor_with_axes() {
        let t = Tensor::new(vec![1.0; 24], vec![2, 3, 4])
            .with_axes(vec![AxisKind::Batch, AxisKind::Y, AxisKind::X]);
        assert_eq!(t.axis(0), AxisKind::Batch);
        assert_eq!(t.axis(1), AxisKind::Y);
        assert_eq!(t.axis(2), AxisKind::X);
        assert!(t.has_spatial_axes());
        assert_eq!(t.parallel_axes_count(), 1); // only Batch
    }

    #[test]
    fn test_tensor_new_labeled() {
        let t = Tensor::new_labeled(
            vec![0.0; 8],
            vec![2, 2, 2],
            vec![AxisKind::X, AxisKind::Y, AxisKind::Z],
        );
        assert!(t.has_spatial_axes());
        assert_eq!(t.parallel_axes_count(), 0);
    }

    #[test]
    fn test_tensor_unlabeled_axes_are_unknown() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.axis(0), AxisKind::Unknown);
        assert_eq!(t.axis(1), AxisKind::Unknown);
        assert!(!t.has_spatial_axes());
    }

    #[test]
    fn test_reshape_drops_axes() {
        // Reshape changes dimension semantics — axes must be cleared.
        let t = Tensor::new(vec![1.0; 12], vec![3, 4])
            .with_axes(vec![AxisKind::Y, AxisKind::X]);
        let r = t.reshape(vec![4, 3]);
        assert!(r.axes.is_none(), "reshape should drop axes");
    }

    #[test]
    fn test_spatial_3d_tensor() {
        // A 64³ voxel grid labeled {X, Y, Z} — the spatial intelligence
        // use case: the cascade uses axis labels to distinguish "3D dense
        // voxel sample" (L4Communication) from "3D dense MatMul" (FloatMul).
        let voxels = Tensor::zeros(vec![64, 64, 64])
            .with_axes(vec![AxisKind::X, AxisKind::Y, AxisKind::Z]);
        assert!(voxels.has_spatial_axes());
        assert_eq!(voxels.parallel_axes_count(), 0);
        // All 3 spatial axes.
        assert!(voxels.axis(0).is_spatial());
        assert!(voxels.axis(1).is_spatial());
        assert!(voxels.axis(2).is_spatial());
    }

    #[test]
    fn test_multi_view_tensor() {
        // A batch of multi-view images: {Batch, Camera, Height, Width, Channel}
        let imgs = Tensor::zeros(vec![4, 6, 480, 640, 3])
            .with_axes(vec![
                AxisKind::Batch,
                AxisKind::Camera,
                AxisKind::Y,
                AxisKind::X,
                AxisKind::Channel,
            ]);
        assert!(imgs.has_spatial_axes());
        assert_eq!(imgs.parallel_axes_count(), 3); // Batch, Camera, Channel
    }

    // ── v10 ContrastType tests (step 6) ─────────────────────────

    #[test]
    fn contrast_type_sensory_cognitive_partition() {
        // Sensory types come from the environment
        assert!(ContrastType::Magnitude.is_sensory());
        assert!(ContrastType::Temporal.is_sensory());
        assert!(ContrastType::Spatial.is_sensory());
        assert!(ContrastType::Spectral.is_sensory());
        assert!(ContrastType::Structural.is_sensory());
        // Cognitive types require integration
        assert!(ContrastType::Relational.is_cognitive());
        assert!(ContrastType::Causal.is_cognitive());
        assert!(ContrastType::SelfReferential.is_cognitive());
        // Sensory and cognitive are disjoint
        assert!(!ContrastType::Magnitude.is_cognitive());
        assert!(!ContrastType::Causal.is_sensory());
        // Unspecified is neither
        assert!(!ContrastType::Unspecified.is_sensory());
        assert!(!ContrastType::Unspecified.is_cognitive());
    }

    #[test]
    fn contrast_type_compatibility_rules() {
        // Same type always compatible
        assert!(ContrastType::Spatial.is_compatible_with(ContrastType::Spatial));
        assert!(ContrastType::Temporal.is_compatible_with(ContrastType::Temporal));

        // Unspecified is compatible with everything
        assert!(ContrastType::Unspecified.is_compatible_with(ContrastType::Spatial));
        assert!(ContrastType::Spatial.is_compatible_with(ContrastType::Unspecified));

        // Different specified types are incompatible
        assert!(!ContrastType::Magnitude.is_compatible_with(ContrastType::Spatial));
        assert!(!ContrastType::Spatial.is_compatible_with(ContrastType::Spectral));
        assert!(!ContrastType::SelfReferential.is_compatible_with(ContrastType::Causal));
    }

    #[test]
    fn contrast_type_short_code_roundtrip() {
        let types = [
            ContrastType::Magnitude,
            ContrastType::Temporal,
            ContrastType::Spatial,
            ContrastType::Spectral,
            ContrastType::Structural,
            ContrastType::Relational,
            ContrastType::Causal,
            ContrastType::SelfReferential,
            ContrastType::Unspecified,
        ];
        for ct in &types {
            let code = ct.short_code();
            let parsed = ContrastType::from_short_code(code).unwrap();
            assert_eq!(*ct, parsed,
                "roundtrip failed for {:?} via code {:?}", ct, code);
        }
    }

    #[test]
    fn contrast_type_from_short_code_handles_invalid() {
        assert!(ContrastType::from_short_code("FOO").is_none());
        assert!(ContrastType::from_short_code("mag").is_none()); // lowercase not accepted
    }

    #[test]
    fn tensor_default_contrast_is_unspecified() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.contrast_type(), ContrastType::Unspecified);
        assert!(!t.has_contrast_type());
    }

    #[test]
    fn tensor_with_contrast_type() {
        let t = Tensor::zeros(vec![64, 64, 64])
            .with_contrast_type(ContrastType::Spatial);
        assert_eq!(t.contrast_type(), ContrastType::Spatial);
        assert!(t.has_contrast_type());
    }

    #[test]
    fn tensor_new_with_contrast() {
        let data = vec![0.0; 100];
        let t = Tensor::new_with_contrast(data, vec![10, 10], ContrastType::Structural);
        assert_eq!(t.contrast_type(), ContrastType::Structural);
        assert!(t.has_contrast_type());
        assert_eq!(t.shape, vec![10, 10]);
    }

    #[test]
    fn tensor_new_fully_annotated() {
        // A 4D image tensor: Batch, Channel, Height, Width — a spatial tensor
        let t = Tensor::new_fully_annotated(
            vec![0.0; 2 * 3 * 4 * 4],
            vec![2, 3, 4, 4],
            vec![AxisKind::Batch, AxisKind::Channel, AxisKind::Y, AxisKind::X],
            ContrastType::Spatial,
        );
        assert_eq!(t.axis(0), AxisKind::Batch);
        assert_eq!(t.axis(3), AxisKind::X);
        assert!(t.has_spatial_axes());
        assert_eq!(t.contrast_type(), ContrastType::Spatial);
    }

    #[test]
    fn tensor_compat_with_unspecified() {
        let a = Tensor::zeros(vec![3]).with_contrast_type(ContrastType::Magnitude);
        let b = Tensor::zeros(vec![3]); // Unspecified
        assert!(a.is_contrast_compatible_with(&b));
        assert!(b.is_contrast_compatible_with(&a));
    }

    #[test]
    fn tensor_compat_same_type() {
        let a = Tensor::zeros(vec![3]).with_contrast_type(ContrastType::Spatial);
        let b = Tensor::zeros(vec![3]).with_contrast_type(ContrastType::Spatial);
        assert!(a.is_contrast_compatible_with(&b));
    }

    #[test]
    fn tensor_incompat_different_types() {
        // You can't element-wise add a magnitude tensor and a causal tensor.
        let mag = Tensor::zeros(vec![3]).with_contrast_type(ContrastType::Magnitude);
        let caus = Tensor::zeros(vec![3]).with_contrast_type(ContrastType::Causal);
        assert!(!mag.is_contrast_compatible_with(&caus));
        assert!(!caus.is_contrast_compatible_with(&mag));
    }

    #[test]
    fn tensor_reshape_drops_contrast_type() {
        // Reshape changes the structure, so contrast type should be dropped
        // (via Self::new falling back to None). This is the safe default:
        // the caller re-applies the contrast type after reshape if still valid.
        let t = Tensor::zeros(vec![3, 4])
            .with_contrast_type(ContrastType::Spatial);
        let r = t.reshape(vec![4, 3]);
        // reshape uses Self::new which sets contrast_type: None
        assert_eq!(r.contrast_type(), ContrastType::Unspecified);
    }

    #[test]
    fn spatial_tensor_with_attractor_coordinates() {
        // An image tensor that the compiler can route to a spatial template
        // match (attractor #10: SPAT × L2 × D2 × E1)
        let image = Tensor::zeros(vec![1, 3, 224, 224])
            .with_axes(vec![
                AxisKind::Batch,
                AxisKind::Channel,
                AxisKind::Y,
                AxisKind::X,
            ])
            .with_contrast_type(ContrastType::Spatial);

        assert!(image.has_spatial_axes());
        assert_eq!(image.contrast_type(), ContrastType::Spatial);
        assert!(image.contrast_type().is_sensory());
        assert!(!image.contrast_type().is_cognitive());
        // Compatible with another spatial tensor (e.g., a stored template)
        let template = Tensor::zeros(vec![3, 224, 224])
            .with_contrast_type(ContrastType::Spatial);
        assert!(image.is_contrast_compatible_with(&template));
    }

    #[test]
    fn spectral_tensor_from_fft_output() {
        // A spectrogram (output of FFT on a time-domain signal) — SPEC contrast.
        // This is what attractor #24 ParallelSpectralDecomposition produces.
        let spectrogram = Tensor::zeros(vec![1024, 512])
            .with_axes(vec![AxisKind::Time, AxisKind::Channel])
            .with_contrast_type(ContrastType::Spectral);

        assert_eq!(spectrogram.contrast_type(), ContrastType::Spectral);
        assert_eq!(spectrogram.contrast_type().short_code(), "SPEC");
        // Not compatible with a spatial tensor
        let image = Tensor::zeros(vec![512, 512])
            .with_contrast_type(ContrastType::Spatial);
        assert!(!spectrogram.is_contrast_compatible_with(&image));
    }

    #[test]
    fn self_referential_tensor_for_prediction_error() {
        // A prediction error tensor — SELF contrast, produced by attractor #7
        // ForwardModelPredictionError.
        let prediction_error = Tensor::zeros(vec![128])
            .with_contrast_type(ContrastType::SelfReferential);
        assert!(prediction_error.contrast_type().is_cognitive());
        assert_eq!(prediction_error.contrast_type().short_code(), "SELF");
    }

    // ── Periodic Tensor axis tests ──────────────────────────────

    #[test]
    fn test_energy_cost() {
        assert!(EnergyCost::L0Free.is_free());
        assert!(!EnergyCost::L2Erasure.is_free());
        assert_eq!(EnergyCost::L0Free.landauer_floor_pj(), 0.0);
        assert!(EnergyCost::L2Erasure.landauer_floor_pj() > 0.0);
        assert_eq!(EnergyCost::from_short_code("L0"), Some(EnergyCost::L0Free));
        assert_eq!(EnergyCost::from_short_code("L4"), Some(EnergyCost::L4Communication));
        assert!(EnergyCost::L0Free < EnergyCost::L4Communication);
    }

    #[test]
    fn test_integration_depth() {
        assert_eq!(IntegrationDepth::D0Raw.cascade_tier(), 0);
        assert_eq!(IntegrationDepth::D3Model.cascade_tier(), 3);
        assert_eq!(IntegrationDepth::from_short_code("D2"), Some(IntegrationDepth::D2Pattern));
        assert!(IntegrationDepth::D0Raw < IntegrationDepth::D3Model);
    }

    #[test]
    fn test_reference_type() {
        assert!(!ReferenceType::E0External.requires_memory());
        assert!(ReferenceType::E1Stored.requires_memory());
        assert!(!ReferenceType::E1Stored.requires_model());
        assert!(ReferenceType::E2Computed.requires_model());
        assert!(ReferenceType::E3Self.requires_memory());
        assert!(ReferenceType::E3Self.requires_model());
    }

    #[test]
    fn test_tensor_cell_notation() {
        let cell = TensorCell::new(
            ContrastType::Spatial, EnergyCost::L2Erasure,
            IntegrationDepth::D1Feature, ReferenceType::E0External,
        );
        assert_eq!(cell.notation(), "SPAT x L2 x D1 x E0");
        assert_eq!(cell.cascade_tier(), 1);
        assert!(!cell.is_free());
        assert!(!cell.requires_memory());

        let parsed = TensorCell::from_notation("CAUS x L2 x D3 x E2").unwrap();
        assert_eq!(parsed.alpha, ContrastType::Causal);
        assert_eq!(parsed.gamma, IntegrationDepth::D3Model);
        assert!(parsed.requires_model());
        assert_eq!(parsed.cascade_tier(), 3);
    }

    #[test]
    fn test_tensor_cell_display() {
        let cell = TensorCell::new(
            ContrastType::SelfReferential, EnergyCost::L2Erasure,
            IntegrationDepth::D2Pattern, ReferenceType::E2Computed,
        );
        assert_eq!(format!("{}", cell), "SELF x L2 x D2 x E2");
    }

    #[test]
    fn test_8_convergent_attractors() {
        let attractors = [
            "MAG x L2 x D0 x E1",   // #1 threshold crossing
            "SPAT x L2 x D1 x E0",  // #2 spatial gradient
            "TEMP x L2 x D1 x E3",  // #3 temporal adaptation
            "MAG x L2 x D1 x E3",   // #4 gain adaptation
            "STRUC x L2 x D2 x E1", // #5 template match
            "REL x L2 x D2 x E0",   // #6 competitive comparison
            "SELF x L2 x D2 x E2",  // #7 prediction error
            "SELF x L2 x D3 x E2",  // #8 deep model error
        ];
        for notation in &attractors {
            let cell = TensorCell::from_notation(notation).unwrap();
            assert_eq!(&cell.notation(), notation);
        }
        // All 8 are L2 (erasure cost)
        assert!(attractors.iter().all(|n| TensorCell::from_notation(n).unwrap().beta == EnergyCost::L2Erasure));
        // Tiers span D0 to D3
        assert_eq!(TensorCell::from_notation(attractors[0]).unwrap().cascade_tier(), 0);
        assert_eq!(TensorCell::from_notation(attractors[7]).unwrap().cascade_tier(), 3);
    }

    #[test]
    fn test_l0_operations_are_free() {
        let fft_cell = TensorCell::new(
            ContrastType::Spectral, EnergyCost::L0Free,
            IntegrationDepth::D1Feature, ReferenceType::E0External,
        );
        assert!(fft_cell.is_free());
        assert_eq!(fft_cell.landauer_floor_pj(), 0.0);
        assert_eq!(fft_cell.notation(), "SPEC x L0 x D1 x E0");
    }

    #[test]
    fn test_cell_requires_memory_and_model() {
        // Metacognition: SELF x L3 x D3 x E3 — requires both memory AND model
        let metacog = TensorCell::new(
            ContrastType::SelfReferential, EnergyCost::L3Sustained,
            IntegrationDepth::D3Model, ReferenceType::E3Self,
        );
        assert!(metacog.requires_memory());
        assert!(metacog.requires_model());
        assert!(!metacog.is_free());
        assert_eq!(metacog.cascade_tier(), 3);
    }
}
