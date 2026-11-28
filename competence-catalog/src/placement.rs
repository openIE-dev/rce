//! Living optimization — capability is a lookup; placement is derived, not flagged.
//!
//! The curse of ignorance: today peak performance depends on a human *knowing* the flags
//! (`-mfma`, `-march=native`, …). RCE removes it: what a target has is a **lookup** (CPUID /
//! ARM ID regs / sysctl), and the optimal primitive use is a deterministic function of
//! (operation, capability, per-primitive cost). Placement = argmin joules, re-evaluated as
//! hardware changes — living software.
//!
//! Cost is Horowitz-form `E ≈ a·flops + b·bytes + c` with `b ≫ a` (data movement dominates).
//! Coefficients here are a **modelled** (est) table, labeled as such; a measured calibration
//! replaces the prior when available (flowg's placement pass). The capability table is data,
//! like the emission LUTs — adding a fabric is adding a row.

/// A compute primitive from the bounded floor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prim {
    Add,
    Mul,
    Fma, // fused multiply-add: one primitive where the fabric has it
    Load,
    Store,
}

impl Prim {
    /// (flops, bytes moved) for the primitive — the work the cost model prices.
    pub fn work(self) -> (f64, f64) {
        match self {
            Prim::Add => (1.0, 0.0),
            Prim::Mul => (1.0, 0.0),
            Prim::Fma => (2.0, 0.0),
            Prim::Load => (0.0, 8.0),
            Prim::Store => (0.0, 8.0),
        }
    }
}

/// A backend's Horowitz-form cost coefficients (modelled prior, est).
#[derive(Clone, Copy)]
pub struct CostModel {
    pub a: f64, // per flop
    pub b: f64, // per byte (b ≫ a)
    pub c: f64, // fixed dispatch
}

impl CostModel {
    /// Estimated microjoules for a primitive on this backend.
    pub fn energy(&self, p: Prim) -> f64 {
        let (flops, bytes) = p.work();
        self.a * flops + self.b * bytes + self.c
    }
}

/// A target fabric: its capability (a lookup result) + its cost model.
pub struct Target {
    pub name: &'static str,
    /// Whether the fabric has a fused multiply-add primitive (CPUID / FEAT_FHM lookup).
    pub has_fma: bool,
    pub model: CostModel,
}

/// The capability/cost table — data, not flags. Adding a fabric is adding a row.
/// `has_fma` mirrors what a capability query reports (this M5 Max: FEAT_FHM=1).
pub fn builtin_targets() -> Vec<Target> {
    vec![
        Target { name: "cpu-scalar", has_fma: false, model: CostModel { a: 1.0, b: 100.0, c: 1.0 } },
        Target { name: "cpu-fma", has_fma: true, model: CostModel { a: 1.0, b: 100.0, c: 1.0 } },
        Target { name: "gpu-simt", has_fma: true, model: CostModel { a: 0.1, b: 10.0, c: 0.5 } },
        Target { name: "amx-tensor", has_fma: true, model: CostModel { a: 0.05, b: 5.0, c: 0.5 } },
    ]
}

/// A placement plan for `a*b + c`: the primitive sequence + its estimated cost.
pub struct Plan {
    pub ops: Vec<Prim>,
    pub est_uj: f64,
}

/// Derive the plan for `a*b+c` on a target: use the single FMA primitive **iff the capability
/// lookup says the fabric has it**, else Mul+Add. No human flag — the carve-out is derived.
pub fn plan_muladd(t: &Target) -> Plan {
    let ops = if t.has_fma { vec![Prim::Fma] } else { vec![Prim::Mul, Prim::Add] };
    let est_uj = ops.iter().map(|&p| t.model.energy(p)).sum();
    Plan { ops, est_uj }
}

/// Living placement: pick the target (backend) minimizing the derived cost for `a*b+c`.
/// This is `place(op) = argmin_backend Ê(op, backend)` — re-runnable as capability changes.
pub fn place_muladd(targets: &[Target]) -> Option<(&Target, Plan)> {
    targets.iter().map(|t| (t, plan_muladd(t)))
        .min_by(|x, y| x.1.est_uj.partial_cmp(&y.1.est_uj).unwrap_or(std::cmp::Ordering::Equal))
}

/// The saving from deriving FMA vs the ignorant mul+add on the same fma-capable fabric —
/// the waste the curse of ignorance leaves on the table, made measurable.
pub fn fma_saving(t: &Target) -> f64 {
    if !t.has_fma {
        return 0.0;
    }
    let ignorant = t.model.energy(Prim::Mul) + t.model.energy(Prim::Add);
    let derived = t.model.energy(Prim::Fma);
    ignorant - derived
}

/// Numeric determinism under placement (handoff open question): lowest-joule choice and
/// bit-for-bit replay pull against each other. Policy: **logged placement**. The first run chooses
/// the min-joule backend and records it; replay reuses the logged backend, so the trace is
/// bit-identical even if costs or hardware change. (Alternatives: pinned numerics, or a declared
/// tolerance; logged placement keeps identity exact.)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlacementLog {
    pub choices: std::collections::BTreeMap<String, String>,
}

impl PlacementLog {
    /// First run: choose min-joule and log it. Replay: return the logged backend unchanged.
    pub fn place(&mut self, op: &str, targets: &[Target]) -> Option<String> {
        if let Some(b) = self.choices.get(op) {
            return Some(b.clone());
        }
        let (t, _) = place_muladd(targets)?;
        self.choices.insert(op.to_string(), t.name.to_string());
        Some(t.name.to_string())
    }

    /// Replay never re-chooses: the logged backend is returned regardless of new costs.
    pub fn replay(&self, op: &str) -> Option<&str> {
        self.choices.get(op).map(|s| s.as_str())
    }
}