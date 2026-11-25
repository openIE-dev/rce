//! Value center — the Law of Coding value term: user benefit per joule.
//!
//! Post-autonomy need #13 (rce-post-coding-needs.md): once code is abundant, the
//! scarce question is *what* to build. Rank candidate work by benefit/joule so the
//! deterministic fab spends its floor on the highest-value competence first.
//! Benefit is a boundary input (a stakeholder weight, never inferred); joules come
//! from the cost signal (energy/fleet). Both are validated here; ranking is total
//! and deterministic.

/// A candidate unit of work with its stakeholder benefit weight and joule estimate.
#[derive(Clone, Debug)]
pub struct Candidate {
    pub id: String,
    /// User benefit weight (finite, >= 0). Boundary input, never inferred.
    pub benefit: f64,
    /// Estimated cost, microjoules (finite, > 0).
    pub joules: f64,
}

/// A ranked candidate with its value receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    pub id: String,
    pub benefit: f64,
    pub joules: f64,
    /// benefit / joules — the Law of Coding value term.
    pub value: f64,
}

/// The value term: user benefit per joule. Refuses non-finite or negative benefit
/// and non-positive or non-finite cost — a zero-joule claim is a measurement
/// error, not infinite value.
pub fn value(benefit: f64, joules: f64) -> Result<f64, String> {
    if !benefit.is_finite() || benefit < 0.0 {
        return Err("benefit must be finite and >= 0".into());
    }
    if !joules.is_finite() || joules <= 0.0 {
        return Err("joules must be finite and > 0".into());
    }
    Ok(benefit / joules)
}

/// Rank candidates by value descending; ties break by id ascending so the order
/// is deterministic (replay-stable). Any invalid candidate refuses the batch.
pub fn rank(candidates: &[Candidate]) -> Result<Vec<Ranked>, String> {
    let mut out = Vec::with_capacity(candidates.len());
    for c in candidates {
        out.push(Ranked {
            id: c.id.clone(),
            benefit: c.benefit,
            joules: c.joules,
            value: value(c.benefit, c.joules)?,
        });
    }
    out.sort_by(|a, b| b.value.total_cmp(&a.value).then_with(|| a.id.cmp(&b.id)));
    Ok(out)
}
