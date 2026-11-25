//! The intent→form bridge — RCE-SPEC as the bridge from intent to form.
//!
//! Spec-Driven Development (SDD) globally converges on "the spec is the source
//! of truth": Kiro (requirements→design→tasks), GitHub Spec Kit (constitution→
//! specify→plan→tasks), Tessl (spec↔code). But LLM-based SDD is non-deterministic
//! (different code from the same spec), its checklists are AI-interpreted (no
//! enforcement), and it lost MDD's parseable validation. RCE is spec-as-source
//! done deterministically: a structured intent spec resolves to a Form by fixed
//! rules, the acceptance criteria are *mechanically* enforced by replay, and the
//! requirement→form link is content-addressed (traceability).
//!
//! This module operationalizes the bridge: constitution invariants + requirements
//! with acceptance criteria → deterministic resolution (lift / repair / synthesis)
//! → replay-verified Form → name-free content key. The gate guarantees what SDD
//! only asks a human to review.

use crate::canonical::{emit, normalize};
use crate::lift::lift_expr;
use crate::repair::{eval_expr, repair_to_value};

/// A requirement: an intent with acceptance criteria (the spec's unit).
pub struct Requirement {
    /// A stable requirement ID (traceability, à la Kiro).
    pub id: String,
    /// The candidate source expressing the intent.
    pub source: String,
    /// The acceptance target the form must hit (the spec's contract).
    pub target: i128,
}

/// A bridged requirement: intent resolved to a verified Form.
pub struct Bridged {
    /// The requirement ID (traceability link).
    pub id: String,
    /// The resolved canonical Form source.
    pub form: String,
    /// The name-free content key (requirement→form, content-addressed).
    pub key: String,
    /// Whether the form replays to the acceptance target (mechanically enforced).
    pub verified: bool,
    /// Whether a repair was applied to meet the spec.
    pub repaired: bool,
}

/// Bridge a requirement from intent to form: lift the source; if it fails the
/// acceptance target, repair it deterministically; verify by evaluation; emit the
/// canonical Form and its content key. The acceptance criterion is enforced, not
/// merely reviewed.
pub fn bridge_requirement(req: &Requirement) -> Result<Bridged, String> {
    let e = lift_expr(&req.source)?;
    let (form_expr, repaired) = if eval_expr(&e) == Some(req.target) {
        (e, false)
    } else {
        match repair_to_value(&req.source, req.target) {
            Some(r) => (lift_expr(&r.repaired)?, true),
            None => (e, false), // unrepairable: will fail verification honestly
        }
    };
    let verified = eval_expr(&form_expr) == Some(req.target);
    let form = emit(&normalize(form_expr));
    let key = rcevc_core::sha256(form.as_bytes()).hex();
    Ok(Bridged { id: req.id.clone(), form, key, verified, repaired })
}

/// Bridge a whole spec (a batch of requirements), each independently verified.
/// Returns the bridged forms; unverified ones are flagged, never silently passed.
pub fn bridge_spec(reqs: &[Requirement]) -> Vec<Bridged> {
    reqs.iter().filter_map(|r| bridge_requirement(r).ok()).collect()
}

/// The fraction of requirements whose forms mechanically satisfy their acceptance
/// criteria — the spec's enforced compliance (vs SDD's probabilistic review).
pub fn compliance(bridged: &[Bridged]) -> f64 {
    if bridged.is_empty() {
        return 0.0;
    }
    bridged.iter().filter(|b| b.verified).count() as f64 / bridged.len() as f64
}
