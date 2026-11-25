//! RCE-JEPA — the predictive competence world-model (Joint Embedding Predictive
//! Architecture, LeCun 2022), made deterministic and gated.
//!
//! JEPA's thesis: learn by predicting **abstract representations**, not raw
//! tokens/pixels. A context encoder embeds what you see; a target encoder embeds
//! what you don't; a predictor maps context→target *in representation space*; an
//! energy scores the match. It is non-generative — it never emits tokens, it
//! predicts a latent and retrieves/verifies.
//!
//! RCE's representation space is the **canonical Form space**: the normalizer
//! collapses accidental surface (`3+2` and `2+3`, `2*3` and `6`) to one latent
//! point, so prediction generalizes across surface exactly as the semantic key
//! does — but JEPA adds a *predictor* that proposes a competence for a novel
//! intent from its latent structure, and an *energy* that abstains when the
//! intent is too novel. The prediction is a proposal: the replay gate verifies it
//! before use. JEPA is the learned proposer on the gear ladder, never trusted.
//!
//! Tractable prototype over the arithmetic Form fragment: the latent is a
//! deterministic feature vector (operator histogram, constant mass, variable
//! count, depth); the predictor is nearest-context in latent space; the energy is
//! latent distance. Full hierarchical JEPA (learned encoders, collapse prevention,
//! multi-scale world models) is the open research edge — documented, not faked.

use std::collections::HashMap;

use rce_spec_check::ast::Expr;

use crate::canonical::{emit, normalize};
use crate::lift::lift_expr;
use crate::repair::eval_expr;

/// A deterministic latent representation of a Form: the abstract space where
/// prediction happens. Accidental surface collapses (canonicalize first), so
/// equivalent behavior lands on one point and near-equivalent structure lands
/// nearby. This is the "representation, not tokens" invariant.
pub fn latent(src: &str) -> Result<Vec<f64>, String> {
    let e = normalize(lift_expr(src)?);
    let mut f = Feat::default();
    walk(&e, &mut f, 0.0);
    Ok(f.vec())
}

#[derive(Default)]
struct Feat {
    add: f64,
    sub: f64,
    mul: f64,
    div: f64,
    rem: f64,
    neg: f64,
    ints: f64,
    const_mass: f64,
    vars: f64,
    depth: f64,
}

impl Feat {
    fn vec(self) -> Vec<f64> {
        vec![
            self.add, self.sub, self.mul, self.div, self.rem, self.neg,
            self.ints, self.const_mass, self.vars, self.depth,
        ]
    }
}

fn walk(e: &Expr, f: &mut Feat, d: f64) {
    use Expr::*;
    f.depth = f.depth.max(d);
    match e {
        Int(n, _) => {
            f.ints += 1.0;
            f.const_mass += (*n as f64).abs();
        }
        Name(_, _) => f.vars += 1.0,
        Neg(a, _) => {
            f.neg += 1.0;
            walk(a, f, d + 1.0);
        }
        Bin(op, a, b, _) => {
            match op.as_str() {
                "+" => f.add += 1.0,
                "-" => f.sub += 1.0,
                "*" => f.mul += 1.0,
                "/" => f.div += 1.0,
                "%" => f.rem += 1.0,
                _ => {}
            }
            walk(a, f, d + 1.0);
            walk(b, f, d + 1.0);
        }
        _ => {}
    }
}

/// Euclidean distance in latent space (the energy between two representations).
pub fn energy(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// A verified observation: an intent embedded beside the competence that satisfied
/// it. The joint embedding — context (intent) and target (form) in one space.
struct Observation {
    context: Vec<f64>,
    form: String,
    target: Vec<f64>,
}

/// A JEPA prediction: the proposed competence, its latent energy, and how novel
/// the context was. The caller replay-verifies before use — JEPA never bypasses
/// the gate.
pub struct Prediction {
    /// The proposed canonical competence source.
    pub form: String,
    /// Latent energy of the predicted target vs the retrieved form (0 = exact).
    pub energy: f64,
    /// Distance from the query intent to the nearest observed context (novelty).
    pub novelty: f64,
}

/// The predictive competence world-model. Learns the joint embedding of intents
/// and verified competences; predicts a competence for a novel intent in latent
/// space; abstains (returns `None`) when the intent is too novel — non-generative,
/// energy-based, gated.
pub struct Jepa {
    obs: Vec<Observation>,
    /// Novelty above which the model abstains rather than hallucinate.
    pub abstain_energy: f64,
}

impl Jepa {
    /// An empty world-model with a default abstention threshold.
    pub fn new() -> Self {
        Jepa { obs: Vec::new(), abstain_energy: 6.0 }
    }

    /// Observe a verified (intent, competence) pair into the joint embedding.
    /// The competence is canonicalized; both are embedded into the shared latent
    /// space. Only verified experience is learned (the gate already ran upstream).
    pub fn observe(&mut self, intent: &str, form: &str) -> Result<(), String> {
        let context = latent(intent)?;
        let canon = emit(&normalize(lift_expr(form)?));
        let target = latent(&canon)?;
        self.obs.push(Observation { context, form: canon, target });
        Ok(())
    }

    /// Predict a competence for an intent: embed the context, find the nearest
    /// observed context (the predictor), take its target representation as the
    /// prediction, and retrieve the owned form nearest that predicted target.
    /// Abstains when the context is too novel (energy above threshold). The result
    /// is a *proposal* — replay-verify it with [`verify`] before use.
    pub fn predict(&self, intent: &str) -> Result<Option<Prediction>, String> {
        if self.obs.is_empty() {
            return Ok(None);
        }
        let context = latent(intent)?;
        // Nearest observed context (the context→target predictor, non-parametric).
        let nearest = self.obs.iter()
            .map(|o| (energy(&context, &o.context), o))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let Some((novelty, seed)) = nearest else { return Ok(None) };
        if novelty > self.abstain_energy {
            return Ok(None); // too novel: abstain, don't hallucinate
        }
        // Retrieve the owned form nearest the predicted target representation.
        let best = self.obs.iter()
            .map(|o| (energy(&seed.target, &o.target), &o.form))
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let Some((e, form)) = best else { return Ok(None) };
        Ok(Some(Prediction { form: form.clone(), energy: e, novelty }))
    }

    /// Number of verified observations in the joint embedding.
    pub fn len(&self) -> usize {
        self.obs.len()
    }

    /// Whether nothing has been observed yet.
    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }
}

impl Default for Jepa {
    fn default() -> Self {
        Self::new()
    }
}

/// The gate for the arithmetic fragment: a proposed competence satisfies an intent
/// iff it replays to the intent's value. JEPA proposes; this disposes. A prediction
/// that fails here is refused, never used — the model is never trusted.
pub fn verify(intent: &str, form: &str) -> bool {
    let Ok(i) = lift_expr(intent) else { return false };
    let Ok(f) = lift_expr(form) else { return false };
    match (eval_expr(&i), eval_expr(&f)) {
        (Some(a), Some(b)) => a == b,
        _ => false,
    }
}

/// Predict-and-verify: JEPA proposes a competence and the gate replays it. Returns
/// the verified form, or `None` if the model abstained or the proposal failed the
/// gate. This is the learned proposer on the gear ladder — competence-before-model,
/// with the model predicting in representation space and the gate keeping trust.
pub fn predict_verified(model: &Jepa, intent: &str) -> Option<String> {
    let p = model.predict(intent).ok()??;
    if verify(intent, &p.form) {
        Some(p.form)
    } else {
        None
    }
}

/// A map of distinct latent points (for diagnostics): how many accidental surfaces
/// collapsed to one representation. Demonstrates the representation-space collapse.
pub fn latent_collapse(surfaces: &[&str]) -> HashMap<String, Vec<String>> {
    let mut by: HashMap<String, Vec<String>> = HashMap::new();
    for s in surfaces {
        if let Ok(l) = latent(s) {
            let key = l.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",");
            by.entry(key).or_default().push(s.to_string());
        }
    }
    by
}