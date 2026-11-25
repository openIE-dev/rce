//! Trained-proposer seed — a calibrated energy scorer fitted on gate receipts.
//!
//! The full FormJEPA/FormEBM (latent predictive proposer at scale) is the open research frontier.
//! This is the buildable seed of the *trained* inverse: a small logistic energy scorer over Form
//! features, fitted by gradient descent on (features, admitted) gate receipts, that ranks candidate
//! Forms before the gate. It is a proposer only — the gate still decides. This demonstrates the
//! training loop (receipts → calibrated ranker) that scales to the JEPA/EBM architectures.

use rce_spec_check::ast::Expr;

/// Deterministic Form features (the proposer's input representation).
pub fn features(e: &Expr) -> Vec<f64> {
    fn walk(e: &Expr, n: &mut f64, bin: &mut f64, neg: &mut f64, d: f64, maxd: &mut f64) {
        use Expr::*;
        *n += 1.0;
        *maxd = maxd.max(d);
        match e {
            Neg(a, _) => { *neg += 1.0; walk(a, n, bin, neg, d + 1.0, maxd); }
            Bin(_, a, b, _) => { *bin += 1.0; walk(a, n, bin, neg, d + 1.0, maxd); walk(b, n, bin, neg, d + 1.0, maxd); }
            _ => {}
        }
    }
    let (mut n, mut bin, mut neg, mut maxd) = (0.0, 0.0, 0.0, 0.0);
    walk(e, &mut n, &mut bin, &mut neg, 0.0, &mut maxd);
    vec![n, bin, neg, maxd]
}

fn sigmoid(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }

/// A calibrated logistic energy scorer (the trained proposer seed).
pub struct Scorer {
    pub w: Vec<f64>,
    pub b: f64,
}

impl Scorer {
    /// Fit on (features, admitted) receipts by batch gradient descent (logistic loss).
    pub fn train(samples: &[(Vec<f64>, bool)], epochs: usize, lr: f64) -> Scorer {
        let dim = samples.first().map(|(x, _)| x.len()).unwrap_or(0);
        let mut w = vec![0.0; dim];
        let mut b = 0.0;
        for _ in 0..epochs {
            let mut gw = vec![0.0; dim];
            let mut gb = 0.0;
            for (x, y) in samples {
                let z: f64 = w.iter().zip(x).map(|(a, c)| a * c).sum::<f64>() + b;
                let err = sigmoid(z) - (*y as i32 as f64);
                for j in 0..dim { gw[j] += err * x[j]; }
                gb += err;
            }
            let n = samples.len().max(1) as f64;
            for j in 0..dim { w[j] -= lr * gw[j] / n; }
            b -= lr * gb / n;
        }
        Scorer { w, b }
    }

    /// P(admitted) for a candidate — ranks before the gate; never replaces it.
    pub fn score(&self, x: &[f64]) -> f64 {
        sigmoid(self.w.iter().zip(x).map(|(a, c)| a * c).sum::<f64>() + self.b)
    }
}

/// A JEPA-style latent predictor seed: maps intent features → Form features in representation
/// space (predict the target summary, not tokens). Trained by gradient descent on verified
/// (intent, form) pairs; energy = distance to a candidate. The full hierarchical JEPA is the open
/// frontier; this is the buildable linear seed of the learned inverse.
pub struct LatentPredictor {
    pub w: Vec<Vec<f64>>,
    pub b: Vec<f64>,
}

impl LatentPredictor {
    pub fn train(pairs: &[(Vec<f64>, Vec<f64>)], epochs: usize, lr: f64) -> LatentPredictor {
        let dx = pairs.first().map(|(x, _)| x.len()).unwrap_or(0);
        let dy = pairs.first().map(|(_, y)| y.len()).unwrap_or(0);
        let mut w = vec![vec![0.0; dx]; dy];
        let mut b = vec![0.0; dy];
        for _ in 0..epochs {
            let mut gw = vec![vec![0.0; dx]; dy];
            let mut gb = vec![0.0; dy];
            for (x, y) in pairs {
                for j in 0..dy {
                    let pred: f64 = w[j].iter().zip(x).map(|(a, c)| a * c).sum::<f64>() + b[j];
                    let err = pred - y[j];
                    for k in 0..dx { gw[j][k] += err * x[k]; }
                    gb[j] += err;
                }
            }
            let n = pairs.len().max(1) as f64;
            for j in 0..dy {
                for k in 0..dx { w[j][k] -= lr * gw[j][k] / n; }
                b[j] -= lr * gb[j] / n;
            }
        }
        LatentPredictor { w, b }
    }

    /// Predict the target (Form) representation from the intent representation.
    pub fn predict(&self, x: &[f64]) -> Vec<f64> {
        (0..self.b.len()).map(|j| self.w[j].iter().zip(x).map(|(a, c)| a * c).sum::<f64>() + self.b[j]).collect()
    }

    /// Latent energy of a candidate against the predicted target (lower = better fit).
    pub fn energy(&self, x: &[f64], y: &[f64]) -> f64 {
        let p = self.predict(x);
        p.iter().zip(y).map(|(a, c)| (a - c) * (a - c)).sum::<f64>()
    }
}