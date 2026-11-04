//! The E_sim simulator.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Telemetry the IDE has before an action runs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Features {
    pub form_nodes: u64,
    pub checker_passes: u64,
    pub wasi_fuel: u64,
    pub gear: String,
    pub model: Option<String>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub net_bytes: u64,
}

/// Names of the numeric terms, in coefficient order. Term 0 is the intercept.
pub const TERMS: [&str; 7] = [
    "intercept",
    "form_nodes",
    "checker_passes",
    "wasi_fuel",
    "tokens_in",
    "tokens_out",
    "net_bytes",
];
pub const N: usize = TERMS.len();

impl Features {
    /// Gear and model select the coefficient set (they are categorical).
    pub fn key(&self) -> String {
        format!("{}/{}", self.gear, self.model.as_deref().unwrap_or("-"))
    }
    pub fn row(&self) -> [f64; N] {
        [
            1.0,
            self.form_nodes as f64,
            self.checker_passes as f64,
            self.wasi_fuel as f64,
            self.tokens_in as f64,
            self.tokens_out as f64,
            self.net_bytes as f64,
        ]
    }
}

/// A prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub joules: f64,
    /// Residual standard deviation of the fit, joules (for the anomaly test).
    pub sigma: f64,
}

pub trait Simulator {
    /// `None` when there is no calibration for this gear and model.
    fn predict(&self, f: &Features) -> Option<Prediction>;
}

/// Fitted coefficients for one gear/model key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fit {
    pub coef: [f64; N],
    pub sigma: f64,
    pub samples: usize,
}

/// Linear simulator: E_sim = c . row(features), one coefficient set per key.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LinearSim {
    pub fits: BTreeMap<String, Fit>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CalibError {
    #[error("key {0}: no samples")]
    Empty(String),
    #[error("key {0}: a measured joule value is not finite or is negative")]
    BadSample(String),
}

impl Simulator for LinearSim {
    fn predict(&self, f: &Features) -> Option<Prediction> {
        let fit = self.fits.get(&f.key())?;
        let r = f.row();
        let j: f64 = fit.coef.iter().zip(r).map(|(c, x)| c * x).sum();
        Some(Prediction {
            joules: j.max(0.0),
            sigma: fit.sigma,
        })
    }
}

impl LinearSim {
    /// Fit every key present in `samples` (features, measured joules) by
    /// least squares. Columns that never vary for a key (all zero, or
    /// collinear) are held at zero by a tiny ridge term, so the fit is
    /// determined even with few samples. Replaces earlier fits for those keys.
    pub fn calibrate(&mut self, samples: &[(Features, f64)]) -> Result<(), CalibError> {
        let mut by_key: BTreeMap<String, Vec<(&Features, f64)>> = BTreeMap::new();
        for (f, j) in samples {
            if !j.is_finite() || *j < 0.0 {
                return Err(CalibError::BadSample(f.key()));
            }
            by_key.entry(f.key()).or_default().push((f, *j));
        }
        for (k, rows) in by_key {
            if rows.is_empty() {
                return Err(CalibError::Empty(k));
            }
            let coef = least_squares(&rows.iter().map(|(f, j)| (f.row(), *j)).collect::<Vec<_>>());
            let n = rows.len();
            let sse: f64 = rows
                .iter()
                .map(|(f, j)| {
                    let p: f64 = coef.iter().zip(f.row()).map(|(c, x)| c * x).sum();
                    (j - p).powi(2)
                })
                .sum();
            let used = coef.iter().filter(|c| c.abs() > 0.0).count().max(1);
            let dof = n.saturating_sub(used).max(1);
            self.fits.insert(
                k,
                Fit {
                    coef,
                    sigma: (sse / dof as f64).sqrt(),
                    samples: n,
                },
            );
        }
        Ok(())
    }
}

/// Solve min ||X c - y||^2 with columns scaled to unit RMS, a 1e-9 ridge on
/// the scaled normal equations, and Cholesky. Columns that are zero in every
/// row get coefficient 0.
pub fn least_squares(rows: &[([f64; N], f64)]) -> [f64; N] {
    let mut scale = [0.0f64; N];
    for (x, _) in rows {
        for i in 0..N {
            scale[i] += x[i] * x[i];
        }
    }
    let n = rows.len().max(1) as f64;
    let active: Vec<usize> = (0..N).filter(|&i| scale[i] > 0.0).collect();
    for s in scale.iter_mut() {
        *s = (*s / n).sqrt();
    }
    let m = active.len();
    let mut a = vec![vec![0.0f64; m]; m];
    let mut b = vec![0.0f64; m];
    for (x, y) in rows {
        let z: Vec<f64> = active.iter().map(|&i| x[i] / scale[i]).collect();
        for p in 0..m {
            b[p] += z[p] * y;
            for q in 0..m {
                a[p][q] += z[p] * z[q];
            }
        }
    }
    let tr: f64 = (0..m).map(|i| a[i][i]).sum::<f64>().max(1.0);
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += 1e-9 * tr;
    }
    let sol = cholesky_solve(a, b);
    let mut c = [0.0f64; N];
    for (p, &i) in active.iter().enumerate() {
        c[i] = sol[p] / scale[i];
    }
    c
}

fn cholesky_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let m = b.len();
    for j in 0..m {
        let d = a[j][j] - a[j][..j].iter().map(|x| x * x).sum::<f64>();
        let d = d.max(1e-300).sqrt();
        a[j][j] = d;
        for i in j + 1..m {
            let s = a[i][j] - (0..j).map(|k| a[i][k] * a[j][k]).sum::<f64>();
            a[i][j] = s / d;
        }
    }
    for i in 0..m {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i][k] * b[k];
        }
        b[i] = s / a[i][i];
    }
    for i in (0..m).rev() {
        let mut s = b[i];
        for k in i + 1..m {
            s -= a[k][i] * b[k];
        }
        b[i] = s / a[i][i];
    }
    b
}
