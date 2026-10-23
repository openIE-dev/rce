//! Calibration: run a fixed set of workloads under every discovered meter
//! and fit the simulator's coefficients to what the meters measured.
//!
//! Each workload run is one window: every meter is sampled right before and
//! right after, and the run reports the features it actually produced (the
//! same features receipts carry). Runs are interleaved round-robin across
//! `reps` passes so drift (thermal, background load) spreads over all
//! workloads instead of biasing one. A meter that errors or is stale for a
//! window contributes no sample for it; nothing is ever recorded as 0 J.
//!
//! The meters see the whole package / rail / system, so measured joules
//! include idle draw over the window. That is also what an action's receipt
//! measures, so the fit predicts comparable numbers; the intercept and the
//! idle workload absorb the floor. Terms that no workload varies stay at 0
//! (see [`crate::sim::least_squares`]) and are listed as not calibrated.
use crate::meter::{Meter, MeterError};
use crate::sim::{Features, LinearSim, N, Simulator, TERMS};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// One calibration workload.
pub trait Workload {
    fn name(&self) -> String;
    /// Do the work once; return the features it produced.
    fn run(&mut self) -> Result<Features, String>;
}

#[derive(Debug, Clone)]
pub struct Plan {
    /// Passes over the workload list.
    pub reps: usize,
    /// Pause before each window (lets counters tick and the core settle).
    pub settle: Duration,
}

impl Default for Plan {
    fn default() -> Self {
        Plan {
            reps: 3,
            settle: Duration::from_millis(50),
        }
    }
}

/// One measured window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Run {
    pub workload: String,
    pub rep: usize,
    pub features: Features,
    pub seconds: f64,
    pub joules: f64,
    /// Prediction of the final fit for this run (filled after fitting).
    pub predicted: Option<f64>,
}

/// Calibration for one meter. Loads as a [`LinearSim`] (extra fields are
/// ignored), so the file can be passed straight to `rce-mcp --calibration`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeterCalibration {
    pub meter: String,
    #[serde(flatten)]
    pub sim: LinearSim,
    /// Terms some workload varied, per key; the others are held at 0.
    pub calibrated_terms: std::collections::BTreeMap<String, Vec<String>>,
    /// Coefficient of determination of the fit over all runs.
    pub r2: f64,
    pub runs: Vec<Run>,
    /// Windows this meter could not measure, with why.
    pub skipped: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub version: u32,
    pub plan_reps: usize,
    pub workloads: Vec<String>,
    pub meters: Vec<MeterCalibration>,
    /// Meters that produced no usable window at all.
    pub unusable: Vec<(String, String)>,
}

#[derive(Debug, thiserror::Error)]
pub enum CalibrateError {
    #[error("no energy meter is available here (see `rce-ide telemetry probe`)")]
    NoMeter,
    #[error("workload {0}: {1}")]
    Workload(String, String),
    #[error("no meter measured any window: {0:?}")]
    NothingMeasured(Vec<(String, String)>),
    #[error("fit: {0}")]
    Fit(String),
}

fn delta(a: &Result<f64, String>, b: &Result<f64, String>) -> Result<f64, String> {
    match (a, b) {
        (Ok(a), Ok(b)) if b >= a => Ok(b - a),
        (Ok(_), Ok(_)) => Err("counter went backwards".into()),
        (Err(e), _) | (_, Err(e)) => Err(e.clone()),
    }
}

fn read(m: &mut dyn Meter) -> Result<f64, String> {
    m.sample().map(|s| s.joules).map_err(|e| match e {
        MeterError::Stale(..) => format!("stale: {e}"),
        e => e.to_string(),
    })
}

/// Runs measured by one meter, and windows it skipped (workload, why).
type PerMeter = (Vec<Run>, Vec<(String, String)>);

/// Run `workloads` `plan.reps` times under `meters` and fit one simulator
/// per meter.
pub fn calibrate(
    meters: &mut [Box<dyn Meter>],
    workloads: &mut [Box<dyn Workload>],
    plan: &Plan,
) -> Result<Calibration, CalibrateError> {
    if meters.is_empty() {
        return Err(CalibrateError::NoMeter);
    }
    let names: Vec<String> = workloads.iter().map(|w| w.name()).collect();
    let mut per: Vec<PerMeter> = vec![(vec![], vec![]); meters.len()];
    // Prime the meters (polled ones need a first reading).
    for m in meters.iter_mut() {
        let _ = m.sample();
    }
    for rep in 0..plan.reps {
        for (wi, w) in workloads.iter_mut().enumerate() {
            std::thread::sleep(plan.settle);
            let before: Vec<_> = meters.iter_mut().map(|m| read(m.as_mut())).collect();
            let t0 = std::time::Instant::now();
            let f = w
                .run()
                .map_err(|e| CalibrateError::Workload(names[wi].clone(), e))?;
            let secs = t0.elapsed().as_secs_f64();
            let after: Vec<_> = meters.iter_mut().map(|m| read(m.as_mut())).collect();
            for (i, (a, b)) in before.iter().zip(&after).enumerate() {
                match delta(a, b) {
                    Ok(j) if j.is_finite() => per[i].0.push(Run {
                        workload: names[wi].clone(),
                        rep,
                        features: f.clone(),
                        seconds: secs,
                        joules: j,
                        predicted: None,
                    }),
                    Ok(_) => per[i].1.push((names[wi].clone(), "not finite".into())),
                    Err(e) => per[i].1.push((names[wi].clone(), e)),
                }
            }
        }
    }
    let mut out = Calibration {
        version: 1,
        plan_reps: plan.reps,
        workloads: names,
        meters: vec![],
        unusable: vec![],
    };
    for (m, (mut runs, skipped)) in meters.iter().zip(per) {
        let src = m.source().to_string();
        if runs.is_empty() {
            let why = skipped
                .first()
                .map(|s| s.1.clone())
                .unwrap_or_else(|| "no windows".into());
            out.unusable.push((src, why));
            continue;
        }
        let mut sim = LinearSim::default();
        let samples: Vec<(Features, f64)> = runs
            .iter()
            .map(|r| (r.features.clone(), r.joules))
            .collect();
        sim.calibrate(&samples)
            .map_err(|e| CalibrateError::Fit(e.to_string()))?;
        let mut calibrated_terms = std::collections::BTreeMap::new();
        for k in sim.fits.keys() {
            let rows: Vec<[f64; N]> = runs
                .iter()
                .filter(|r| &r.features.key() == k)
                .map(|r| r.features.row())
                .collect();
            let varied: Vec<String> = (0..N)
                .filter(|&c| c == 0 || rows.iter().any(|r| r[c] != rows[0][c]))
                .map(|c| TERMS[c].to_string())
                .collect();
            calibrated_terms.insert(k.clone(), varied);
        }
        for r in runs.iter_mut() {
            r.predicted = sim.predict(&r.features).map(|p| p.joules);
        }
        let mean = runs.iter().map(|r| r.joules).sum::<f64>() / runs.len() as f64;
        let sst: f64 = runs.iter().map(|r| (r.joules - mean).powi(2)).sum();
        let sse: f64 = runs
            .iter()
            .map(|r| (r.joules - r.predicted.unwrap_or(0.0)).powi(2))
            .sum();
        let r2 = if sst > 0.0 { 1.0 - sse / sst } else { 0.0 };
        out.meters.push(MeterCalibration {
            meter: src,
            sim,
            calibrated_terms,
            r2,
            runs,
            skipped,
        });
    }
    if out.meters.is_empty() {
        return Err(CalibrateError::NothingMeasured(out.unusable));
    }
    Ok(out)
}

/// Save `c`: the whole report at `path`, and per meter
/// `<stem>.<meter>.json` beside it. The main file's top level is the first
/// (preferred) meter's simulator, so `rce-mcp --calibration <path>` and the
/// IDE load it directly. Written atomically (temp file + rename).
pub fn save(c: &Calibration, path: &std::path::Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let first = c
        .meters
        .first()
        .ok_or_else(|| std::io::Error::other("no meters"))?;
    let mut top = serde_json::to_value(first).map_err(std::io::Error::other)?;
    top["report"] = serde_json::to_value(c).map_err(std::io::Error::other)?;
    let mut written = vec![];
    let put = |p: &std::path::Path, v: &serde_json::Value| -> std::io::Result<()> {
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        let tmp = p.with_extension("json.tmp");
        std::fs::write(
            &tmp,
            serde_json::to_vec_pretty(v).map_err(std::io::Error::other)?,
        )?;
        std::fs::rename(&tmp, p)
    };
    put(path, &top)?;
    written.push(path.to_path_buf());
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("calibration");
    for m in &c.meters {
        let p = path.with_file_name(format!("{stem}.{}.json", m.meter));
        put(&p, &serde_json::to_value(m).map_err(std::io::Error::other)?)?;
        written.push(p);
    }
    Ok(written)
}

/// Load a saved calibration as a simulator (the main file or a per-meter one).
pub fn load_sim(path: &std::path::Path) -> Result<LinearSim, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))
}
