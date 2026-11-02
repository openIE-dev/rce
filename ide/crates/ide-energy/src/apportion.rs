//! Split a shared device's measured energy among the processes using it.
//!
//! Lesson from Ampere (adam-mcdaniel/ampere): when one GPU or package is
//! shared by several ranks, its energy must be divided among them, not
//! charged to all or to one. We split each sampled interval by each
//! process's activity weight (for example busy time or utilization) in that
//! interval. Intervals where no process was active are reported as idle,
//! never silently dropped or spread.

use crate::attrib::Sample;
use std::collections::BTreeMap;

/// Activity of one process on the device over `[start_ns, end_ns)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    pub process: String,
    pub start_ns: u64,
    pub end_ns: u64,
    /// Relative weight while active (1.0 = fully busy). Must be >= 0.
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Apportioned {
    pub per_process: BTreeMap<String, f64>,
    /// Joules in intervals with no active process.
    pub idle: f64,
}

impl Apportioned {
    pub fn total(&self) -> f64 {
        self.idle + self.per_process.values().sum::<f64>()
    }
}

fn overlap(a0: u64, a1: u64, b0: u64, b1: u64) -> u64 {
    a1.min(b1).saturating_sub(a0.max(b0))
}

/// Apportion a cumulative counter's energy across processes. Conserves
/// energy: `result.total()` equals the counter delta over the samples.
pub fn apportion(samples: &[Sample], acts: &[Activity]) -> Apportioned {
    let mut out = Apportioned::default();
    for w in samples.windows(2) {
        let (a, b) = (w[0], w[1]);
        let dj = (b.joules - a.joules).max(0.0);
        if b.t_ns <= a.t_ns || dj == 0.0 {
            continue;
        }
        let mut shares: BTreeMap<&str, f64> = BTreeMap::new();
        for x in acts {
            let ov = overlap(a.t_ns, b.t_ns, x.start_ns, x.end_ns);
            if ov > 0 && x.weight > 0.0 {
                *shares.entry(&x.process).or_default() += ov as f64 * x.weight;
            }
        }
        let sum: f64 = shares.values().sum();
        if sum == 0.0 {
            out.idle += dj;
            continue;
        }
        for (p, s) in shares {
            *out.per_process.entry(p.to_string()).or_default() += dj * s / sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    fn s(t: u64, j: f64) -> Sample { Sample { t_ns: t, joules: j } }
    fn act(p: &str, a: u64, b: u64, w: f64) -> Activity {
        Activity { process: p.into(), start_ns: a, end_ns: b, weight: w }
    }

    #[test]
    fn two_ranks_share_one_gpu_equally() {
        let r = apportion(&[s(0, 0.0), s(10, 20.0)], &[act("a", 0, 10, 1.0), act("b", 0, 10, 1.0)]);
        assert_eq!(r.per_process["a"], 10.0);
        assert_eq!(r.per_process["b"], 10.0);
        assert_eq!(r.idle, 0.0);
    }

    #[test]
    fn weights_and_idle_conserve_energy() {
        let smp = [s(0, 0.0), s(10, 30.0), s(20, 40.0)];
        let r = apportion(&smp, &[act("a", 0, 10, 3.0), act("b", 0, 10, 1.0)]);
        assert!((r.per_process["a"] - 22.5).abs() < 1e-9);
        assert!((r.per_process["b"] - 7.5).abs() < 1e-9);
        assert_eq!(r.idle, 10.0);
        assert!((r.total() - 40.0).abs() < 1e-9);
    }
}
