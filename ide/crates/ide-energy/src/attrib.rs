//! Attribute measured energy to an action by its time window.

/// One reading of a cumulative counter: time (ns, monotonic or Unix, as long
/// as one clock is used throughout) and joules since the meter started.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub t_ns: u64,
    pub joules: f64,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum AttribError {
    #[error("window [{0}, {1}] is empty or reversed")]
    Window(u64, u64),
    #[error("samples do not cover the window")]
    Uncovered,
}

fn at(s: &[Sample], t: u64) -> Option<f64> {
    let i = s.partition_point(|x| x.t_ns <= t);
    if i == 0 {
        return None;
    }
    let a = s[i - 1];
    if a.t_ns == t {
        return Some(a.joules);
    }
    let b = *s.get(i)?;
    let w = (t - a.t_ns) as f64 / (b.t_ns - a.t_ns) as f64;
    Some(a.joules + w * (b.joules - a.joules))
}

/// Joules drawn in `[start, end]`, linearly interpolating the cumulative
/// counter between samples. Samples must be sorted by time. The meter sees
/// the whole machine (or package, or GPU): concurrent work inside the window
/// is included; that is what the delta against E_sim is meant to expose.
pub fn attribute(samples: &[Sample], start_ns: u64, end_ns: u64) -> Result<f64, AttribError> {
    if end_ns <= start_ns {
        return Err(AttribError::Window(start_ns, end_ns));
    }
    let a = at(samples, start_ns).ok_or(AttribError::Uncovered)?;
    let b = at(samples, end_ns).ok_or(AttribError::Uncovered)?;
    Ok((b - a).max(0.0))
}
