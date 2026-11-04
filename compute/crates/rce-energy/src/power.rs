//! Power (watts) sources turned into cumulative joules.
//!
//! Some sources give instantaneous or averaged power, not an energy counter
//! (SMC `PSTR`, hwmon `power*_input`/`power*_average`, battery `power_now`,
//! INA3221 voltage x current, powermetrics). A background thread polls the
//! source at a fixed period and integrates with the trapezoid rule. The
//! result is only as good as the period: power changes faster than it are
//! smoothed or missed, which each source notes in its accuracy text.
use crate::attrib::Sample;
use crate::meter::{Meter, MeterError, now_ns};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Trapezoid integration of (t_ns, watts) points.
#[derive(Debug, Default, Clone)]
pub struct Integrator {
    last: Option<(u64, f64)>,
    joules: f64,
    points: u64,
}

impl Integrator {
    /// Add a point. Negative, NaN or out-of-order points are an error.
    pub fn push(&mut self, t_ns: u64, watts: f64) -> Result<(), String> {
        if !watts.is_finite() || watts < 0.0 {
            return Err(format!("bad power reading {watts} W"));
        }
        if let Some((t0, w0)) = self.last {
            if t_ns < t0 {
                return Err("time went backwards".into());
            }
            self.joules += (t_ns - t0) as f64 * 1e-9 * (w0 + watts) / 2.0;
        }
        self.last = Some((t_ns, watts));
        self.points += 1;
        Ok(())
    }
    pub fn joules(&self) -> f64 {
        self.joules
    }
    pub fn points(&self) -> u64 {
        self.points
    }
    pub fn last(&self) -> Option<(u64, f64)> {
        self.last
    }
}

struct State {
    integ: Integrator,
    err: Option<String>,
}

/// A meter over a polled power reading.
pub struct PolledMeter {
    tag: &'static str,
    state: Arc<Mutex<State>>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PolledMeter {
    /// Start polling `read` (watts) every `period`. The first read happens
    /// before this returns, so a source that cannot be read fails here.
    pub fn start<F>(tag: &'static str, period: Duration, mut read: F) -> Result<Self, MeterError>
    where
        F: FnMut() -> Result<f64, MeterError> + Send + 'static,
    {
        let first = read()?;
        let mut integ = Integrator::default();
        integ
            .push(now_ns(), first)
            .map_err(|e| MeterError::Read(tag, e))?;
        let state = Arc::new(Mutex::new(State { integ, err: None }));
        let stop = Arc::new(AtomicBool::new(false));
        let (s2, st2) = (state.clone(), stop.clone());
        let thread = std::thread::spawn(move || {
            while !st2.load(Ordering::Relaxed) {
                std::thread::sleep(period);
                let r = read();
                let t = now_ns();
                let mut g = s2.lock().unwrap_or_else(|p| p.into_inner());
                match r {
                    Ok(w) => {
                        if let Err(e) = g.integ.push(t, w) {
                            g.err = Some(e);
                        }
                    }
                    Err(e) => g.err = Some(e.to_string()),
                }
            }
        });
        Ok(PolledMeter {
            tag,
            state,
            stop,
            thread: Some(thread),
        })
    }

    /// Number of power points integrated so far.
    pub fn points(&self) -> u64 {
        self.state.lock().map(|g| g.integ.points()).unwrap_or(0)
    }
}

impl Drop for PolledMeter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Meter for PolledMeter {
    fn source(&self) -> &'static str {
        self.tag
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        let g = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(e) = &g.err {
            return Err(MeterError::Read(self.tag, e.clone()));
        }
        let (t, _) = g
            .integ
            .last()
            .ok_or_else(|| MeterError::Read(self.tag, "no reading yet".into()))?;
        Ok(Sample {
            t_ns: t,
            joules: g.integ.joules(),
        })
    }
}
