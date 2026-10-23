//! macOS `powermetrics` (root only). Used only after the user has set up a
//! sudoers entry with `rce-ide telemetry setup`; it runs `sudo -n`, which
//! never prompts, so without that entry it is `Unsupported`.
//!
//! Each powermetrics sample prints average power over its interval, so
//! energy per interval is power x interval exactly as powermetrics measured
//! it (it reads the same counters as IOReport, with root).
use crate::attrib::Sample;
use crate::meter::{Meter, MeterError, now_ns};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

pub const PATH: &str = "/usr/bin/powermetrics";

/// Which powermetrics line to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Cpu,
    Gpu,
    Ane,
    Combined,
}

impl Field {
    fn prefix(self) -> &'static str {
        match self {
            Field::Cpu => "CPU Power:",
            Field::Gpu => "GPU Power:",
            Field::Ane => "ANE Power:",
            Field::Combined => "Combined Power (CPU + GPU + ANE):",
        }
    }
}

/// Watts from one powermetrics text line, when it is the line of `f`.
pub fn parse_line(line: &str, f: Field) -> Option<f64> {
    let rest = line.trim().strip_prefix(f.prefix())?.trim();
    let (n, unit) = rest.split_once(' ').unwrap_or((rest, ""));
    let v: f64 = n.parse().ok()?;
    let w = match unit.trim() {
        "mW" => v / 1e3,
        "W" => v,
        _ => return None,
    };
    (w.is_finite() && w >= 0.0).then_some(w)
}

/// The sudoers entry `rce-ide telemetry setup` writes.
pub const SUDOERS: &str = "/etc/sudoers.d/rce-powermetrics";

/// True only when the user set this source up: our sudoers entry exists and
/// `sudo -n -l` confirms the rule. Never runs powermetrics and never relies
/// on a sudo timestamp cached by some other terminal: without the entry,
/// this is false even if `sudo -n` would happen to work.
pub fn allowed() -> bool {
    std::path::Path::new(SUDOERS).exists()
        && Command::new("sudo")
            .args(["-n", "-l", PATH])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
}

struct St {
    joules: f64,
    t: u64,
    n: u64,
    err: Option<String>,
}

pub struct PowermetricsMeter {
    child: Child,
    st: Arc<Mutex<St>>,
}

impl PowermetricsMeter {
    pub fn open(f: Field, interval_ms: u64) -> Result<Self, MeterError> {
        if !cfg!(target_os = "macos") {
            return Err(MeterError::Unsupported("powermetrics", "macOS only".into()));
        }
        if !allowed() {
            return Err(MeterError::Unsupported(
                "powermetrics",
                "needs root: run `rce-ide telemetry setup` to add a sudoers entry (asks first)"
                    .into(),
            ));
        }
        let mut child = Command::new("sudo")
            .args([
                "-n",
                PATH,
                "-i",
                &interval_ms.to_string(),
                "--samplers",
                "cpu_power,gpu_power,ane_power",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| MeterError::Unsupported("powermetrics", e.to_string()))?;
        let out = child
            .stdout
            .take()
            .ok_or_else(|| MeterError::Read("powermetrics", "no stdout".into()))?;
        let st = Arc::new(Mutex::new(St {
            joules: 0.0,
            t: now_ns(),
            n: 0,
            err: None,
        }));
        let s2 = st.clone();
        let dt = interval_ms as f64 / 1e3;
        std::thread::spawn(move || {
            for l in BufReader::new(out).lines() {
                let Ok(l) = l else { break };
                if let Some(w) = parse_line(&l, f) {
                    let mut g = s2.lock().unwrap_or_else(|p| p.into_inner());
                    g.joules += w * dt;
                    g.t = now_ns();
                    g.n += 1;
                }
            }
            s2.lock().unwrap_or_else(|p| p.into_inner()).err = Some("powermetrics exited".into());
        });
        Ok(PowermetricsMeter { child, st })
    }
}

impl Drop for PowermetricsMeter {
    fn drop(&mut self) {
        // SIGTERM to sudo is relayed to powermetrics; SIGKILL would orphan it.
        let _ = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status();
        let _ = self.child.wait();
    }
}

impl Meter for PowermetricsMeter {
    fn source(&self) -> &'static str {
        "powermetrics"
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        let g = self.st.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(e) = &g.err {
            return Err(MeterError::Read("powermetrics", e.clone()));
        }
        if g.n == 0 {
            return Err(MeterError::Read("powermetrics", "no sample yet".into()));
        }
        Ok(Sample {
            t_ns: g.t,
            joules: g.joules,
        })
    }
}
