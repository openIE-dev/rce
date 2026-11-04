//! Cumulative microjoules per channel, for the placement crates.
//!
//! Two backends. `linux-rapl` reads the kernel's RAPL counters under
//! `/sys/class/powercap` (package domains count as `Cpu`, `dram`
//! subdomains as `Dram`, their sum as `Total`), correcting for counter
//! wraparound. `fallback-tdp` is an estimate, not a measurement: process
//! CPU time on Linux, or elapsed time elsewhere, times a per-core power.
//! Both are cumulative, so the energy of an interval is the difference
//! of two snapshots (`EnergySnapshot::total_uj_since`).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Channel {
    Total,
    Cpu,
    Gpu,
    Dram,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnergySnapshot {
    /// Cumulative microjoules per channel.
    pub channels: BTreeMap<Channel, u64>,
}

impl EnergySnapshot {
    /// Microjoules on `Total` since `prior` (zero if the counter went back).
    pub fn total_uj_since(&self, prior: &EnergySnapshot) -> u64 {
        let at = |s: &EnergySnapshot| s.channels.get(&Channel::Total).copied().unwrap_or(0);
        at(self).saturating_sub(at(prior))
    }
}

#[derive(Debug)]
pub enum EnergyError {
    Read(String),
}

impl std::fmt::Display for EnergyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnergyError::Read(s) => write!(f, "energy read failed: {s}"),
        }
    }
}

impl std::error::Error for EnergyError {}

/// What the backend knows about the machine it measures.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Calibration {
    /// The processor model, from `/proc/cpuinfo`; empty when unknown.
    pub model: String,
}

impl Calibration {
    fn detect() -> Calibration {
        let model = fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| s.lines().find(|l| l.starts_with("model name")).and_then(|l| l.split_once(':')).map(|(_, m)| m.trim().to_string()))
            .unwrap_or_default();
        Calibration { model }
    }
}

pub trait EnergyTelemetry: Send {
    fn name(&self) -> &'static str;
    fn calibration(&self) -> &Calibration;
    fn snapshot(&mut self) -> Result<EnergySnapshot, EnergyError>;
}

/// The RAPL counters if this host exposes a readable one, else the estimate.
pub fn open_default_backend() -> Box<dyn EnergyTelemetry> {
    match Rapl::open() {
        Some(r) => Box::new(r),
        None => Box::new(FallbackTdp::new()),
    }
}

struct Domain {
    path: PathBuf,
    channel: Channel,
    max_uj: u64,
    last_raw: u64,
    acc_uj: u64,
}

/// Linux RAPL (powercap) counters, accumulated across wraparound.
pub struct Rapl {
    domains: Vec<Domain>,
    calibration: Calibration,
}

impl Rapl {
    pub fn open() -> Option<Rapl> {
        let mut domains = vec![];
        for e in fs::read_dir("/sys/class/powercap").ok()?.flatten() {
            let dir = e.path();
            let Some(base) = dir.file_name().and_then(|n| n.to_str()) else { continue };
            if !base.starts_with("intel-rapl:") {
                continue;
            }
            let name = fs::read_to_string(dir.join("name")).unwrap_or_default();
            let channel = match name.trim() {
                n if n.starts_with("package") => Channel::Cpu,
                "dram" => Channel::Dram,
                _ => continue,
            };
            let read = |f: &str| fs::read_to_string(dir.join(f)).ok()?.trim().parse::<u64>().ok();
            let (Some(raw), max) = (read("energy_uj"), read("max_energy_range_uj").unwrap_or(u64::MAX)) else { continue };
            domains.push(Domain { path: dir.join("energy_uj"), channel, max_uj: max, last_raw: raw, acc_uj: 0 });
        }
        (!domains.is_empty()).then_some(Rapl { domains, calibration: Calibration::detect() })
    }
}

impl EnergyTelemetry for Rapl {
    fn name(&self) -> &'static str {
        "linux-rapl"
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    fn snapshot(&mut self) -> Result<EnergySnapshot, EnergyError> {
        let mut s = EnergySnapshot::default();
        for d in &mut self.domains {
            let raw: u64 = fs::read_to_string(&d.path)
                .map_err(|e| EnergyError::Read(format!("{}: {e}", d.path.display())))?
                .trim()
                .parse()
                .map_err(|e| EnergyError::Read(format!("{}: {e}", d.path.display())))?;
            let step = crate::meter::wrap_delta(d.last_raw, raw, d.max_uj);
            d.last_raw = raw;
            d.acc_uj = d.acc_uj.saturating_add(step);
            *s.channels.entry(d.channel).or_insert(0) += d.acc_uj;
            *s.channels.entry(Channel::Total).or_insert(0) += d.acc_uj;
        }
        Ok(s)
    }
}

/// An estimate: CPU time (Linux) or elapsed time (elsewhere) times a
/// per-core power. Use it only where no counter can be read.
pub struct FallbackTdp {
    start: Instant,
    watts_per_core: f64,
    calibration: Calibration,
}

impl FallbackTdp {
    pub fn new() -> FallbackTdp {
        FallbackTdp { start: Instant::now(), watts_per_core: 10.0, calibration: Calibration::detect() }
    }

    fn cpu_seconds(&self) -> f64 {
        // /proc/self/stat fields 14 and 15: user and system time in clock ticks (100 Hz).
        if let Ok(s) = fs::read_to_string("/proc/self/stat")
            && let Some(rest) = s.rsplit_once(')').map(|(_, r)| r)
        {
            let f: Vec<&str> = rest.split_whitespace().collect();
            if let (Some(u), Some(k)) = (f.get(11).and_then(|x| x.parse::<f64>().ok()), f.get(12).and_then(|x| x.parse::<f64>().ok())) {
                return (u + k) / 100.0;
            }
        }
        self.start.elapsed().as_secs_f64()
    }
}

impl Default for FallbackTdp {
    fn default() -> Self {
        Self::new()
    }
}

impl EnergyTelemetry for FallbackTdp {
    fn name(&self) -> &'static str {
        "fallback-tdp"
    }

    fn calibration(&self) -> &Calibration {
        &self.calibration
    }

    fn snapshot(&mut self) -> Result<EnergySnapshot, EnergyError> {
        let uj = (self.cpu_seconds() * self.watts_per_core * 1e6) as u64;
        let mut s = EnergySnapshot::default();
        s.channels.insert(Channel::Cpu, uj);
        s.channels.insert(Channel::Total, uj);
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_backend_snapshots_are_cumulative() {
        let mut b = open_default_backend();
        let a = b.snapshot().unwrap();
        let mut x = 0u64;
        for i in 0..2_000_000u64 {
            x = x.wrapping_add(i.wrapping_mul(13));
        }
        std::hint::black_box(x);
        let c = b.snapshot().unwrap();
        assert!(c.channels[&Channel::Total] >= a.channels[&Channel::Total]);
        assert_eq!(c.total_uj_since(&a), c.channels[&Channel::Total] - a.channels[&Channel::Total]);
    }
}
