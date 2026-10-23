//! Hardware meters. Each returns cumulative joules since it was opened.
use crate::attrib::Sample;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, thiserror::Error)]
pub enum MeterError {
    #[error("{0} is not supported on this machine: {1}")]
    Unsupported(&'static str, String),
    #[error("{0}: {1}")]
    Read(&'static str, String),
    /// The source answered, but its counter did not move over a window in
    /// which the hardware certainly drew power: no number, not 0 J.
    #[error("{0} counter is stale: {1}")]
    Stale(&'static str, String),
}

/// Tag of a meter, matching `ide_receipt::MeterSource` by name.
pub trait Meter {
    fn source(&self) -> &'static str;
    /// Cumulative joules since open, stamped with Unix nanoseconds.
    fn sample(&mut self) -> Result<Sample, MeterError>;
}

pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

struct Zone {
    energy: PathBuf,
    max_uj: u64,
    last_uj: u64,
    total_uj: u128,
}

/// Linux powercap RAPL. Reads every top-level `intel-rapl:N` package zone
/// (the AMD driver uses the same names), so sub-zones (core, uncore, dram
/// inside the package) are not double counted. `psys` zones, when present,
/// are skipped because they overlap the packages.
pub struct RaplMeter {
    zones: Vec<Zone>,
}

fn read_u64(p: &Path) -> Result<u64, MeterError> {
    let s = std::fs::read_to_string(p)
        .map_err(|e| MeterError::Read("rapl", format!("{}: {e}", p.display())))?;
    s.trim()
        .parse()
        .map_err(|e| MeterError::Read("rapl", format!("{}: {e}", p.display())))
}

impl RaplMeter {
    /// Open the zones under `/sys/class/powercap`.
    pub fn open() -> Result<Self, MeterError> {
        Self::open_at(Path::new("/sys/class/powercap"))
    }

    /// Open the zones under a powercap root (tests use a fake tree).
    pub fn open_at(root: &Path) -> Result<Self, MeterError> {
        let rd = std::fs::read_dir(root)
            .map_err(|e| MeterError::Unsupported("rapl", format!("{}: {e}", root.display())))?;
        let mut names: Vec<String> = rd
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("intel-rapl:") && n.matches(':').count() == 1)
            .collect();
        names.sort();
        let mut zones = vec![];
        for n in names {
            let d = root.join(&n);
            if std::fs::read_to_string(d.join("name"))
                .map(|s| s.trim() == "psys")
                .unwrap_or(false)
            {
                continue;
            }
            let energy = d.join("energy_uj");
            let last_uj = read_u64(&energy)?;
            let max_uj = read_u64(&d.join("max_energy_range_uj"))?;
            zones.push(Zone {
                energy,
                max_uj,
                last_uj,
                total_uj: 0,
            });
        }
        if zones.is_empty() {
            return Err(MeterError::Unsupported(
                "rapl",
                format!("no package zones under {}", root.display()),
            ));
        }
        Ok(RaplMeter { zones })
    }
}

/// Microjoules between two counter reads, handling one wraparound at `max`.
pub fn wrap_delta(prev: u64, now: u64, max: u64) -> u64 {
    if now >= prev {
        now - prev
    } else {
        max.saturating_sub(prev) + now + 1
    }
}

impl Meter for RaplMeter {
    fn source(&self) -> &'static str {
        "rapl"
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        let t = now_ns();
        let mut total = 0u128;
        for z in &mut self.zones {
            let now = read_u64(&z.energy)?;
            z.total_uj += wrap_delta(z.last_uj, now, z.max_uj) as u128;
            z.last_uj = now;
            total += z.total_uj;
        }
        Ok(Sample {
            t_ns: t,
            joules: total as f64 * 1e-6,
        })
    }
}

/// NVIDIA NVML total energy counter, summed over all GPUs.
#[cfg(feature = "nvml")]
pub struct NvmlMeter {
    nvml: nvml_wrapper::Nvml,
    start_mj: Vec<u64>,
}

#[cfg(feature = "nvml")]
impl NvmlMeter {
    pub fn open() -> Result<Self, MeterError> {
        let nvml = nvml_wrapper::Nvml::init()
            .map_err(|e| MeterError::Unsupported("nvml", e.to_string()))?;
        let n = nvml
            .device_count()
            .map_err(|e| MeterError::Read("nvml", e.to_string()))?;
        let mut start_mj = vec![];
        for i in 0..n {
            let d = nvml
                .device_by_index(i)
                .map_err(|e| MeterError::Read("nvml", e.to_string()))?;
            start_mj.push(
                d.total_energy_consumption()
                    .map_err(|e| MeterError::Unsupported("nvml", e.to_string()))?,
            );
        }
        if start_mj.is_empty() {
            return Err(MeterError::Unsupported("nvml", "no GPUs".into()));
        }
        Ok(NvmlMeter { nvml, start_mj })
    }
}

#[cfg(feature = "nvml")]
impl Meter for NvmlMeter {
    fn source(&self) -> &'static str {
        "nvml"
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        let t = now_ns();
        let mut mj = 0u64;
        for (i, s) in self.start_mj.iter().enumerate() {
            let d = self
                .nvml
                .device_by_index(i as u32)
                .map_err(|e| MeterError::Read("nvml", e.to_string()))?;
            let now = d
                .total_energy_consumption()
                .map_err(|e| MeterError::Read("nvml", e.to_string()))?;
            mj += now.saturating_sub(*s);
        }
        Ok(Sample {
            t_ns: t,
            joules: mj as f64 * 1e-3,
        })
    }
}
