//! Linux hwmon (`/sys/class/hwmon`): energy counters, power readings and
//! voltage x current pairs exposed by kernel drivers.
//!
//! Covers, by driver name: `amdgpu` (AMD GPU power, `power1_average` or
//! `power1_input`), `i915` / `xe` (Intel GPU `energy1_input`), `amd_energy`
//! and `zenpower` (AMD CPU energy), `ina3221` (NVIDIA Jetson and other
//! boards: per-rail voltage and current), `ina2xx`, and any other driver
//! exposing the standard attributes. Units are the kernel's hwmon ABI:
//! energy uJ, power uW, voltage mV, current mA
//! (Documentation/hwmon/sysfs-interface.rst).
use crate::attrib::Sample;
use crate::meter::{Meter, MeterError, now_ns};
use crate::power::PolledMeter;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// What a hwmon device can give.
#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Cumulative energy files (uJ).
    Energy(Vec<PathBuf>),
    /// Power files (uW), polled.
    Power(Vec<PathBuf>),
    /// (voltage mV, current mA) file pairs, polled; power = V x I.
    VoltCurrent(Vec<(PathBuf, PathBuf)>),
}

/// One hwmon device with something usable.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub dir: PathBuf,
    pub name: String,
    pub kind: Kind,
}

impl Device {
    /// Hardware domain guessed from the driver name.
    pub fn domain(&self) -> &'static str {
        match self.name.as_str() {
            "amdgpu" | "nouveau" | "i915" | "xe" | "radeon" => "gpu",
            "amd_energy" | "zenpower" | "zenergy" => "cpu",
            "ina3221" | "ina2xx" | "ina226" | "ina219" => "board",
            _ => "other",
        }
    }
    /// The source tag: `jetson` for INA3221 rails, else `hwmon`.
    pub fn tag(&self) -> &'static str {
        if self.name == "ina3221" {
            "jetson"
        } else {
            "hwmon"
        }
    }
}

fn read_u64(p: &Path) -> Result<u64, String> {
    std::fs::read_to_string(p)
        .map_err(|e| format!("{}: {e}", p.display()))?
        .trim()
        .parse()
        .map_err(|e| format!("{}: {e}", p.display()))
}

fn files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// INA3221 channels whose label names a sum or the input supply would
/// double count the rails behind them; they are skipped.
fn skip_label(dir: &Path, idx: &str) -> bool {
    std::fs::read_to_string(dir.join(format!("in{idx}_label")))
        .map(|l| {
            let l = l.trim().to_ascii_uppercase();
            l.contains("SUM") || l == "VDD_IN" || l.contains("TOTAL")
        })
        .unwrap_or(false)
}

/// Scan a hwmon root (normally `/sys/class/hwmon`). Unreadable files are
/// left out; a device with nothing readable is not listed.
pub fn scan(root: &Path) -> Vec<Device> {
    let mut out = vec![];
    for d in files(root) {
        let dir = root.join(&d);
        let name = std::fs::read_to_string(dir.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let fs = files(&dir);
        let pick = |pre: &str, suf: &str| -> Vec<PathBuf> {
            fs.iter()
                .filter(|f| f.starts_with(pre) && f.ends_with(suf))
                .map(|f| dir.join(f))
                .filter(|p| read_u64(p).is_ok())
                .collect()
        };
        let energy = pick("energy", "_input");
        if !energy.is_empty() {
            out.push(Device {
                dir,
                name,
                kind: Kind::Energy(energy),
            });
            continue;
        }
        // power*_average preferred over *_input for the same index.
        let mut power = pick("power", "_average");
        if power.is_empty() {
            power = pick("power", "_input");
        }
        if !power.is_empty() {
            out.push(Device {
                dir,
                name,
                kind: Kind::Power(power),
            });
            continue;
        }
        let mut pairs = vec![];
        for f in &fs {
            if let Some(idx) = f
                .strip_prefix("curr")
                .and_then(|r| r.strip_suffix("_input"))
            {
                let (v, c) = (dir.join(format!("in{idx}_input")), dir.join(f));
                if read_u64(&v).is_ok() && read_u64(&c).is_ok() && !skip_label(&dir, idx) {
                    pairs.push((v, c));
                }
            }
        }
        if !pairs.is_empty() {
            out.push(Device {
                dir,
                name,
                kind: Kind::VoltCurrent(pairs),
            });
        }
    }
    out
}

/// Watts from a power or voltage-current device now.
pub fn watts(kind: &Kind) -> Result<f64, String> {
    match kind {
        Kind::Power(ps) => ps
            .iter()
            .try_fold(0.0, |a, p| Ok(a + read_u64(p)? as f64 * 1e-6)),
        Kind::VoltCurrent(vs) => vs.iter().try_fold(0.0, |a, (v, c)| {
            Ok(a + read_u64(v)? as f64 * read_u64(c)? as f64 * 1e-6)
        }),
        Kind::Energy(_) => Err("energy device has no instantaneous power".into()),
    }
}

/// Cumulative energy counters of one device.
pub struct EnergyMeter {
    tag: &'static str,
    files: Vec<PathBuf>,
    start: Vec<u64>,
}

impl Meter for EnergyMeter {
    fn source(&self) -> &'static str {
        self.tag
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        let t = now_ns();
        let mut uj = 0u128;
        for (f, s) in self.files.iter().zip(&self.start) {
            let now = read_u64(f).map_err(|e| MeterError::Read(self.tag, e))?;
            // hwmon energy counters are 64-bit and do not wrap in practice;
            // a decrease means a reset, which is an error, not negative energy.
            if now < *s {
                return Err(MeterError::Read(
                    self.tag,
                    format!("{} went backwards", f.display()),
                ));
            }
            uj += (now - s) as u128;
        }
        Ok(Sample {
            t_ns: t,
            joules: uj as f64 * 1e-6,
        })
    }
}

/// Open a meter on one device: energy counters directly, power and
/// voltage-current by polling every `period`.
pub fn open(dev: &Device, period: Duration) -> Result<Box<dyn Meter + Send>, MeterError> {
    let tag = dev.tag();
    match &dev.kind {
        Kind::Energy(fs) => {
            let start = fs
                .iter()
                .map(|f| read_u64(f).map_err(|e| MeterError::Unsupported(tag, e)))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Box::new(EnergyMeter {
                tag,
                files: fs.clone(),
                start,
            }))
        }
        k => {
            let k = k.clone();
            Ok(Box::new(PolledMeter::start(tag, period, move || {
                watts(&k).map_err(|e| MeterError::Read(tag, e))
            })?))
        }
    }
}
