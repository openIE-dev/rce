//! Linux / Android `power_supply` class: battery discharge power.
//!
//! On a phone, a Raspberry Pi with a battery HAT, or a laptop running on
//! battery, `power_now` (uW), or `current_now` (uA) x `voltage_now` (uV), is
//! what the whole device draws. It means that only while the battery is
//! discharging; when charging or full it says nothing about consumption, so
//! the reading is refused then. Android's BatteryManager reads the same
//! kernel attributes (BATTERY_PROPERTY_CURRENT_NOW).
use crate::meter::MeterError;
use crate::power::PolledMeter;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn rd(p: &Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
}

/// Batteries under a power_supply root (normally `/sys/class/power_supply`).
pub fn batteries(root: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(root)
        .map(|r| r.filter_map(|e| e.ok()).map(|e| e.path()).collect())
        .unwrap_or_default();
    v.retain(|d| rd(&d.join("type")).as_deref() == Some("Battery"));
    v.sort();
    v
}

/// Discharge power of one battery in watts, or why there is none.
pub fn discharge_watts(dir: &Path) -> Result<f64, String> {
    let status = rd(&dir.join("status")).unwrap_or_default();
    if status != "Discharging" {
        return Err(format!("battery status is {status:?}, not Discharging"));
    }
    let num = |f: &str| rd(&dir.join(f)).and_then(|s| s.parse::<i64>().ok());
    // Some drivers report discharge current as negative.
    if let Some(p) = num("power_now") {
        return Ok(p.unsigned_abs() as f64 * 1e-6);
    }
    match (num("current_now"), num("voltage_now")) {
        (Some(i), Some(v)) => Ok(i.unsigned_abs() as f64 * v.unsigned_abs() as f64 * 1e-12),
        _ => Err("no power_now, or current_now and voltage_now".into()),
    }
}

/// Polled battery discharge meter.
pub fn open(dir: &Path, period: Duration) -> Result<PolledMeter, MeterError> {
    discharge_watts(dir).map_err(|e| MeterError::Unsupported("power-supply", e))?;
    let d = dir.to_path_buf();
    PolledMeter::start("power-supply", period, move || {
        discharge_watts(&d).map_err(|e| MeterError::Read("power-supply", e))
    })
}
