//! Capability discovery: probe every energy source this build knows, report
//! what was found, what could not be used and why, and list the sources
//! that are tracked but not implemented. Nothing here returns a number for
//! a source that is not working.
use crate::meter::MeterError;
use std::path::Path;

/// Status of one source on this machine.
#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    /// Opened and read.
    Available,
    /// Present but needs a permission the user can grant with
    /// `rce-ide telemetry setup` (with their consent).
    NeedsSetup(String),
    /// Not usable here, with the reason.
    Unsupported(String),
    /// Present but its counter did not move.
    Stale(String),
}

/// One probed source.
#[derive(Debug, Clone, PartialEq)]
pub struct Probe {
    /// Source tag, as in `Meter::source` and the receipt.
    pub tag: &'static str,
    /// What it measures, e.g. "cpu", "gpu", "system", "battery".
    pub domain: String,
    /// Which device or channel, e.g. "CLPC CPU Energy" or a sysfs path.
    pub detail: String,
    pub status: Status,
    pub resolution: &'static str,
    pub accuracy: &'static str,
}

/// A source known but not implemented here.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tracked {
    pub tag: &'static str,
    pub domain: &'static str,
    pub why: &'static str,
    pub reference: &'static str,
}

pub fn status_of(e: &MeterError) -> Status {
    match e {
        MeterError::Unsupported(_, m) => Status::Unsupported(m.clone()),
        MeterError::Stale(_, m) => Status::Stale(m.clone()),
        MeterError::Read(_, m) => Status::Unsupported(m.clone()),
    }
}

/// Resolution and accuracy notes per tag.
pub fn notes(tag: &str) -> (&'static str, &'static str) {
    match tag {
        "rapl" => (
            "energy counter, ~1 ms update, 15.3 uJ units on Intel",
            "model-based on Intel before Haswell, measured on later Intel and AMD Zen; package domain covers cores, uncore and (server) DRAM",
        ),
        "ioreport" => (
            "energy counter in nJ/mJ, read on demand",
            "Apple's on-chip energy model per rail; CLPC aggregates on macOS 27; not a shunt measurement",
        ),
        "smc" => (
            "instantaneous power polled every 100 ms, trapezoid integrated",
            "SMC PSTR is whole-system power from the SMC's sensors; changes faster than the poll period are smoothed",
        ),
        "powermetrics" => (
            "average power per interval (default 200 ms)",
            "same counters as IOReport, read by Apple's tool with root",
        ),
        "nvml" => (
            "energy counter in mJ (Volta and later)",
            "board power per NVIDIA, about +/-5 W on datacenter boards",
        ),
        "hwmon" => (
            "driver-defined: energy counter (uJ) or power (uW) polled every 100 ms",
            "driver-defined; amdgpu power is averaged by the SMU, i915/xe energy is a counter",
        ),
        "jetson" => (
            "INA3221 voltage x current polled every 100 ms",
            "shunt measurement per rail, about 1-2 %; rails behind the module input only",
        ),
        "power-supply" => (
            "battery power or current x voltage polled every 100 ms",
            "fuel-gauge reading, often averaged over seconds; whole device; only while discharging",
        ),
        _ => ("unknown", "unknown"),
    }
}

fn probe(tag: &'static str, domain: &str, detail: String, status: Status) -> Probe {
    let (resolution, accuracy) = notes(tag);
    Probe {
        tag,
        domain: domain.into(),
        detail,
        status,
        resolution,
        accuracy,
    }
}

/// Probe every implemented source. Opening a source is the probe; a source
/// that opens is read once.
pub fn probe_all() -> Vec<Probe> {
    let mut out = vec![];
    probe_rapl(Path::new("/sys/class/powercap"), &mut out);
    for d in crate::hwmon::scan(Path::new("/sys/class/hwmon")) {
        let st = match crate::hwmon::open(&d, std::time::Duration::from_millis(100)) {
            Ok(mut m) => match m.sample() {
                Ok(_) => Status::Available,
                Err(e) => status_of(&e),
            },
            Err(e) => status_of(&e),
        };
        out.push(probe(
            d.tag(),
            d.domain(),
            format!("{} ({})", d.dir.display(), d.name),
            st,
        ));
    }
    for b in crate::supply::batteries(Path::new("/sys/class/power_supply")) {
        let st = match crate::supply::discharge_watts(&b) {
            Ok(_) => Status::Available,
            Err(e) => Status::Unsupported(e),
        };
        out.push(probe(
            "power-supply",
            "battery",
            b.display().to_string(),
            st,
        ));
    }
    out.push(probe_nvml());
    probe_macos(&mut out);
    out
}

/// RAPL under a powercap root; split out for tests.
pub fn probe_rapl(root: &Path, out: &mut Vec<Probe>) {
    use crate::meter::{Meter, RaplMeter};
    let has = std::fs::read_dir(root)
        .map(|r| {
            r.filter_map(|e| e.ok())
                .any(|e| e.file_name().to_string_lossy().starts_with("intel-rapl:"))
        })
        .unwrap_or(false);
    if !has {
        if cfg!(target_os = "linux") {
            out.push(probe(
                "rapl",
                "cpu",
                root.display().to_string(),
                Status::Unsupported("no intel-rapl zones".into()),
            ));
        }
        return;
    }
    let st = match RaplMeter::open_at(root).and_then(|mut m| m.sample()) {
        Ok(_) => Status::Available,
        Err(MeterError::Read(_, m)) if m.contains("ermission denied") => {
            Status::NeedsSetup(format!(
                "{m}; energy_uj is root-only since Linux 5.10 (CVE-2020-8694): `rce-ide telemetry setup` can add a udev rule for a `power` group"
            ))
        }
        Err(e) => status_of(&e),
    };
    out.push(probe("rapl", "cpu", root.display().to_string(), st));
}

fn probe_nvml() -> Probe {
    #[cfg(feature = "nvml")]
    let st = match crate::meter::NvmlMeter::open() {
        Ok(_) => Status::Available,
        Err(e) => status_of(&e),
    };
    #[cfg(not(feature = "nvml"))]
    let st = Status::Unsupported("built without the `nvml` feature".into());
    probe("nvml", "gpu", "NVIDIA GPUs".into(), st)
}

#[cfg(target_os = "macos")]
fn probe_macos(out: &mut Vec<Probe>) {
    use crate::ioreport::{IoReportMeter, Rail};
    use crate::meter::Meter;
    for rail in [Rail::Cpu, Rail::Gpu, Rail::Ane, Rail::Dram] {
        let (st, detail) = match IoReportMeter::open(rail) {
            Ok(mut m) => {
                std::thread::sleep(std::time::Duration::from_millis(50));
                let st = match m.sample() {
                    Ok(_) => Status::Available,
                    Err(e) => status_of(&e),
                };
                (st, format!("{:?} via {:?}", rail, m.origin()))
            }
            Err(e) => (status_of(&e), format!("{rail:?}")),
        };
        out.push(probe("ioreport", rail.name(), detail, st));
    }
    let st = match crate::smc::SmcMeter::open(std::time::Duration::from_millis(100)) {
        Ok(_) => Status::Available,
        Err(e) => status_of(&e),
    };
    out.push(probe("smc", "system", "SMC PSTR".into(), st));
    let st = if crate::powermetrics::allowed() {
        Status::Available
    } else {
        Status::NeedsSetup("root only; `rce-ide telemetry setup` can add a sudoers entry".into())
    };
    out.push(probe(
        "powermetrics",
        "cpu,gpu,ane",
        crate::powermetrics::PATH.into(),
        st,
    ));
}

#[cfg(not(target_os = "macos"))]
fn probe_macos(_: &mut Vec<Probe>) {}

/// Sources known but not implemented in this build.
pub const TRACKED: &[Tracked] = &[
    Tracked {
        tag: "windows-emi",
        domain: "cpu,gpu,system",
        why: "Energy Meter Interface (IOCTL_EMI_GET_MEASUREMENT on GUID_DEVICE_ENERGY_METER); needs Win32 FFI and a Windows machine to test",
        reference: "https://learn.microsoft.com/en-us/windows-hardware/drivers/powermeter/energy-meter-interface",
    },
    Tracked {
        tag: "amd-smi",
        domain: "gpu,cpu",
        why: "AMD SMI / ROCm SMI C library; amdgpu hwmon covers GPU power without it",
        reference: "https://github.com/ROCm/amdsmi",
    },
    Tracked {
        tag: "level-zero",
        domain: "gpu,npu",
        why: "Intel Level Zero Sysman zesPowerGetEnergyCounter; i915/xe hwmon covers GPU energy without it",
        reference: "https://spec.oneapi.io/level-zero/latest/sysman/api.html",
    },
    Tracked {
        tag: "intel-npu",
        domain: "npu",
        why: "no public energy counter for the Intel NPU (intel_vpu driver) found",
        reference: "https://docs.kernel.org/accel/intel_vpu/",
    },
    Tracked {
        tag: "arm-scmi",
        domain: "cpu",
        why: "SCMI powercap / perf energy counters appear as hwmon or powercap on some SoCs (covered when present); no generic Arm PMU energy event",
        reference: "https://docs.kernel.org/hwmon/scmi-hwmon.html",
    },
    Tracked {
        tag: "qualcomm",
        domain: "npu,soc",
        why: "Snapdragon power rails via vendor tools (Snapdragon Profiler) or Android power HAL; not exposed generically",
        reference: "https://source.android.com/docs/core/power/values",
    },
    Tracked {
        tag: "tpu",
        domain: "accelerator",
        why: "Cloud TPU exposes power only through Cloud Monitoring metrics, not on the host",
        reference: "https://cloud.google.com/tpu/docs/tpu-monitoring-library",
    },
    Tracked {
        tag: "tegrastats",
        domain: "board",
        why: "same INA3221 data as hwmon (implemented as `jetson`); parsing tegrastats output adds nothing",
        reference: "https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html",
    },
    Tracked {
        tag: "rpi-pmic",
        domain: "board",
        why: "Raspberry Pi 5 PMIC rails via `vcgencmd pmic_read_adc` (firmware mailbox), not sysfs",
        reference: "https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd",
    },
    Tracked {
        tag: "probe-rs",
        domain: "mcu",
        why: "MCU current sensing needs a debug probe with a power monitor (e.g. ST-LINK V3PWR) and probe-rs support for it",
        reference: "https://probe.rs",
    },
    Tracked {
        tag: "ipmi-dcmi",
        domain: "system",
        why: "DCMI Get Power Reading over /dev/ipmi0 needs root and a BMC",
        reference: "https://www.intel.com/content/dam/www/public/us/en/documents/technical-specifications/dcmi-v1-5-rev-spec.pdf",
    },
    Tracked {
        tag: "redfish",
        domain: "system",
        why: "Redfish Power/PowerSubsystem over HTTPS needs BMC credentials",
        reference: "https://www.dmtf.org/standards/redfish",
    },
    Tracked {
        tag: "ios",
        domain: "battery",
        why: "no public per-process or rail energy API; Xcode Energy Organizer only",
        reference: "https://developer.apple.com/documentation/xcode/analyzing-your-app-s-battery-use",
    },
    Tracked {
        tag: "freebsd",
        domain: "cpu",
        why: "RAPL via cpuctl(4) MSR reads needs root; no powercap equivalent",
        reference: "https://man.freebsd.org/cgi/man.cgi?cpuctl(4)",
    },
    Tracked {
        tag: "usb-meter",
        domain: "external",
        why: "USB power meters (e.g. FNIRSI, UM25C over Bluetooth/serial) each have their own protocol",
        reference: "https://sigrok.org/wiki/Supported_hardware#Multimeters",
    },
    Tracked {
        tag: "smart-pdu",
        domain: "external",
        why: "PDU outlet power over SNMP or vendor HTTP APIs; per-model",
        reference: "https://www.raritan.com/support/product/pdu",
    },
];

/// Open every implemented source that works here, for measuring (the
/// probe's sources, minus `powermetrics`, which needs a root helper). Order
/// is preference: package/rail counters before polled power readings.
pub fn open_all() -> Vec<Box<dyn crate::meter::Meter>> {
    use crate::meter::Meter;
    let period = std::time::Duration::from_millis(50);
    let mut out: Vec<Box<dyn Meter>> = vec![];
    if let Ok(m) = crate::meter::RaplMeter::open() {
        out.push(Box::new(m));
    }
    #[cfg(target_os = "macos")]
    {
        use crate::ioreport::{IoReportMeter, Rail};
        for rail in [Rail::Cpu, Rail::Gpu, Rail::Ane, Rail::Dram] {
            if let Ok(m) = IoReportMeter::open(rail) {
                out.push(Box::new(m));
            }
        }
        if let Ok(m) = crate::smc::SmcMeter::open(period) {
            out.push(Box::new(m));
        }
    }
    #[cfg(feature = "nvml")]
    if let Ok(m) = crate::meter::NvmlMeter::open() {
        out.push(Box::new(m));
    }
    for d in crate::hwmon::scan(Path::new("/sys/class/hwmon")) {
        if let Ok(m) = crate::hwmon::open(&d, period) {
            out.push(m);
        }
    }
    for b in crate::supply::batteries(Path::new("/sys/class/power_supply")) {
        if let Ok(m) = crate::supply::open(&b, period) {
            out.push(Box::new(m));
        }
    }
    out
}
