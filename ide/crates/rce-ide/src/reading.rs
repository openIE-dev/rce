//! A package RAPL reading as a receipt term of spec Appendix D, D.11.
//!
//! `rce test measure` reads the machine's RAPL package counters around a
//! test run. RAPL is an on-chip energy counter, so the term's basis is
//! `reported` (D.11: "an OS or on-chip energy counter: RAPL, NVML,
//! IOReport"), never `measured`, which D.11 keeps for a wall or shunt meter
//! on a named rail. The term carries D.11's `provenance` = {`sampler`,
//! `counter`, `interval_s`, `idle_baseline_w`, `machine`, `boundary`} and an
//! `amortization`, and nothing else, since D.11 reports a provenance with
//! unknown fields as invalid. The sampling start and end, the sample count
//! and the wraparounds seen are printed on a line of their own.
//!
//! The joules are the whole package (every top-level powercap zone except
//! `psys`), not the test run's own share: everything else the machine ran in
//! the interval is in them. No idle baseline is subtracted, so
//! `idle_baseline_w` is 0 (D.11: "0 when no baseline was subtracted").
//!
//! No counter, a failed read or a counter that did not move gives no figure:
//! the result says "unmeasured", never an estimate.
use ide_energy::meter::{Meter, MeterError, RaplMeter};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How often the counters are read during a run. A zone's counter wraps
/// past its `max_energy_range_uj` (about 262 kJ on current parts); one read
/// a second sees every wrap unless a package draws more than that range in
/// one second.
pub const PERIOD: Duration = Duration::from_secs(1);

/// Where a reading comes from: the powercap root and the machine's name.
pub struct Env {
    pub powercap: PathBuf,
    /// The machine as the provenance names it; `None` reads it from /proc.
    pub machine: Option<String>,
    pub period: Duration,
}

impl Env {
    /// This machine: `/sys/class/powercap`, the name read from /proc.
    pub fn system() -> Env {
        Env { powercap: PathBuf::from("/sys/class/powercap"), machine: None, period: PERIOD }
    }

    fn machine(&self) -> String {
        self.machine.clone().unwrap_or_else(machine)
    }
}

/// Host name, CPU model and kernel release, from /proc.
pub fn machine() -> String {
    let read = |p: &str| std::fs::read_to_string(p).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let host = read("/proc/sys/kernel/hostname").unwrap_or_else(|| "unknown host".into());
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split_once(':'))
                .map(|(_, v)| v.trim().to_string())
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown CPU".into());
    let kernel = read("/proc/sys/kernel/osrelease").unwrap_or_else(|| "unknown".into());
    format!("{host}; cpu {cpu}; kernel Linux {kernel}")
}

/// Why there is no figure.
#[derive(Debug, Clone, PartialEq)]
pub enum Unmeasured {
    /// No RAPL package zone could be opened.
    NoCounter(String),
    /// A read failed during the run.
    ReadFailed(String),
    /// The counters did not move over the run.
    Stale(String),
}

impl Unmeasured {
    /// The line `rce test measure` prints in place of a receipt.
    pub fn line(&self) -> String {
        match self {
            Unmeasured::NoCounter(w) => format!("energy: unmeasured: no counter available ({w})"),
            Unmeasured::ReadFailed(w) => format!("energy: unmeasured: counter read failed ({w})"),
            Unmeasured::Stale(w) => format!("energy: unmeasured: the counter did not move ({w})"),
        }
    }
}

/// A package-level counter reading: a D.11 term with basis `reported`.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub j: f64,
    // D.11 provenance
    pub sampler: String,
    pub counter: String,
    pub interval_s: f64,
    pub idle_baseline_w: f64,
    pub machine: String,
    pub boundary: String,
    // outside D.11's provenance
    pub start_unix_ns: u64,
    pub end_unix_ns: u64,
    pub samples: u64,
    pub period_s: f64,
    pub wraps: u64,
}

fn js(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

fn jf(x: f64) -> String {
    serde_json::to_string(&x).expect("a finite float serializes")
}

impl Reading {
    /// The term as canonical JSON (keys in byte order, no spaces): the
    /// receipt form of D.11, `{"j", "basis", "provenance", "amortization"}`.
    /// One `rce test` run is one call, so the amortization is `per_call`.
    pub fn term_json(&self) -> String {
        format!(
            "{{\"amortization\":{{\"kind\":\"per_call\"}},\"basis\":\"reported\",\"j\":{},\"provenance\":{{\"boundary\":{},\"counter\":{},\"idle_baseline_w\":{},\"interval_s\":{},\"machine\":{},\"sampler\":{}}}}}",
            jf(self.j),
            js(&self.boundary),
            js(&self.counter),
            jf(self.idle_baseline_w),
            jf(self.interval_s),
            js(&self.machine),
            js(&self.sampler),
        )
    }

    /// The lines `rce test measure` prints: the receipt, then the sampling
    /// facts that D.11's provenance has no field for.
    pub fn lines(&self) -> [String; 2] {
        [
            format!("energy receipt: {}", self.term_json()),
            format!(
                "energy sampling: start_unix_ns {} end_unix_ns {} samples {} period_s {} wraps {} (not fields of the D.11 provenance)",
                self.start_unix_ns,
                self.end_unix_ns,
                self.samples,
                jf(self.period_s),
                self.wraps
            ),
        ]
    }
}

struct Done {
    meter: RaplMeter,
    last: Result<ide_energy::attrib::Sample, MeterError>,
    samples: u64,
    error: Option<String>,
}

/// The counters being read around a run.
pub struct Sampling {
    first: ide_energy::attrib::Sample,
    started: Instant,
    machine: String,
    period: Duration,
    stop: mpsc::Sender<()>,
    worker: std::thread::JoinHandle<Done>,
}

impl Sampling {
    /// Open the package zones and take the first read.
    pub fn start(env: &Env) -> Result<Sampling, Unmeasured> {
        let mut meter = RaplMeter::open_at(&env.powercap).map_err(|e| Unmeasured::NoCounter(e.to_string()))?;
        let first = meter.sample().map_err(|e| Unmeasured::ReadFailed(e.to_string()))?;
        let started = Instant::now();
        let (stop, rx) = mpsc::channel::<()>();
        let period = env.period;
        let worker = std::thread::spawn(move || {
            let mut samples = 1u64;
            let mut error = None;
            loop {
                match rx.recv_timeout(period) {
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        samples += 1;
                        if let Err(e) = meter.sample() {
                            error.get_or_insert(e.to_string());
                        }
                    }
                    _ => break,
                }
            }
            let last = meter.sample();
            samples += 1;
            Done { meter, last, samples, error }
        });
        Ok(Sampling { first, started, machine: env.machine(), period, stop, worker })
    }

    /// Take the last read and make the receipt, or say why there is none.
    pub fn finish(self) -> Result<Reading, Unmeasured> {
        let _ = self.stop.send(());
        let elapsed = self.started.elapsed().as_secs_f64();
        let d = self.worker.join().map_err(|_| Unmeasured::ReadFailed("the sampling thread panicked".into()))?;
        if let Some(e) = d.error {
            return Err(Unmeasured::ReadFailed(e));
        }
        let last = d.last.map_err(|e| Unmeasured::ReadFailed(e.to_string()))?;
        let zones = d.meter.zones();
        let j = last.joules - self.first.joules;
        if j <= 0.0 || !j.is_finite() {
            return Err(Unmeasured::Stale(format!("{} over {elapsed:.3} s", zones.iter().map(|z| z.zone.as_str()).collect::<Vec<_>>().join(", "))));
        }
        if elapsed <= 0.0 {
            return Err(Unmeasured::ReadFailed("zero sampling interval".into()));
        }
        let counter = zones
            .iter()
            .map(|z| format!("powercap {} {} ({}, max_energy_range_uj {})", z.zone, z.domain, z.energy_file.display(), z.max_energy_range_uj))
            .collect::<Vec<_>>()
            .join("; ");
        Ok(Reading {
            j,
            sampler: "rce test measure (IDE build): Linux powercap RAPL energy_uj".into(),
            counter,
            interval_s: elapsed,
            idle_baseline_w: 0.0,
            machine: self.machine,
            boundary: "package".into(),
            start_unix_ns: self.first.t_ns,
            end_unix_ns: last.t_ns,
            samples: d.samples,
            period_s: self.period.as_secs_f64(),
            wraps: zones.iter().map(|z| z.wraps).sum(),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fs;

    pub(crate) fn zone(root: &std::path::Path, n: &str, name: &str, e: u64, max: u64) {
        let d = root.join(n);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("name"), name).unwrap();
        fs::write(d.join("energy_uj"), e.to_string()).unwrap();
        fs::write(d.join("max_energy_range_uj"), max.to_string()).unwrap();
    }

    fn env(root: &std::path::Path) -> Env {
        // test-only: a fake powercap tree and a fixed machine name
        Env { powercap: root.to_path_buf(), machine: Some("test-host; cpu Test CPU; kernel Linux 0.0".into()), period: Duration::from_secs(3600) }
    }

    #[test]
    fn receipt_fields_and_wraparound() {
        let t = tempfile::tempdir().unwrap();
        let r = t.path();
        zone(r, "intel-rapl:0", "package-0", 999_000_000, 1_000_000_000);
        zone(r, "intel-rapl:0:0", "core", 1, 10); // sub-zone: not read
        let s = Sampling::start(&env(r)).unwrap();
        // the counter wraps: 999_000_000 -> 1_000_000 over a range of 1e9
        fs::write(r.join("intel-rapl:0/energy_uj"), "1000000").unwrap();
        let g = s.finish().unwrap();
        assert!((g.j - 2.000001).abs() < 1e-9, "{}", g.j);
        assert_eq!(g.wraps, 1);
        assert_eq!(g.boundary, "package");
        assert_eq!(g.idle_baseline_w, 0.0);
        assert_eq!(g.machine, "test-host; cpu Test CPU; kernel Linux 0.0");
        assert_eq!(g.counter, format!("powercap intel-rapl:0 package-0 ({}, max_energy_range_uj 1000000000)", r.join("intel-rapl:0/energy_uj").display()));
        assert!(g.interval_s > 0.0 && g.end_unix_ns >= g.start_unix_ns && g.samples == 2);
        let v: serde_json::Value = serde_json::from_str(&g.term_json()).unwrap();
        assert_eq!(v["basis"], "reported");
        assert_eq!(v["amortization"], serde_json::json!({"kind": "per_call"}));
        let p = v["provenance"].as_object().unwrap();
        let keys: Vec<&str> = p.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, ["boundary", "counter", "idle_baseline_w", "interval_s", "machine", "sampler"]);
        assert_eq!(v.as_object().unwrap().len(), 4);
    }

    #[test]
    fn no_counter_is_unmeasured() {
        let t = tempfile::tempdir().unwrap();
        let e = Sampling::start(&env(&t.path().join("nope"))).err().unwrap();
        assert!(matches!(e, Unmeasured::NoCounter(_)));
        assert!(e.line().starts_with("energy: unmeasured: no counter available ("), "{}", e.line());
    }

    #[test]
    fn still_counter_is_unmeasured() {
        let t = tempfile::tempdir().unwrap();
        zone(t.path(), "intel-rapl:0", "package-0", 5, 1000);
        let e = Sampling::start(&env(t.path())).unwrap().finish().err().unwrap();
        assert!(matches!(e, Unmeasured::Stale(_)), "{e:?}");
    }

    #[test]
    fn formatting_is_deterministic() {
        let g = Reading {
            j: 12.5,
            sampler: "rce test measure (IDE build): Linux powercap RAPL energy_uj".into(),
            counter: "powercap intel-rapl:0 package-0 (/sys/class/powercap/intel-rapl:0/energy_uj, max_energy_range_uj 262143328850)".into(),
            interval_s: 2.25,
            idle_baseline_w: 0.0,
            machine: "h; cpu \"X\"; kernel Linux 6.1".into(),
            boundary: "package".into(),
            start_unix_ns: 10,
            end_unix_ns: 2_250_000_010,
            samples: 4,
            period_s: 1.0,
            wraps: 0,
        };
        let want = [
            r#"energy receipt: {"amortization":{"kind":"per_call"},"basis":"reported","j":12.5,"provenance":{"boundary":"package","counter":"powercap intel-rapl:0 package-0 (/sys/class/powercap/intel-rapl:0/energy_uj, max_energy_range_uj 262143328850)","idle_baseline_w":0.0,"interval_s":2.25,"machine":"h; cpu \"X\"; kernel Linux 6.1","sampler":"rce test measure (IDE build): Linux powercap RAPL energy_uj"}}"#.to_string(),
            "energy sampling: start_unix_ns 10 end_unix_ns 2250000010 samples 4 period_s 1.0 wraps 0 (not fields of the D.11 provenance)".to_string(),
        ];
        assert_eq!(g.lines(), want);
        assert_eq!(g.clone().lines(), g.lines());
    }
}
