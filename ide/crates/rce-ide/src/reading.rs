//! A package counter reading as a receipt term of spec Appendix D, D.11.
//!
//! `rce test measure` reads the machine's package energy counters around a
//! test run: RAPL on Linux, IOReport on macOS (Apple silicon). Both are
//! on-chip energy counters, so the term's basis is `reported` (D.11: "an
//! OS or on-chip energy counter: RAPL, NVML, IOReport"), never `measured`,
//! which D.11 keeps for a wall or shunt meter on a named rail. The term carries D.11's `provenance` = {`sampler`,
//! `counter`, `interval_s`, `idle_baseline_w`, `machine`, `boundary`,
//! `start_unix_ns`, `end_unix_ns`, `wraps`} and an `amortization`, and
//! nothing else, since D.11 reports a provenance with unknown fields as
//! invalid. The number of reads and the period between them have no field
//! in D.11 and are printed on a line of their own.
//!
//! [`Provenance::parse`] reads a provenance from JSON and checks it by D.11
//! rule 1: `rce test measured J PROVENANCE.json` and `reported J
//! PROVENANCE.json` take a figure from another meter only with one.
//!
//! On Linux the joules are the whole package (every top-level powercap zone
//! except `psys`). On macOS they are the sum of the CPU, GPU and ANE
//! domains (`ide_energy::ioreport::PACKAGE`; DRAM is outside, as it is
//! outside a RAPL package zone), each read from the AppleCLPC energy
//! counter where it moves, else from the `Energy Model` group, in the unit
//! each channel's label gives; the provenance's `counter` names every
//! channel summed. IOReport values are 64-bit deltas that do not reach
//! their maximum, and a negative one is refused as a failed read, so
//! `wraps` is 0 there. Either way the figure is not the test run's own
//! share: everything else the machine ran in the interval is in it. No idle
//! baseline is subtracted, so `idle_baseline_w` is 0 (D.11: "0 when no
//! baseline was subtracted").
//!
//! No counter, a failed read or a counter that did not move gives no figure:
//! the result says "unmeasured", never an estimate.
use ide_energy::ioreport::Channel;
use ide_energy::meter::{Meter, MeterError, RaplMeter};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How often the counters are read during a run. A zone's counter wraps
/// past its `max_energy_range_uj` (about 262 kJ on current parts); one read
/// a second sees every wrap unless a package draws more than that range in
/// one second.
pub const PERIOD: Duration = Duration::from_secs(1);

/// The powercap root of this machine.
pub const POWERCAP: &str = "/sys/class/powercap";

/// Where a reading comes from: the powercap root and the machine's name.
/// On macOS, [`Env::system`] reads IOReport in place of the powercap root;
/// any other root (the tests' fake trees) is read as powercap everywhere.
pub struct Env {
    pub powercap: PathBuf,
    /// The machine as the provenance names it; `None` reads it from /proc.
    pub machine: Option<String>,
    pub period: Duration,
}

impl Env {
    /// This machine: `/sys/class/powercap`, the name read from /proc.
    pub fn system() -> Env {
        Env { powercap: PathBuf::from(POWERCAP), machine: None, period: PERIOD }
    }

    /// The counters of this machine, not a fake tree.
    pub fn is_system(&self) -> bool {
        self.powercap == Path::new(POWERCAP)
    }

    fn machine(&self) -> String {
        self.machine.clone().unwrap_or_else(machine)
    }
}

/// Host name, CPU model and kernel release, from /proc (Linux), or host
/// name, chip, model and OS from `sysctl` (macOS, [`apple_machine`]).
#[cfg(not(target_os = "macos"))]
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

/// Host name, chip, model and OS from `sysctl` (macOS).
#[cfg(target_os = "macos")]
pub fn machine() -> String {
    apple_machine(ide_energy::ioreport::sysctl)
}

/// The machine of a macOS reading, from the sysctl strings `kern.hostname`,
/// `machdep.cpu.brand_string` (such as "Apple M5 Max"), `hw.model` (such as
/// "MacBookPro18,2"), `kern.osproductversion` (the version `sw_vers` prints) and
/// `kern.osrelease` (the Darwin kernel); `get` reads one by name.
pub fn apple_machine(get: impl Fn(&str) -> Option<String>) -> String {
    let or = |k: &str, d: &str| get(k).filter(|s| !s.is_empty()).unwrap_or_else(|| d.to_string());
    format!(
        "{}; cpu {}; model {}; macOS {}; kernel Darwin {}",
        or("kern.hostname", "unknown host"),
        or("machdep.cpu.brand_string", "unknown CPU"),
        or("hw.model", "unknown model"),
        or("kern.osproductversion", "unknown"),
        or("kern.osrelease", "unknown"),
    )
}

/// The sampler of a Linux reading.
pub const RAPL_SAMPLER: &str = "rce test measure (IDE build): Linux powercap RAPL energy_uj";

/// The start of the sampler of a macOS reading.
pub const IOREPORT_SAMPLER: &str = "rce test measure (IDE build): macOS IOReport";

/// The provenance fields that name the counter, not the interval: what a
/// counter reading's `sampler`, `counter`, `boundary` and `wraps` say, and
/// the words a "did not move" line uses for it.
#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub sampler: String,
    pub counter: String,
    pub boundary: String,
    pub wraps: u64,
    pub still: String,
}

/// The source of a macOS reading over `channels` (the ones
/// `ide_energy::ioreport::select_package` chose): the `Energy Model` group,
/// with the AppleCLPC counters when a channel comes from them; boundary
/// `package` (CPU + GPU + ANE); each channel with its unit label; wraps 0,
/// with the reason in the sampler.
pub fn ioreport_source(channels: &[Channel]) -> Source {
    let clpc = channels.iter().any(|c| c.group == "CLPC");
    let counter = channels.iter().map(Channel::describe).collect::<Vec<_>>().join("; ");
    Source {
        sampler: format!(
            "{IOREPORT_SAMPLER} Energy Model group{} via libIOReport (int64 deltas; a negative delta is refused, so no wraparound)",
            if clpc { " and AppleCLPC Energy Counters" } else { "" }
        ),
        still: counter.clone(),
        counter: format!("IOReport package = cpu + gpu + ane: {counter}"),
        boundary: "package".into(),
        wraps: 0,
    }
}

/// Why there is no figure.
#[derive(Debug, Clone, PartialEq)]
pub enum Unmeasured {
    /// No package counter could be opened (no RAPL zone; on macOS, no
    /// IOReport library, group or channel).
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

/// The provenance of a D.11 reading (rule 1): which sampler read which
/// counter, over what interval and when, against which idle baseline, on
/// which machine, inside which boundary, with how many wraparounds.
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    pub sampler: String,
    pub counter: String,
    pub interval_s: f64,
    pub idle_baseline_w: f64,
    pub machine: String,
    pub boundary: String,
    pub start_unix_ns: u64,
    pub end_unix_ns: u64,
    pub wraps: u64,
}

/// The fields of a provenance, in byte order (D.11 rule 1).
pub const FIELDS: [&str; 9] =
    ["boundary", "counter", "end_unix_ns", "idle_baseline_w", "interval_s", "machine", "sampler", "start_unix_ns", "wraps"];

fn js(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}

fn jf(x: f64) -> String {
    serde_json::to_string(&x).expect("a finite float serializes")
}

impl Provenance {
    /// D.11 rule 1: the four names non-empty, `interval_s` finite and > 0,
    /// `idle_baseline_w` finite and >= 0, `start_unix_ns` < `end_unix_ns`.
    /// `wraps` and the timestamps are unsigned by their type.
    pub fn check(&self) -> Result<(), String> {
        for (k, v) in [("sampler", &self.sampler), ("counter", &self.counter), ("machine", &self.machine), ("boundary", &self.boundary)] {
            if v.is_empty() {
                return Err(format!("`{k}` is empty"));
            }
        }
        if !(self.interval_s.is_finite() && self.interval_s > 0.0) {
            return Err(format!("`interval_s` {} is not finite and > 0", self.interval_s));
        }
        if !(self.idle_baseline_w.is_finite() && self.idle_baseline_w >= 0.0) {
            return Err(format!("`idle_baseline_w` {} is not finite and >= 0", self.idle_baseline_w));
        }
        if self.start_unix_ns >= self.end_unix_ns {
            return Err(format!("`start_unix_ns` {} is not before `end_unix_ns` {}", self.start_unix_ns, self.end_unix_ns));
        }
        Ok(())
    }

    /// A provenance from a JSON object with exactly the nine fields of
    /// D.11 rule 1, checked by that rule. The error says which field is
    /// missing, unknown, of the wrong type or out of range.
    pub fn parse(text: &str) -> Result<Provenance, String> {
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("not JSON ({e})"))?;
        let o = v.as_object().ok_or("not a JSON object")?;
        if let Some(k) = o.keys().find(|k| !FIELDS.contains(&k.as_str())) {
            return Err(format!("unknown field `{k}`"));
        }
        let get = |k: &str| o.get(k).ok_or(format!("missing field `{k}`"));
        let text = |k: &str| get(k)?.as_str().map(String::from).ok_or(format!("`{k}` is not a string"));
        let num = |k: &str| get(k)?.as_f64().ok_or(format!("`{k}` is not a number"));
        let int = |k: &str| get(k)?.as_u64().ok_or(format!("`{k}` is not an integer >= 0"));
        let p = Provenance {
            sampler: text("sampler")?,
            counter: text("counter")?,
            interval_s: num("interval_s")?,
            idle_baseline_w: num("idle_baseline_w")?,
            machine: text("machine")?,
            boundary: text("boundary")?,
            start_unix_ns: int("start_unix_ns")?,
            end_unix_ns: int("end_unix_ns")?,
            wraps: int("wraps")?,
        };
        p.check()?;
        Ok(p)
    }

    /// Canonical JSON: keys in byte order, no spaces.
    pub fn json(&self) -> String {
        format!(
            "{{\"boundary\":{},\"counter\":{},\"end_unix_ns\":{},\"idle_baseline_w\":{},\"interval_s\":{},\"machine\":{},\"sampler\":{},\"start_unix_ns\":{},\"wraps\":{}}}",
            js(&self.boundary),
            js(&self.counter),
            self.end_unix_ns,
            jf(self.idle_baseline_w),
            jf(self.interval_s),
            js(&self.machine),
            js(&self.sampler),
            self.start_unix_ns,
            self.wraps,
        )
    }
}

/// The two labels of a D.11 reading.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Basis {
    /// An OS or on-chip energy counter (RAPL, NVML, IOReport).
    Reported,
    /// A wall or shunt meter on a named rail.
    Measured,
}

impl Basis {
    pub fn word(self) -> &'static str {
        match self {
            Basis::Reported => "reported",
            Basis::Measured => "measured",
        }
    }
}

/// A D.11 reading: joules, a label and a checked provenance. The only
/// constructor, [`Term::new`], checks the provenance, so there is no term
/// from a bare number. One `rce test` run is one call, so the amortization
/// is `per_call`.
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    j: f64,
    basis: Basis,
    provenance: Provenance,
}

impl Term {
    /// A reading of `j` joules, finite and >= 0, with a provenance valid by
    /// D.11 rule 1.
    pub fn new(j: f64, basis: Basis, provenance: Provenance) -> Result<Term, String> {
        if !(j.is_finite() && j >= 0.0) {
            return Err(format!("J {j} is not a finite figure >= 0"));
        }
        provenance.check()?;
        Ok(Term { j, basis, provenance })
    }
    pub fn j(&self) -> f64 {
        self.j
    }
    pub fn basis(&self) -> Basis {
        self.basis
    }
    pub fn provenance(&self) -> &Provenance {
        &self.provenance
    }

    /// The term as canonical JSON (keys in byte order, no spaces): the
    /// receipt form of D.11, `{"j", "basis", "provenance", "amortization"}`.
    pub fn json(&self) -> String {
        format!(
            "{{\"amortization\":{{\"kind\":\"per_call\"}},\"basis\":\"{}\",\"j\":{},\"provenance\":{}}}",
            self.basis.word(),
            jf(self.j),
            self.provenance.json()
        )
    }

    /// The line `rce test` prints for the term.
    pub fn line(&self) -> String {
        format!("energy receipt: {}", self.json())
    }
}

/// A package-level counter reading: a D.11 term with basis `reported`, and
/// the two sampling facts that D.11's provenance has no field for.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub term: Term,
    pub samples: u64,
    pub period_s: f64,
}

impl Reading {
    /// The source named after a budget verdict.
    pub fn kind(&self) -> &'static str {
        if self.term.provenance.sampler.starts_with(IOREPORT_SAMPLER) {
            "reported: IOReport package counters, cpu + gpu + ane"
        } else {
            "reported: RAPL package counter, whole package"
        }
    }

    /// The lines `rce test measure` prints: the receipt, then the number of
    /// reads and their period.
    pub fn lines(&self) -> [String; 2] {
        [
            self.term.line(),
            format!("energy sampling: samples {} period_s {} (not fields of the D.11 provenance)", self.samples, jf(self.period_s)),
        ]
    }
}

/// The package counter read: RAPL, or IOReport on macOS.
enum Counter {
    Rapl(RaplMeter),
    #[cfg(target_os = "macos")]
    IoReport(ide_energy::ioreport::IoReportPackageMeter),
}

impl Counter {
    fn open(env: &Env) -> Result<Counter, MeterError> {
        #[cfg(target_os = "macos")]
        if env.is_system() {
            return ide_energy::ioreport::IoReportPackageMeter::open().map(Counter::IoReport);
        }
        RaplMeter::open_at(&env.powercap).map(Counter::Rapl)
    }

    fn sample(&mut self) -> Result<ide_energy::attrib::Sample, MeterError> {
        match self {
            Counter::Rapl(m) => m.sample(),
            #[cfg(target_os = "macos")]
            Counter::IoReport(m) => m.sample(),
        }
    }

    fn source(&self) -> Source {
        match self {
            Counter::Rapl(m) => {
                let zones = m.zones();
                Source {
                    sampler: RAPL_SAMPLER.into(),
                    counter: zones
                        .iter()
                        .map(|z| format!("powercap {} {} ({}, max_energy_range_uj {})", z.zone, z.domain, z.energy_file.display(), z.max_energy_range_uj))
                        .collect::<Vec<_>>()
                        .join("; "),
                    boundary: "package".into(),
                    wraps: zones.iter().map(|z| z.wraps).sum(),
                    still: zones.iter().map(|z| z.zone.as_str()).collect::<Vec<_>>().join(", "),
                }
            }
            #[cfg(target_os = "macos")]
            Counter::IoReport(m) => {
                let mut s = ioreport_source(m.channels());
                s.wraps = m.wraps();
                s
            }
        }
    }
}

fn unmeasured(e: MeterError) -> Unmeasured {
    match e {
        MeterError::Stale(..) => Unmeasured::Stale(e.to_string()),
        _ => Unmeasured::ReadFailed(e.to_string()),
    }
}

struct Done {
    meter: Counter,
    last: Result<ide_energy::attrib::Sample, MeterError>,
    samples: u64,
    error: Option<MeterError>,
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
    /// Open the package counters and take the first read.
    pub fn start(env: &Env) -> Result<Sampling, Unmeasured> {
        let mut meter = Counter::open(env).map_err(|e| Unmeasured::NoCounter(e.to_string()))?;
        let first = meter.sample().map_err(unmeasured)?;
        let started = Instant::now();
        let (stop, rx) = mpsc::channel::<()>();
        let period = env.period;
        let worker = std::thread::spawn(move || {
            let mut samples = 1u64;
            let mut error = None;
            // until `finish` sends or drops the sender
            while let Err(mpsc::RecvTimeoutError::Timeout) = rx.recv_timeout(period) {
                samples += 1;
                if let Err(e) = meter.sample() {
                    error.get_or_insert(e);
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
        let d = self.worker.join().map_err(|_| Unmeasured::ReadFailed("the sampling thread panicked".into()))?;
        // from the first read to the last, on the monotonic clock
        let elapsed = self.started.elapsed().as_secs_f64();
        if let Some(e) = d.error {
            return Err(unmeasured(e));
        }
        let last = d.last.map_err(unmeasured)?;
        let src = d.meter.source();
        let j = last.joules - self.first.joules;
        if j <= 0.0 || !j.is_finite() {
            return Err(Unmeasured::Stale(format!("{} over {elapsed:.3} s", src.still)));
        }
        if elapsed <= 0.0 {
            return Err(Unmeasured::ReadFailed("zero sampling interval".into()));
        }
        if last.t_ns <= self.first.t_ns {
            return Err(Unmeasured::ReadFailed(format!("the system clock did not advance over the run ({} ns to {} ns)", self.first.t_ns, last.t_ns)));
        }
        let provenance = Provenance {
            sampler: src.sampler,
            counter: src.counter,
            interval_s: elapsed,
            idle_baseline_w: 0.0,
            machine: self.machine,
            boundary: src.boundary,
            start_unix_ns: self.first.t_ns,
            end_unix_ns: last.t_ns,
            wraps: src.wraps,
        };
        let term = Term::new(j, Basis::Reported, provenance).map_err(Unmeasured::ReadFailed)?;
        Ok(Reading { term, samples: d.samples, period_s: self.period.as_secs_f64() })
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
        let r0 = s.finish().unwrap();
        let (g, t) = (r0.term.provenance(), &r0.term);
        assert!((t.j() - 2.000001).abs() < 1e-9, "{}", t.j());
        assert_eq!(t.basis(), Basis::Reported);
        assert_eq!(g.wraps, 1);
        assert_eq!(g.boundary, "package");
        assert_eq!(g.idle_baseline_w, 0.0);
        assert_eq!(g.machine, "test-host; cpu Test CPU; kernel Linux 0.0");
        assert_eq!(g.counter, format!("powercap intel-rapl:0 package-0 ({}, max_energy_range_uj 1000000000)", r.join("intel-rapl:0/energy_uj").display()));
        assert!(g.interval_s > 0.0 && g.end_unix_ns > g.start_unix_ns && r0.samples == 2);
        let v: serde_json::Value = serde_json::from_str(&t.json()).unwrap();
        assert_eq!(v["basis"], "reported");
        assert_eq!(v["amortization"], serde_json::json!({"kind": "per_call"}));
        let p = v["provenance"].as_object().unwrap();
        let keys: Vec<&str> = p.keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, FIELDS);
        assert_eq!(v["provenance"]["wraps"], 1);
        assert_eq!(v.as_object().unwrap().len(), 4);
        // the printed provenance reads back as the same provenance
        assert_eq!(&Provenance::parse(&v["provenance"].to_string()).unwrap(), g);
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

    pub(crate) fn prov() -> Provenance {
        Provenance {
            sampler: "rce test measure (IDE build): Linux powercap RAPL energy_uj".into(),
            counter: "powercap intel-rapl:0 package-0 (/sys/class/powercap/intel-rapl:0/energy_uj, max_energy_range_uj 262143328850)".into(),
            interval_s: 2.25,
            idle_baseline_w: 0.0,
            machine: "h; cpu \"X\"; kernel Linux 6.1".into(),
            boundary: "package".into(),
            start_unix_ns: 10,
            end_unix_ns: 2_250_000_010,
            wraps: 0,
        }
    }

    #[test]
    fn formatting_is_deterministic() {
        let g = Reading { term: Term::new(12.5, Basis::Reported, prov()).unwrap(), samples: 4, period_s: 1.0 };
        let want = [
            r#"energy receipt: {"amortization":{"kind":"per_call"},"basis":"reported","j":12.5,"provenance":{"boundary":"package","counter":"powercap intel-rapl:0 package-0 (/sys/class/powercap/intel-rapl:0/energy_uj, max_energy_range_uj 262143328850)","end_unix_ns":2250000010,"idle_baseline_w":0.0,"interval_s":2.25,"machine":"h; cpu \"X\"; kernel Linux 6.1","sampler":"rce test measure (IDE build): Linux powercap RAPL energy_uj","start_unix_ns":10,"wraps":0}}"#.to_string(),
            "energy sampling: samples 4 period_s 1.0 (not fields of the D.11 provenance)".to_string(),
        ];
        assert_eq!(g.lines(), want);
        assert_eq!(g.clone().lines(), g.lines());
        let m = Term::new(3.0, Basis::Measured, prov()).unwrap();
        assert!(m.line().starts_with(r#"energy receipt: {"amortization":{"kind":"per_call"},"basis":"measured","j":3.0,"provenance":{"boundary":"package","#), "{}", m.line());
    }

    #[test]
    fn apple_machine_from_sysctl() {
        let mac = |k: &str| {
            Some(
                match k {
                    "kern.hostname" => "studio",
                    "machdep.cpu.brand_string" => "Apple M1 Max",
                    "hw.model" => "MacBookPro18,2",
                    "kern.osproductversion" => "15.1",
                    "kern.osrelease" => "24.1.0",
                    _ => return None,
                }
                .to_string(),
            )
        };
        assert_eq!(apple_machine(mac), "studio; cpu Apple M1 Max; model MacBookPro18,2; macOS 15.1; kernel Darwin 24.1.0");
        assert_eq!(apple_machine(|_| None), "unknown host; cpu unknown CPU; model unknown model; macOS unknown; kernel Darwin unknown");
    }

    /// The receipt of a macOS reading, from the channels the M5 Max gives
    /// (`ide-energy` tests `m5max_reads_clpc_and_sums_cpu_gpu_ane`).
    #[test]
    fn ioreport_receipt_is_deterministic() {
        use ide_energy::ioreport::{Row, select_package};
        let rows = [
            Row::new("Energy Model", "CPU Energy", 0, "mJ"),
            Row::new("Energy Model", "GPU Energy", 39_224_054_504, "nJ"),
            Row::new("Energy Model", "ANE0", 0, "mJ"),
            Row::new("CLPC", "CPU Energy", 30_992_368_044, "nJ"),
            Row::new("CLPC", "GPU Energy", 39_201_512_326, "nJ"),
            Row::new("CLPC", "ANE", 0, "nJ"),
        ];
        let src = ioreport_source(&select_package(&rows).unwrap());
        assert_eq!(src.wraps, 0);
        assert_eq!(src.boundary, "package");
        let p = Provenance {
            sampler: src.sampler,
            counter: src.counter,
            interval_s: 4.0,
            idle_baseline_w: 0.0,
            machine: "HOST; cpu Apple M5 Max; model MODEL; macOS 27.0.1; kernel Darwin RELEASE".into(),
            boundary: src.boundary,
            start_unix_ns: 1_000_000_000,
            end_unix_ns: 5_000_000_000,
            wraps: src.wraps,
        };
        let g = Reading { term: Term::new(70.19388037, Basis::Reported, p).unwrap(), samples: 5, period_s: 1.0 };
        assert_eq!(
            g.lines()[0],
            r#"energy receipt: {"amortization":{"kind":"per_call"},"basis":"reported","j":70.19388037,"provenance":{"boundary":"package","counter":"IOReport package = cpu + gpu + ane: cpu: CLPC \"CPU Energy\" nJ; gpu: CLPC \"GPU Energy\" nJ; ane: CLPC \"ANE\" nJ","end_unix_ns":5000000000,"idle_baseline_w":0.0,"interval_s":4.0,"machine":"HOST; cpu Apple M5 Max; model MODEL; macOS 27.0.1; kernel Darwin RELEASE","sampler":"rce test measure (IDE build): macOS IOReport Energy Model group and AppleCLPC Energy Counters via libIOReport (int64 deltas; a negative delta is refused, so no wraparound)","start_unix_ns":1000000000,"wraps":0}}"#
        );
        assert_eq!(g.kind(), "reported: IOReport package counters, cpu + gpu + ane");
        // all nine fields, read back as the same provenance
        let v: serde_json::Value = serde_json::from_str(&g.term.json()).unwrap();
        let keys: Vec<&str> = v["provenance"].as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(keys, FIELDS);
        assert_eq!(&Provenance::parse(&v["provenance"].to_string()).unwrap(), g.term.provenance());
        // Energy Model only (no CLPC driver known): the sampler says so
        let em: Vec<Row> = rows.iter().filter(|r| r.group == "Energy Model").cloned().map(|r| Row { raw: r.raw.max(1), ..r }).collect();
        let src = ioreport_source(&select_package(&em).unwrap());
        assert_eq!(src.sampler, "rce test measure (IDE build): macOS IOReport Energy Model group via libIOReport (int64 deltas; a negative delta is refused, so no wraparound)");
        assert_eq!(src.counter, r#"IOReport package = cpu + gpu + ane: cpu: Energy Model "CPU Energy" mJ; gpu: Energy Model "GPU Energy" nJ; ane: Energy Model "ANE0" mJ"#);
        // a Linux reading keeps its own label
        assert_eq!(Reading { term: Term::new(1.0, Basis::Reported, prov()).unwrap(), samples: 2, period_s: 1.0 }.kind(), "reported: RAPL package counter, whole package");
    }

    #[test]
    fn provenance_rule_1() {
        let good = prov().json();
        assert_eq!(Provenance::parse(&good).unwrap(), prov());
        let with = |k: &str, v: serde_json::Value| {
            let mut o: serde_json::Value = serde_json::from_str(&good).unwrap();
            o[k] = v;
            Provenance::parse(&o.to_string())
        };
        let without = |k: &str| {
            let mut o: serde_json::Value = serde_json::from_str(&good).unwrap();
            o.as_object_mut().unwrap().remove(k);
            Provenance::parse(&o.to_string())
        };
        // every field is required, for both labels
        for k in FIELDS {
            assert_eq!(without(k).unwrap_err(), format!("missing field `{k}`"));
        }
        assert_eq!(with("samples", 4.into()).unwrap_err(), "unknown field `samples`");
        assert_eq!(with("sampler", "".into()).unwrap_err(), "`sampler` is empty");
        assert_eq!(with("boundary", 1.into()).unwrap_err(), "`boundary` is not a string");
        assert_eq!(with("interval_s", 0.into()).unwrap_err(), "`interval_s` 0 is not finite and > 0");
        assert_eq!(with("idle_baseline_w", (-1.0).into()).unwrap_err(), "`idle_baseline_w` -1 is not finite and >= 0");
        assert_eq!(with("wraps", (-1).into()).unwrap_err(), "`wraps` is not an integer >= 0");
        assert_eq!(with("wraps", 1.5.into()).unwrap_err(), "`wraps` is not an integer >= 0");
        assert_eq!(with("end_unix_ns", 10.into()).unwrap_err(), "`start_unix_ns` 10 is not before `end_unix_ns` 10");
        assert_eq!(with("idle_baseline_w", 0.into()).unwrap(), prov());
        assert!(Provenance::parse("[]").unwrap_err().starts_with("not a JSON object"));
        assert!(Provenance::parse("2").unwrap_err().starts_with("not a JSON object"));
        assert!(Provenance::parse("x").unwrap_err().starts_with("not JSON ("));
        // no term from a bare number, nor from a figure that is not one
        assert!(Term::new(f64::NAN, Basis::Measured, prov()).is_err());
        assert!(Term::new(-1.0, Basis::Measured, prov()).is_err());
        assert!(Term::new(1.0, Basis::Measured, Provenance { wraps: 0, end_unix_ns: 0, ..prov() }).is_err());
    }
}
