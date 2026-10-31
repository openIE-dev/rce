//! `rce-ide telemetry calibrate`: the fixed calibration workloads (real
//! rce-mcp checker runs over bundled RCE programs) and the command.
//!
//! Workloads (each one measured window, repeated `reps N` times, interleaved):
//! - `empty`: an empty window (the per-action floor: meter reads and
//!   bookkeeping; all features 0). Not a timed idle: the model has no
//!   duration term, so a long idle window would be fitted as overhead.
//! - `check <set> xK`: K `rce_check` calls on a workspace holding one of
//!   three program sets (1, 2 or 3 bundled examples), so passes and form
//!   nodes vary independently.
//! - `test large x20`: `rce_test` (checks and counts tests).
//!
//! K runs from 10 to 160 so windows are tens of milliseconds and more, well
//! above RAPL's update interval (~1 ms) and polled meters' 50 ms period.
//!
//! Each workload's features are what rce-mcp itself records for one call
//! (read from its receipt in a warm-up call), times K: the same features a
//! real `rce_check` receipt carries, so the fit applies to them.
//! Only the `checker` gear is exercised; agent gears (`g4`, tokens) need real
//! agent turns and stay uncalibrated (they predict nothing).
use ide_energy::calib::{Calibration, Plan, Workload, calibrate, save};
use ide_energy::meter::Meter;
use ide_energy::sim::{Features, LinearSim};
use ide_receipt::{Ledger, chain::read_jsonl, generate_key};
use rce_mcp::{Server, ws::Workspace};
use serde_json::json;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;

const PROGRAMS: [(&str, &str); 3] = [
    ("counter.rce", include_str!("../calib/01_counter.rce")),
    (
        "bank_account.rce",
        include_str!("../calib/04_bank_account.rce"),
    ),
    ("iterators.rce", include_str!("../calib/50_iterators.rce")),
];

struct Ws {
    server: Rc<RefCell<Server>>,
    ledger: PathBuf,
}

fn workspace(root: &Path, files: usize) -> Result<Ws, String> {
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    for (n, src) in &PROGRAMS[..files] {
        std::fs::write(root.join(n), src).map_err(|e| e.to_string())?;
    }
    let ws = Workspace::new(root).map_err(|e| e.to_string())?;
    let ledger = root.join(".rce-ide/calibration.jsonl");
    std::fs::create_dir_all(ledger.parent().unwrap()).map_err(|e| e.to_string())?;
    let l = Ledger::open(&ledger, generate_key().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Ok(Ws {
        server: Rc::new(RefCell::new(Server::new(ws, l, LinearSim::default(), None))),
        ledger,
    })
}

fn call(s: &RefCell<Server>, tool: &str) -> Result<(), String> {
    let r = s
        .borrow_mut()
        .handle(&json!({"jsonrpc":"2.0","id":1,"method":"tools/call",
            "params":{"name": tool, "arguments": {}}}))
        .ok_or("no reply")?;
    if r.get("error").is_some() || r["result"]["isError"] == json!(true) {
        return Err(format!("{tool}: {r}"));
    }
    Ok(())
}

struct Empty;
impl Workload for Empty {
    fn name(&self) -> String {
        "empty".into()
    }
    fn run(&mut self) -> Result<Features, String> {
        Ok(Features {
            gear: "checker".into(),
            ..Default::default()
        })
    }
}

struct Check {
    name: String,
    server: Rc<RefCell<Server>>,
    tool: &'static str,
    times: u64,
    one: Features,
}
impl Workload for Check {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn run(&mut self) -> Result<Features, String> {
        for _ in 0..self.times {
            call(&self.server, self.tool)?;
        }
        Ok(Features {
            form_nodes: self.one.form_nodes * self.times,
            checker_passes: self.one.checker_passes * self.times,
            wasi_fuel: self.one.wasi_fuel * self.times,
            ..self.one.clone()
        })
    }
}

/// Features rce-mcp records for one call (warm-up; also warms caches).
fn one_call(w: &Ws, tool: &'static str) -> Result<Features, String> {
    call(&w.server, tool)?;
    let rs = read_jsonl(&w.ledger).map_err(|e| e.to_string())?;
    let d = &rs.last().ok_or("no receipt")?.body.draft.detail;
    serde_json::from_value(d["features"].clone()).map_err(|e| format!("features: {e}"))
}

/// The fixed workload set, in temporary workspaces under `dir`.
pub fn workloads(dir: &Path) -> Result<Vec<Box<dyn Workload>>, String> {
    let mut out: Vec<Box<dyn Workload>> = vec![Box::new(Empty)];
    let sets = [("small", 1usize), ("medium", 2), ("large", 3)];
    let reps: [&[u64]; 3] = [&[10, 40, 160], &[10, 40], &[10, 40, 80]];
    let mut large = None;
    for ((name, files), times) in sets.iter().zip(reps) {
        let w = workspace(&dir.join(name), *files)?;
        let one = one_call(&w, "rce_check")?;
        for &t in times {
            out.push(Box::new(Check {
                name: format!("check {name} x{t}"),
                server: w.server.clone(),
                tool: "rce_check",
                times: t,
                one: one.clone(),
            }));
        }
        if *name == "large" {
            large = Some(w);
        }
    }
    let w = large.expect("large set");
    let one = one_call(&w, "rce_test")?;
    out.push(Box::new(Check {
        name: "test large x20".into(),
        server: w.server.clone(),
        tool: "rce_test",
        times: 20,
        one,
    }));
    Ok(out)
}

/// Where the IDE looks for a calibration: `$RCE_IDE_CALIBRATION`, else
/// `$XDG_CONFIG_HOME/rce-ide/calibration.json`, else
/// `~/.config/rce-ide/calibration.json`.
pub fn default_path() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RCE_IDE_CALIBRATION") {
        return Some(p.into());
    }
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("rce-ide/calibration.json"))
}

/// A fake meter for exercising the command without hardware: constant
/// `watts` over wall time. Its source tag is `fake`, recorded in the file.
pub struct FakeMeter {
    watts: f64,
    t0: std::time::Instant,
}
impl FakeMeter {
    pub fn new(watts: f64) -> Self {
        FakeMeter {
            watts,
            t0: std::time::Instant::now(),
        }
    }
}
impl Meter for FakeMeter {
    fn source(&self) -> &'static str {
        "fake"
    }
    fn sample(&mut self) -> Result<ide_energy::attrib::Sample, ide_energy::meter::MeterError> {
        Ok(ide_energy::attrib::Sample {
            t_ns: ide_energy::meter::now_ns(),
            joules: self.watts * self.t0.elapsed().as_secs_f64(),
        })
    }
}

/// Run the workloads under `meters` and save. Returns the calibration and
/// the files written.
pub fn run(
    meters: &mut [Box<dyn Meter>],
    plan: &Plan,
    out: &Path,
) -> Result<(Calibration, Vec<PathBuf>), String> {
    let mut b = [0u8; 8];
    getrandom::fill(&mut b).map_err(|e| e.to_string())?;
    let dir = std::env::temp_dir().join(format!("rce-ide-calibrate-{}", hex::encode(b)));
    let r = (|| {
        let mut w = workloads(&dir)?;
        let c = calibrate(meters, &mut w, plan).map_err(|e| e.to_string())?;
        let files = save(&c, out).map_err(|e| format!("{}: {e}", out.display()))?;
        Ok((c, files))
    })();
    let _ = std::fs::remove_dir_all(&dir);
    r
}

/// The command line of `rce-ide telemetry calibrate`, as data
/// (spec/proposals/positional-cli.md PC-8): words only.
pub const CALIBRATE: rce_spec_check::args::Command = rce_spec_check::args::Command {
    path: &["telemetry", "calibrate"],
    words: &[
        rce_spec_check::args::Word::set("out", &["FILE"]),
        rce_spec_check::args::Word::set("reps", &["N"]),
        rce_spec_check::args::Word::set("fake-meter", &["WATTS"]),
    ],
    operands: "",
    min: 0,
    max: 0,
    help: "run the fixed checker workloads under every discovered meter and fit the simulator\n(fake-meter: a constant-power stand-in, for testing without hardware)",
};

/// The flags `telemetry calibrate` had (positional-cli.md 1.4).
pub const REMOVED: &[rce_spec_check::args::Removed] = &[
    rce_spec_check::args::Removed { flag: "--out", paths: &["telemetry calibrate"], values: 1, fate: rce_spec_check::args::Fate::Word },
    rce_spec_check::args::Removed { flag: "--reps", paths: &["telemetry calibrate"], values: 1, fate: rce_spec_check::args::Fate::Word },
    rce_spec_check::args::Removed { flag: "--fake-meter", paths: &["telemetry calibrate"], values: 1, fate: rce_spec_check::args::Fate::Word },
];

/// The command; returns the exit code.
pub fn command(p: &rce_spec_check::args::Parsed) -> i32 {
    let out = p.value("out").map(PathBuf::from).or_else(default_path);
    let mut plan = Plan {
        reps: 5,
        ..Plan::default()
    };
    if let Some(r) = p.value("reps") {
        match r.parse() {
            Ok(n) if n > 0 => plan.reps = n,
            _ => {
                eprintln!("reps {r}: N is a count of repetitions, at least 1");
                return 2;
            }
        }
    }
    let fake = p.value("fake-meter").and_then(|w| w.parse::<f64>().ok());
    let Some(out) = out else {
        eprintln!("rce-ide telemetry calibrate: no out FILE and no HOME to default to");
        return 2;
    };
    let mut meters: Vec<Box<dyn Meter>> = match fake {
        Some(w) => vec![Box::new(FakeMeter::new(w))],
        None => ide_energy::discover::open_all(),
    };
    if meters.is_empty() {
        eprintln!(
            "rce-ide telemetry calibrate: no energy meter works here; see `rce-ide telemetry probe` (and `setup`)"
        );
        return 1;
    }
    let names: Vec<_> = meters.iter().map(|m| m.source()).collect();
    eprintln!("calibrating with meters: {names:?}, {} reps", plan.reps);
    match run(&mut meters, &plan, &out) {
        Ok((c, files)) => {
            for m in &c.meters {
                println!("{}: R² {:.3}, {} windows", m.meter, m.r2, m.runs.len());
                for (k, f) in &m.sim.fits {
                    let terms = &m.calibrated_terms[k];
                    let coefs: Vec<String> = ide_energy::sim::TERMS
                        .iter()
                        .zip(f.coef)
                        .filter(|(t, _)| terms.iter().any(|x| x == *t))
                        .map(|(t, c)| format!("{t}={c:.3e}"))
                        .collect();
                    println!("  {k}: {} (σ {:.3e} J)", coefs.join(" "), f.sigma);
                }
            }
            for (m, why) in &c.unusable {
                println!("{m}: not used ({why})");
            }
            for f in files {
                println!("wrote {}", f.display());
            }
            0
        }
        Err(e) => {
            eprintln!("rce-ide telemetry calibrate: {e}");
            1
        }
    }
}
