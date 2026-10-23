//! Calibration against fake meters (no hardware): a shared "true" energy
//! counter that workloads advance by a known linear law.
use ide_energy::attrib::Sample;
use ide_energy::calib::{Plan, Workload, calibrate, load_sim, save};
use ide_energy::meter::{Meter, MeterError};
use ide_energy::sim::{Features, Simulator};
use std::cell::Cell;
use std::rc::Rc;
use std::time::Duration;

type Counter = Rc<Cell<f64>>;

/// Reads the shared counter, scaled (a second domain), or always stale.
struct Fake {
    tag: &'static str,
    c: Counter,
    scale: f64,
    stale: bool,
}
impl Meter for Fake {
    fn source(&self) -> &'static str {
        self.tag
    }
    fn sample(&mut self) -> Result<Sample, MeterError> {
        if self.stale {
            return Err(MeterError::Stale(self.tag, "did not move".into()));
        }
        Ok(Sample {
            t_ns: ide_energy::meter::now_ns(),
            joules: self.c.get() * self.scale,
        })
    }
}

/// Truth: 0.3 J per window + 0.05 J per pass + 0.002 J per form node, plus
/// a deterministic wobble (±1%).
struct Check {
    c: Counter,
    passes: u64,
    nodes: u64,
    n: Cell<u64>,
}
impl Workload for Check {
    fn name(&self) -> String {
        format!("check x{} ({} nodes)", self.passes, self.nodes)
    }
    fn run(&mut self) -> Result<Features, String> {
        let n = self.n.get();
        self.n.set(n + 1);
        let wob = 1.0 + 0.01 * ((n as f64) * 1.7).sin();
        let j = (0.3 + 0.05 * self.passes as f64 + 0.002 * (self.passes * self.nodes) as f64) * wob;
        self.c.set(self.c.get() + j);
        Ok(Features {
            gear: "checker".into(),
            checker_passes: self.passes,
            form_nodes: self.passes * self.nodes,
            ..Default::default()
        })
    }
}

fn setup() -> (Counter, Vec<Box<dyn Workload>>) {
    let c: Counter = Rc::default();
    let mut w: Vec<Box<dyn Workload>> = vec![];
    for (passes, nodes) in [
        (0, 0),
        (1, 40),
        (4, 40),
        (16, 40),
        (1, 300),
        (4, 300),
        (8, 120),
    ] {
        w.push(Box::new(Check {
            c: c.clone(),
            passes,
            nodes,
            n: Cell::new(0),
        }));
    }
    (c, w)
}

const FAST: Plan = Plan {
    reps: 4,
    settle: Duration::ZERO,
};

#[test]
fn recovers_the_law_per_meter_and_reports_honestly() {
    let (c, mut w) = setup();
    let mut meters: Vec<Box<dyn Meter>> = vec![
        Box::new(Fake {
            tag: "rapl",
            c: c.clone(),
            scale: 1.0,
            stale: false,
        }),
        Box::new(Fake {
            tag: "hwmon",
            c: c.clone(),
            scale: 0.5,
            stale: false,
        }),
        Box::new(Fake {
            tag: "smc",
            c: c.clone(),
            scale: 1.0,
            stale: true,
        }),
    ];
    let cal = calibrate(&mut meters, &mut w, &FAST).unwrap();
    assert_eq!(cal.meters.len(), 2);
    assert_eq!(cal.unusable.len(), 1);
    assert!(cal.unusable[0].1.starts_with("stale"), "{:?}", cal.unusable);
    let m = &cal.meters[0];
    assert_eq!(m.meter, "rapl");
    assert_eq!(m.runs.len(), 7 * 4);
    let fit = &m.sim.fits["checker/-"];
    let (i, p, n) = (fit.coef[0], fit.coef[2], fit.coef[1]);
    assert!((i - 0.3).abs() < 0.03, "intercept {i}");
    assert!((p - 0.05).abs() < 0.01, "per pass {p}");
    assert!((n - 0.002).abs() < 0.0003, "per node {n}");
    assert!(m.r2 > 0.99, "{}", m.r2);
    // Terms no workload varied are reported as not calibrated (held at 0).
    let t = &m.calibrated_terms["checker/-"];
    assert_eq!(t, &["intercept", "form_nodes", "checker_passes"]);
    assert_eq!(fit.coef[3], 0.0, "wasi_fuel held");
    // Second domain: half the energy, half the coefficients.
    let h = &cal.meters[1].sim.fits["checker/-"];
    assert!((h.coef[2] - 0.025).abs() < 0.005);
    // Other gears stay uncalibrated: no number.
    let g4 = Features {
        gear: "g4".into(),
        tokens_in: 100,
        ..Default::default()
    };
    assert!(m.sim.predict(&g4).is_none());
}

#[test]
fn saved_file_loads_as_the_simulator() {
    let (c, mut w) = setup();
    let mut meters: Vec<Box<dyn Meter>> = vec![Box::new(Fake {
        tag: "rapl",
        c: c.clone(),
        scale: 1.0,
        stale: false,
    })];
    let cal = calibrate(&mut meters, &mut w, &FAST).unwrap();
    let t = tempfile::tempdir().unwrap();
    let p = t.path().join("cfg/calibration.json");
    let files = save(&cal, &p).unwrap();
    assert_eq!(files.len(), 2);
    assert!(files[1].ends_with("calibration.rapl.json"));
    let sim = load_sim(&p).unwrap();
    assert_eq!(sim, cal.meters[0].sim);
    assert_eq!(load_sim(&files[1]).unwrap(), sim);
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
    assert_eq!(v["meter"], "rapl");
    assert_eq!(v["report"]["workloads"].as_array().unwrap().len(), 7);
    let f = Features {
        gear: "checker".into(),
        checker_passes: 2,
        form_nodes: 80,
        ..Default::default()
    };
    let j = sim.predict(&f).unwrap().joules;
    assert!((j - (0.3 + 0.1 + 0.16)).abs() < 0.03, "{j}");
}

#[test]
fn no_meter_or_only_broken_meters_is_an_error_not_zero() {
    let (c, mut w) = setup();
    assert!(calibrate(&mut [], &mut w, &FAST).is_err());
    let mut meters: Vec<Box<dyn Meter>> = vec![Box::new(Fake {
        tag: "smc",
        c,
        scale: 1.0,
        stale: true,
    })];
    let e = calibrate(&mut meters, &mut w, &FAST)
        .unwrap_err()
        .to_string();
    assert!(e.contains("no meter measured"), "{e}");
}
