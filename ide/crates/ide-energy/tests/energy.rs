use ide_energy::anomaly::is_anomaly;
use ide_energy::attrib::{AttribError, Sample, attribute};
use ide_energy::meter::{Meter, MeterError, RaplMeter, wrap_delta};
use ide_energy::sim::{Features, LinearSim, Simulator};
use std::fs;

fn f(gear: &str, nodes: u64, passes: u64, tin: u64, tout: u64) -> Features {
    Features {
        form_nodes: nodes,
        checker_passes: passes,
        gear: gear.into(),
        tokens_in: tin,
        tokens_out: tout,
        ..Default::default()
    }
}

#[test]
fn calibration_recovers_known_coefficients() {
    // Synthetic ground truth for the fitting math only (not an energy claim):
    // E = 0.5 + 0.002*nodes + 0.1*passes + 1e-4*tin + 3e-4*tout
    let truth = |x: &Features| {
        0.5 + 0.002 * x.form_nodes as f64
            + 0.1 * x.checker_passes as f64
            + 1e-4 * x.tokens_in as f64
            + 3e-4 * x.tokens_out as f64
    };
    let mut samples = vec![];
    for i in 0..40u64 {
        let x = f(
            "local",
            10 + i * 7 % 50,
            1 + i % 4,
            100 + i * 37 % 900,
            20 + i * 13 % 300,
        );
        samples.push((x.clone(), truth(&x)));
    }
    let mut s = LinearSim::default();
    s.calibrate(&samples).unwrap();
    let fit = &s.fits["local/-"];
    for (got, want) in fit.coef.iter().zip([0.5, 0.002, 0.1, 0.0, 1e-4, 3e-4, 0.0]) {
        assert!((got - want).abs() < 1e-6, "{got} vs {want}");
    }
    assert!(fit.sigma < 1e-6);
    let p = s.predict(&f("local", 100, 2, 500, 50)).unwrap();
    assert!((p.joules - truth(&f("local", 100, 2, 500, 50))).abs() < 1e-6);
}

#[test]
fn uncalibrated_key_predicts_nothing() {
    let s = LinearSim::default();
    assert!(s.predict(&f("frontier", 1, 1, 1, 1)).is_none());
}

#[test]
fn bad_samples_are_not_recorded() {
    let mut s = LinearSim::default();
    assert!(s.calibrate(&[(f("g", 1, 1, 0, 0), f64::NAN)]).is_err());
    assert!(s.calibrate(&[(f("g", 1, 1, 0, 0), -1.0)]).is_err());
}

#[test]
fn anomaly_threshold() {
    assert!(!is_anomaly(0.2, 0.1, 3.0));
    assert!(is_anomaly(0.31, 0.1, 3.0));
    assert!(is_anomaly(-0.31, 0.1, 3.0));
    assert!(is_anomaly(f64::NAN, 0.1, 3.0));
}

#[test]
fn attribution_interpolates() {
    let s = [
        Sample {
            t_ns: 0,
            joules: 0.0,
        },
        Sample {
            t_ns: 100,
            joules: 10.0,
        },
        Sample {
            t_ns: 200,
            joules: 30.0,
        },
    ];
    assert_eq!(attribute(&s, 50, 150).unwrap(), 15.0); // 5 J at t=50, 20 J at t=150
    assert_eq!(attribute(&s, 0, 200).unwrap(), 30.0);
    assert_eq!(attribute(&s, 150, 250), Err(AttribError::Uncovered));
    assert_eq!(attribute(&s, 10, 10), Err(AttribError::Window(10, 10)));
}

#[test]
fn rapl_wraparound() {
    assert_eq!(wrap_delta(100, 150, 1000), 50);
    assert_eq!(wrap_delta(990, 9, 999), 19); // 990->999 is 9, 999->0 is 1, 0->9 is 9
}

fn zone(root: &std::path::Path, n: &str, name: &str, e: u64, max: u64) {
    let d = root.join(n);
    fs::create_dir_all(&d).unwrap();
    fs::write(d.join("name"), name).unwrap();
    fs::write(d.join("energy_uj"), e.to_string()).unwrap();
    fs::write(d.join("max_energy_range_uj"), max.to_string()).unwrap();
}

#[test]
fn rapl_reads_fake_powercap_tree() {
    let t = tempfile::tempdir().unwrap();
    let r = t.path();
    zone(r, "intel-rapl:0", "package-0", 999_000_000, 1_000_000_000);
    zone(r, "intel-rapl:1", "package-1", 5_000_000, 1_000_000_000);
    zone(r, "intel-rapl:0:0", "core", 1, 10); // sub-zone: ignored
    zone(r, "intel-rapl:2", "psys", 1, 10); // overlaps packages: ignored
    let mut m = RaplMeter::open_at(r).unwrap();
    assert_eq!(m.sample().unwrap().joules, 0.0);
    // package-0 wraps: 999_000_000 -> 1_000_000 over max 1e9
    fs::write(r.join("intel-rapl:0/energy_uj"), "1000000").unwrap();
    fs::write(r.join("intel-rapl:1/energy_uj"), "7000000").unwrap();
    let j = m.sample().unwrap().joules;
    assert!((j - (2.000001 + 2.0)).abs() < 1e-9, "{j}");
}

#[test]
fn rapl_missing_is_unsupported() {
    let t = tempfile::tempdir().unwrap();
    assert!(matches!(
        RaplMeter::open_at(t.path()),
        Err(MeterError::Unsupported(..))
    ));
    assert!(matches!(
        RaplMeter::open_at(&t.path().join("nope")),
        Err(MeterError::Unsupported(..))
    ));
}

#[test]
fn powermetrics_without_setup_never_returns_numbers() {
    use ide_energy::powermetrics::{Field, PowermetricsMeter};
    // On this box (no macOS, or no sudoers entry) it must refuse, not read 0.
    if let Err(e) = PowermetricsMeter::open(Field::Cpu, 200) {
        assert!(matches!(e, MeterError::Unsupported(..)), "{e}");
    }
}

/// On a machine with readable RAPL this reads it; otherwise it reports why.
#[test]
fn rapl_on_this_machine() {
    match RaplMeter::open() {
        Ok(mut m) => {
            let a = m.sample().unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            let b = m.sample().unwrap();
            assert!(b.joules >= a.joules);
            eprintln!(
                "rapl readable: {} J over {} ns",
                b.joules - a.joules,
                b.t_ns - a.t_ns
            );
        }
        Err(e) => eprintln!("rapl not readable here: {e}"),
    }
}

mod ioreport {
    use ide_energy::ioreport::{
        Origin, Rail, RailReading, Row, clpc_channels, joules_from_unit, rail_joules, rail_of,
        read_rail,
    };

    fn r(n: &str, v: i64, u: &str) -> Row {
        Row::new("Energy Model", n, v, u)
    }

    #[test]
    fn units() {
        assert_eq!(joules_from_unit(1500, "mJ"), Some(1.5));
        assert_eq!(joules_from_unit(2_000_000, "uJ"), Some(2.0));
        assert_eq!(joules_from_unit(3_000_000_000, "nJ"), Some(3.0));
        assert_eq!(joules_from_unit(5, "W"), None);
        assert_eq!(joules_from_unit(-1, "mJ"), None);
    }

    #[test]
    fn rails_do_not_double_count() {
        let rows = vec![
            r("ECPU0 CPU Energy", 100, "mJ"),
            r("PCPU0 CPU Energy", 300, "mJ"),
            r("CPU Energy", 400, "mJ"),
            r("GPU Energy", 50_000, "uJ"),
            r("GPU0", 10_000, "uJ"),
            r("GPU SRAM0", 99, "mJ"),
            r("ANE0", 1, "mJ"),
            r("DRAM0", 7, "weird"),
            r("Something", 1, "mJ"),
        ];
        assert_eq!(rail_joules(&rows, Rail::Cpu), Some(0.4));
        assert_eq!(rail_joules(&rows, Rail::Gpu), Some(0.05));
        assert_eq!(rail_joules(&rows, Rail::Ane), Some(0.001));
        assert_eq!(rail_joules(&rows, Rail::Dram), None); // unknown unit: no number
        assert_eq!(rail_of("Something"), None);
        let clusters = vec![
            r("ECPU0 CPU Energy", 100, "mJ"),
            r("PCPU0 CPU Energy", 300, "mJ"),
        ];
        assert_eq!(rail_joules(&clusters, Rail::Cpu), Some(0.4));
        let ultra = vec![
            r("DIE_0_CPU Energy", 100, "mJ"),
            r("DIE_1_CPU Energy", 200, "mJ"),
        ];
        assert!((rail_joules(&ultra, Rail::Cpu).unwrap() - 0.3).abs() < 1e-12);
    }

    /// The channel values read on the M5 Max (macOS 27.0.1, 2026-10-12):
    /// Energy Model CPU/ANE/DRAM at 0 over 4 s under load, GPU moving, CLPC
    /// CPU and GPU moving.
    fn m5max() -> Vec<Row> {
        vec![
            r("CPU Energy", 0, "mJ"),
            r("ANE0", 0, "mJ"),
            r("DRAM0", 0, "mJ"),
            r("GPU Energy", 39_224_054_504, "nJ"),
            Row::new("CLPC", "CPU Energy", 30_992_368_044, "nJ"),
            Row::new("CLPC", "GPU Energy", 39_201_512_326, "nJ"),
            Row::new("CLPC", "ANE", 0, "nJ"),
        ]
    }

    #[test]
    fn clpc_is_preferred_and_stuck_energy_model_is_not_zero() {
        let rows = m5max();
        assert_eq!(
            read_rail(&rows, Rail::Cpu),
            RailReading::Joules(30.992368044, Origin::Clpc)
        );
        assert_eq!(
            read_rail(&rows, Rail::Gpu),
            RailReading::Joules(39.201512326, Origin::Clpc)
        );
        // DRAM: only the stuck Energy Model channel: no number.
        assert_eq!(
            read_rail(&rows, Rail::Dram),
            RailReading::NoChange(Origin::EnergyModel)
        );
        assert_eq!(rail_joules(&rows, Rail::Dram), None);
        // ANE may be idle: 0 J is a real reading there.
        assert_eq!(rail_joules(&rows, Rail::Ane), Some(0.0));
        // Without CLPC (older macOS), the stuck CPU counter is not 0 J.
        let em: Vec<Row> = rows
            .into_iter()
            .filter(|x| x.group == "Energy Model")
            .collect();
        assert_eq!(
            read_rail(&em, Rail::Cpu),
            RailReading::NoChange(Origin::EnergyModel)
        );
        assert_eq!(rail_joules(&em, Rail::Cpu), None);
        assert_eq!(read_rail(&[], Rail::Cpu), RailReading::Missing);
    }

    #[test]
    fn clpc_ids() {
        // M5 Max driver seen on the 128GB Mac; ids as in macmon's own tests.
        let c = clpc_channels("com.apple.driver.AppleT6050CLPC", "27.0.1").unwrap();
        assert_eq!(c[0], (0x0000_0010_d3ab_60bb, "CPU Energy"));
        assert_eq!(c[1], (0x0000_0019_27cd_4ce5, "GPU Energy"));
        assert_eq!(c[2], (0x0000_0018_8c0f_1c59, "ANE"));
        let m2 = clpc_channels("com.apple.driver.AppleT8112CLPC", "27.0").unwrap();
        assert_eq!(
            m2.map(|x| x.0),
            [
                0x0000_0010_b543_5137,
                0x0000_0019_638d_9d52,
                0x0000_0018_ea08_9c36
            ]
        );
        for v in ["26.6", "28.0", "127.0", ""] {
            assert!(clpc_channels("com.apple.driver.AppleT6050CLPC", v).is_none());
        }
        assert!(clpc_channels("com.apple.driver.AppleUnknownCLPC", "27.0.1").is_none());
    }
}

mod power {
    use ide_energy::meter::{Meter, MeterError};
    use ide_energy::power::{Integrator, PolledMeter};
    use std::time::Duration;

    #[test]
    fn trapezoid() {
        let mut i = Integrator::default();
        i.push(0, 10.0).unwrap();
        i.push(1_000_000_000, 20.0).unwrap();
        i.push(3_000_000_000, 20.0).unwrap();
        assert!((i.joules() - 55.0).abs() < 1e-9);
        assert!(i.push(2_000_000_000, 1.0).is_err());
        assert!(i.push(4_000_000_000, -1.0).is_err());
        assert!(i.push(4_000_000_000, f64::NAN).is_err());
    }

    #[test]
    fn polled_meter_integrates_and_reports_errors() {
        let mut m = PolledMeter::start("test", Duration::from_millis(10), || Ok(5.0)).unwrap();
        std::thread::sleep(Duration::from_millis(120));
        let s = m.sample().unwrap();
        assert!(s.joules > 0.3 && s.joules < 1.0, "{}", s.joules);
        assert!(
            PolledMeter::start("test", Duration::from_millis(10), || {
                Err(MeterError::Unsupported("test", "x".into()))
            })
            .is_err()
        );
        let mut n = 0;
        let mut bad = PolledMeter::start("test", Duration::from_millis(5), move || {
            n += 1;
            if n > 2 {
                Err(MeterError::Read("test", "gone".into()))
            } else {
                Ok(1.0)
            }
        })
        .unwrap();
        std::thread::sleep(Duration::from_millis(60));
        assert!(
            bad.sample().is_err(),
            "a failed read is an error, not a frozen total"
        );
    }
}

mod smc {
    use ide_energy::smc::{decode, key};
    #[test]
    fn decodes_power_types_only() {
        assert_eq!(decode("flt ", &70.5f32.to_le_bytes()), Some(70.5));
        assert_eq!(decode("ui16", &[0x01, 0x00]), Some(256.0));
        assert_eq!(decode("fpe2", &[0, 0]), None);
        assert_eq!(decode("flt ", &[0, 0]), None);
        assert_eq!(decode("flt ", &f32::NAN.to_le_bytes()), None);
        assert_eq!(key("PSTR"), Some(0x5053_5452));
        assert_eq!(key("PST"), None);
    }
}

mod hwmon {
    use ide_energy::hwmon::{Kind, open, scan, watts};
    use std::fs;
    use std::time::Duration;

    fn dev(root: &std::path::Path, d: &str, name: &str, files: &[(&str, &str)]) {
        let p = root.join(d);
        fs::create_dir_all(&p).unwrap();
        fs::write(p.join("name"), format!("{name}\n")).unwrap();
        for (f, v) in files {
            fs::write(p.join(f), v).unwrap();
        }
    }

    #[test]
    fn scan_and_read_fake_tree() {
        let t = tempfile::tempdir().unwrap();
        let r = t.path();
        dev(
            r,
            "hwmon0",
            "amdgpu",
            &[("power1_average", "45000000\n"), ("power1_input", "1\n")],
        );
        dev(r, "hwmon1", "i915", &[("energy1_input", "1000000\n")]);
        dev(
            r,
            "hwmon2",
            "ina3221",
            &[
                ("in1_input", "5000"),
                ("curr1_input", "1000"),
                ("in1_label", "VDD_GPU_SOC"),
                ("in2_input", "5000"),
                ("curr2_input", "400"),
                ("in2_label", "VDD_CPU_CV"),
                ("in3_input", "5000"),
                ("curr3_input", "9999"),
                ("in3_label", "VDD_IN"),
            ],
        );
        dev(r, "hwmon3", "coretemp", &[("temp1_input", "40000")]);
        let ds = scan(r);
        assert_eq!(ds.len(), 3, "{ds:?}");
        assert_eq!((ds[0].domain(), ds[0].tag()), ("gpu", "hwmon"));
        assert!(
            matches!(&ds[0].kind, Kind::Power(p) if p.len() == 1 && p[0].ends_with("power1_average"))
        );
        assert!((watts(&ds[0].kind).unwrap() - 45.0).abs() < 1e-9);
        assert!(matches!(ds[1].kind, Kind::Energy(_)));
        assert_eq!(ds[2].tag(), "jetson");
        // VDD_IN (module input) is left out so rails are not double counted: 5 V x 1.4 A.
        assert!((watts(&ds[2].kind).unwrap() - 7.0).abs() < 1e-9);

        let mut m = open(&ds[1], Duration::from_millis(10)).unwrap();
        fs::write(r.join("hwmon1/energy1_input"), "3500000").unwrap();
        assert!((m.sample().unwrap().joules - 2.5).abs() < 1e-9);
        fs::write(r.join("hwmon1/energy1_input"), "10").unwrap();
        assert!(m.sample().is_err(), "a counter reset is an error");
        let mut p = open(&ds[0], Duration::from_millis(10)).unwrap();
        std::thread::sleep(Duration::from_millis(60));
        assert!(p.sample().unwrap().joules > 1.0);
        assert!(scan(&r.join("missing")).is_empty());
    }
}

mod supply {
    use ide_energy::supply::{batteries, discharge_watts, open};
    use std::fs;
    use std::time::Duration;

    #[test]
    fn battery_only_while_discharging() {
        let t = tempfile::tempdir().unwrap();
        let b = t.path().join("BAT0");
        fs::create_dir_all(&b).unwrap();
        fs::write(b.join("type"), "Battery\n").unwrap();
        fs::write(b.join("status"), "Discharging\n").unwrap();
        fs::write(b.join("current_now"), "-500000").unwrap();
        fs::write(b.join("voltage_now"), "4000000").unwrap();
        let ac = t.path().join("AC");
        fs::create_dir_all(&ac).unwrap();
        fs::write(ac.join("type"), "Mains").unwrap();
        assert_eq!(batteries(t.path()), vec![b.clone()]);
        assert!((discharge_watts(&b).unwrap() - 2.0).abs() < 1e-9);
        fs::write(b.join("power_now"), "3000000").unwrap();
        assert!((discharge_watts(&b).unwrap() - 3.0).abs() < 1e-9);
        fs::write(b.join("status"), "Charging").unwrap();
        assert!(discharge_watts(&b).is_err());
        assert!(open(&b, Duration::from_millis(10)).is_err());
    }
}

mod powermetrics {
    use ide_energy::powermetrics::{Field, parse_line};
    #[test]
    fn parses_lines() {
        assert_eq!(parse_line("CPU Power: 1234 mW", Field::Cpu), Some(1.234));
        assert_eq!(parse_line("GPU Power: 56 mW", Field::Gpu), Some(0.056));
        assert_eq!(
            parse_line("Combined Power (CPU + GPU + ANE): 2000 mW", Field::Combined),
            Some(2.0)
        );
        assert_eq!(parse_line("CPU Power: 1234 mW", Field::Gpu), None);
        assert_eq!(parse_line("CPU Power: lots", Field::Cpu), None);
        assert_eq!(parse_line("ANE Power: 0 mW", Field::Ane), Some(0.0));
    }
}

mod discover {
    use ide_energy::discover::{Status, TRACKED, probe_all, probe_rapl};
    use std::fs;

    #[test]
    fn probes_report_reasons_and_tracked_have_references() {
        let t = tempfile::tempdir().unwrap();
        let mut out = vec![];
        probe_rapl(t.path(), &mut out);
        if cfg!(target_os = "linux") {
            assert!(matches!(out[0].status, Status::Unsupported(_)));
        }
        let z = t.path().join("intel-rapl:0");
        fs::create_dir_all(&z).unwrap();
        fs::write(z.join("energy_uj"), "5").unwrap();
        fs::write(z.join("max_energy_range_uj"), "100").unwrap();
        let mut out = vec![];
        probe_rapl(t.path(), &mut out);
        assert_eq!(out[0].status, Status::Available);
        assert!(!out[0].resolution.is_empty() && out[0].accuracy != "unknown");
        for p in probe_all() {
            assert_ne!(p.accuracy, "unknown", "{p:?}");
        }
        assert!(TRACKED.iter().all(|t| t.reference.starts_with("https://")));
        assert!(TRACKED.iter().any(|t| t.tag == "windows-emi"));
    }
}

mod setup {
    use ide_energy::discover::{Probe, Status};
    use ide_energy::setup::{Io, Outcome, RAPL_RULE, apply, plan};

    struct Fake {
        answers: Vec<bool>,
        ran: Vec<String>,
        shown: String,
    }
    impl Io for Fake {
        fn show(&mut self, t: &str) {
            self.shown += t;
        }
        fn consent(&mut self, _: &str) -> bool {
            self.answers.remove(0)
        }
        fn run(&mut self, a: &[String]) -> Result<(), String> {
            self.ran.push(a.join(" "));
            Ok(())
        }
    }

    fn p(tag: &'static str, s: Status) -> Probe {
        Probe {
            tag,
            domain: "x".into(),
            detail: String::new(),
            status: s,
            resolution: "",
            accuracy: "",
        }
    }

    #[test]
    fn nothing_runs_without_consent() {
        let probes = [
            p("rapl", Status::NeedsSetup("denied".into())),
            p("powermetrics", Status::NeedsSetup("root".into())),
            p("smc", Status::Available),
        ];
        let steps = plan(&probes, "dave");
        assert_eq!(steps.len(), 2);
        assert!(plan(&[p("rapl", Status::Available)], "dave").is_empty());
        let mut f = Fake {
            answers: vec![false, false],
            ran: vec![],
            shown: String::new(),
        };
        let r = apply(&steps, &mut f);
        assert!(f.ran.is_empty());
        assert!(r.iter().all(|(_, o)| *o == Outcome::Declined));
        // The full file content and the security note are shown before asking.
        assert!(f.shown.contains(RAPL_RULE) && f.shown.contains("CVE-2020-8694"));
        assert!(
            f.shown
                .contains("dave ALL=(root) NOPASSWD: /usr/bin/powermetrics")
        );

        let mut f = Fake {
            answers: vec![false, true],
            ran: vec![],
            shown: String::new(),
        };
        let r = apply(&steps, &mut f);
        assert_eq!(r[0].1, Outcome::Declined);
        assert_eq!(r[1].1, Outcome::Applied);
        assert!(f.ran.iter().all(|c| c.starts_with("sudo ")));
        assert!(
            f.ran[0].ends_with("/etc/sudoers.d/rce-powermetrics") && f.ran[0].contains("-m 0440")
        );
        assert_eq!(f.ran[1], "sudo visudo -cf /etc/sudoers.d/rce-powermetrics");
    }
}
