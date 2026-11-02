//! `rce test` / `rce-ide test [energy-budget J] [measured J | measure] [strict] FILE...`:
//! runs `rce test`, then gates on energy (CodeGreen lesson). `measure`
//! reads the RAPL package counters around the run and prints the reading as
//! a D.11 receipt term with basis `reported` and its provenance
//! (`crate::reading`); with no counter it prints "unmeasured: no counter
//! available" and the gate has no figure. With neither `measure` nor
//! `measured`, the gate uses the static E_sim from `cost`, labelled
//! simulated; a run with no figure is `no measurement`, which fails only
//! under `strict`.
//!
//! `rce-ide telemetry export [perfetto] out OUT.json FILE...`: static
//! per-function E_sim as a Perfetto trace, flagged synthetic.
//!
//! Both take words, then operands (spec/proposals/positional-cli.md, PC-1).
use crate::cost::{analyze, e_sim, Table};
use crate::reading::{Env, Sampling};
use rce_spec_check::args::{Command, Fate, MANY, Parsed, Removed, Word};
use ide_energy::budget::{check, Verdict};
use ide_energy::perfetto::{to_trace, Span};
use std::io::Write;

pub fn sim_joules(files: &[String]) -> Result<std::collections::BTreeMap<String, f64>, String> {
    let t = Table::default();
    let mut out = std::collections::BTreeMap::new();
    for f in files {
        let src = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
        let m = rce_spec_check::parse::parse(&src).map_err(|e| format!("{f}: {e}"))?;
        for (k, j) in e_sim(&analyze(&m, &t), &t) {
            out.insert(format!("{f}::{k}"), j);
        }
    }
    Ok(out)
}

/// `test` with the energy gate (PC-8: the one table its usage line and its
/// reader share).
pub const TEST: Command = Command {
    path: &["test"],
    words: &[Word::set("energy-budget", &["J"]), Word::set("measured", &["J"]), Word::mode("measure"), Word::mode("strict")],
    operands: "FILE...",
    min: 1,
    max: MANY,
    help: "run every example and check, then gate on energy (measured and measure are exclusive)",
};

/// `telemetry export`: static E_sim as a Perfetto trace.
pub const EXPORT: Command = Command {
    path: &["telemetry", "export"],
    words: &[Word::mode("perfetto"), Word::set("out", &["OUT.json"])],
    operands: "FILE...",
    min: 1,
    max: MANY,
    help: "static per-function E_sim as a Perfetto trace, flagged synthetic (out is required)",
};

/// The flags `test` and `telemetry export` had (positional-cli.md 1.4).
pub const REMOVED: &[Removed] = &[
    Removed { flag: "--energy-budget", paths: &["test"], values: 1, fate: Fate::Word },
    Removed { flag: "--measured", paths: &["test"], values: 1, fate: Fate::Word },
    Removed { flag: "--measure", paths: &["test"], values: 0, fate: Fate::Word },
    Removed { flag: "--strict", paths: &["test"], values: 0, fate: Fate::Word },
    Removed { flag: "--perfetto", paths: &["telemetry export"], values: 0, fate: Fate::Word },
    Removed { flag: "-o", paths: &["telemetry export"], values: 1, fate: Fate::Named("out") },
];

/// `PROGRAM test ...`, given the arguments after `test`: read the words,
/// then run the gate. A usage error prints its line and is status 2.
pub fn test(program: &str, args: &[String], out: &mut dyn Write) -> i32 {
    let mut a = vec!["test".to_string()];
    a.extend(args.iter().cloned());
    match rce_spec_check::args::read(program, &[TEST], REMOVED, &a) {
        Ok((_, p)) => test_words(&p, out),
        Err(e) => {
            eprintln!("{e}");
            2
        }
    }
}

/// The energy gate on the words and FILEs of `test`.
pub fn test_words(p: &Parsed, out: &mut dyn Write) -> i32 {
    gate(p, out, &Env::system())
}

/// The gate, with the counters read from `env` (`Env::system()` outside
/// tests; the tests pass a fake powercap tree).
fn gate(p: &Parsed, out: &mut dyn Write, env: &Env) -> i32 {
    let num = |w: &str| p.value(w).and_then(|v| v.parse::<f64>().ok());
    let (files, budget, measured, strict) = (p.operands.clone(), num("energy-budget"), num("measured"), p.has("strict"));
    let measure = p.has("measure");
    if measure && p.has("measured") {
        eprintln!("test: measured J and measure are exclusive");
        return 2;
    }
    let mut a = vec!["test".to_string()];
    a.extend(files.iter().cloned());
    // `measure`: read the machine's RAPL package counters around the run
    // and print the reading as a D.11 receipt term (basis `reported`). No
    // readable counter means "unmeasured", never an estimate.
    let sampling = if measure { Some(Sampling::start(env)) } else { None };
    let code = rce_spec_check::cli::main(&a, &mut std::io::empty(), out, &mut std::io::stderr());
    let (joules, kind) = match sampling.map(|s| s.and_then(Sampling::finish)) {
        Some(Ok(r)) => {
            for l in r.lines() {
                let _ = writeln!(out, "{l}");
            }
            (Some(r.j), "reported: RAPL package counter, whole package")
        }
        Some(Err(u)) => {
            let _ = writeln!(out, "{}", u.line());
            (None, "unmeasured")
        }
        None => match measured {
            Some(m) => (Some(m), "measured"),
            None => (sim_joules(&files).ok().map(|m| m.values().sum()), "simulated, synthetic prices"),
        },
    };
    let Some(b) = budget else { return code };
    let v = check(joules, b);
    let _ = match &v {
        Verdict::Pass { joules, budget } => writeln!(out, "energy ok: {joules:.3e} J <= {budget:.3e} J ({kind})"),
        Verdict::Over { joules, budget } => writeln!(out, "energy OVER budget: {joules:.3e} J > {budget:.3e} J ({kind})"),
        Verdict::NoMeasurement => writeln!(out, "energy: no measurement"),
    };
    if v.failed(strict) { code.max(1) } else { code }
}

/// `telemetry export` on its words and FILEs.
pub fn export_perfetto(p: &Parsed) -> Result<String, String> {
    let files = p.operands.clone();
    let o = p.value("out").map(String::from).ok_or("out OUT.json is required")?;
    let mut t = 0u64;
    let spans: Vec<Span> = sim_joules(&files)?.into_iter().map(|(name, j)| {
        // Duration is a stand-in proportional to E_sim at 1 W, so the
        // track is readable; it is not a measured time.
        let d = ((j * 1e9) as u64).max(1);
        let s = Span { name, track: "E_sim (static, synthetic)".into(), start_ns: t, end_ns: t + d, joules: j };
        t += d;
        s
    }).collect();
    let v = to_trace(&spans, &[], true);
    std::fs::write(&o, serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn prog() -> String {
        let d = tempfile::tempdir().unwrap().keep();
        let p = d.join("a.rce");
        std::fs::write(&p, "fn f(n: i64) -> i64\n    return n + 1\n").unwrap();
        p.to_string_lossy().into_owned()
    }
    #[test]
    fn budget_gate() {
        let p = prog();
        let mut o = Vec::new();
        assert_eq!(test("rce", &["energy-budget".into(), "1".into(), p.clone()], &mut o), 0);
        assert!(String::from_utf8_lossy(&o).contains("energy ok"));
        let mut o = Vec::new();
        assert_eq!(test("rce", &["energy-budget".into(), "1e-15".into(), p.clone()], &mut o), 1);
        let mut o = Vec::new();
        assert_eq!(test("rce", &["energy-budget".into(), "1".into(), "measured".into(), "2".into(), p.clone()], &mut o), 1);
        // a removed flag is a usage error and runs nothing (positional-cli.md, decision 2)
        let mut o = Vec::new();
        assert_eq!(test("rce", &[p, "--energy-budget".into(), "1".into()], &mut o), 2);
        assert!(o.is_empty());
    }
    fn run_at(words: &[&str], env: &Env) -> (i32, String) {
        let mut a = vec!["test".to_string()];
        a.extend(words.iter().map(|w| w.to_string()));
        let (_, w) = rce_spec_check::args::read("rce", &[TEST], REMOVED, &a).unwrap();
        let mut o = Vec::new();
        let c = gate(&w, &mut o, env);
        (c, String::from_utf8(o).unwrap())
    }
    // test-only: a fake powercap root and a fixed machine name
    fn fake(root: &std::path::Path) -> Env {
        Env { powercap: root.to_path_buf(), machine: Some("test-host; cpu Test CPU; kernel Linux 0.0".into()), period: std::time::Duration::from_secs(3600) }
    }
    #[test]
    fn measure_without_counter_is_unmeasured() {
        let p = prog();
        let t = tempfile::tempdir().unwrap();
        let env = fake(&t.path().join("no-powercap"));
        let (c, o) = run_at(&["measure", &p], &env);
        assert_eq!(c, 0, "{o}");
        assert!(o.contains("energy: unmeasured: no counter available ("), "{o}");
        assert!(!o.contains("energy receipt") && !o.contains("simulated") && !o.contains("measured)"), "{o}");
        // with a budget: no figure, never the estimate; fails only under strict
        let (c, o) = run_at(&["energy-budget", "1", "measure", &p], &env);
        assert_eq!(c, 0, "{o}");
        assert!(o.contains("energy: no measurement") && !o.contains("energy ok"), "{o}");
        let (c, _) = run_at(&["energy-budget", "1", "measure", "strict", &p], &env);
        assert_eq!(c, 1);
        // this machine: a reading only where a counter exists
        let (_, o) = run_at(&["measure", &p], &Env::system());
        assert!(o.contains("energy receipt: {") != o.contains("energy: unmeasured:"), "{o}");
    }
    #[test]
    fn measure_with_still_counter_is_unmeasured() {
        let p = prog();
        let t = tempfile::tempdir().unwrap();
        crate::reading::tests::zone(t.path(), "intel-rapl:0", "package-0", 5, 1000);
        let (_, o) = run_at(&["energy-budget", "1", "measure", &p], &fake(t.path()));
        assert!(o.contains("energy: unmeasured: the counter did not move") && o.contains("energy: no measurement"), "{o}");
    }
    #[test]
    fn perfetto_file() {
        let p = prog();
        let out = format!("{p}.json");
        let a: Vec<String> = vec!["telemetry".into(), "export".into(), "perfetto".into(), "out".into(), out.clone(), p];
        let (_, w) = rce_spec_check::args::read("rce-ide", &[EXPORT], REMOVED, &a).unwrap();
        export_perfetto(&w).unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(out).unwrap()).unwrap();
        assert_eq!(v["metadata"]["synthetic_energy"], true);
    }
}
