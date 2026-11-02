//! `rce test` / `rce-ide test [energy-budget J] [measured J PROVENANCE.json |
//! reported J PROVENANCE.json | measure] [strict] FILE...`: runs `rce test`,
//! then gates on energy (CodeGreen lesson). `measure` reads the RAPL package
//! counters around the run and prints the reading as a D.11 receipt term
//! with basis `reported` and its provenance (`crate::reading`); with no
//! counter it prints "unmeasured: no counter available" and the gate has no
//! figure. `measured J PROVENANCE.json` (a wall or shunt meter) and
//! `reported J PROVENANCE.json` (another OS or on-chip counter) take a figure
//! read elsewhere, with its D.11 provenance from the file; a provenance that
//! is not valid by D.11 rule 1 is refused (status 2) and nothing runs, since
//! D.11 gives a figure without one no label: it is neither a reading nor an
//! estimate. With none of the three, the gate uses the static E_sim from
//! `cost`, labelled simulated; a run with no figure is `no measurement`,
//! which fails only under `strict`.
//!
//! `rce-ide telemetry export [perfetto] out OUT.json FILE...`: static
//! per-function E_sim as a Perfetto trace, flagged synthetic.
//!
//! Both take words, then operands (spec/proposals/positional-cli.md, PC-1).
use crate::cost::{analyze, e_sim, Table};
use crate::reading::{Basis, Env, Provenance, Sampling, Term};
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
    words: &[
        Word::set("energy-budget", &["J"]),
        Word::set("measured", &["J", "PROVENANCE.json"]),
        Word::set("reported", &["J", "PROVENANCE.json"]),
        Word::mode("measure"),
        Word::mode("strict"),
    ],
    operands: "FILE...",
    min: 1,
    max: MANY,
    help: "run every example and check, then gate on energy (measured, reported and measure are exclusive;\na typed J needs its D.11 provenance in PROVENANCE.json)",
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

/// A figure typed after `measured` or `reported`, with its provenance read
/// from the file named after it: a D.11 term, or why it is refused.
pub fn typed(basis: Basis, vals: &[String]) -> Result<Term, String> {
    let w = basis.word();
    let (j, file) = match vals {
        [j, file] => (j, file),
        _ => return Err(format!("test: `{w}` takes J PROVENANCE.json")),
    };
    let refuse = |why: String| format!("test: {w} {j} {file}: {why}; D.11 rule 1: a `{w}` figure carries a valid provenance, and a figure without one is not a reading");
    let jv = j.parse::<f64>().map_err(|_| refuse(format!("J `{j}` is not a number")))?;
    let text = std::fs::read_to_string(file).map_err(|e| refuse(format!("the provenance cannot be read ({e})")))?;
    let prov = Provenance::parse(&text).map_err(|e| refuse(format!("the provenance is not valid: {e}")))?;
    Term::new(jv, basis, prov).map_err(refuse)
}

/// The gate, with the counters read from `env` (`Env::system()` outside
/// tests; the tests pass a fake powercap tree).
fn gate(p: &Parsed, out: &mut dyn Write, env: &Env) -> i32 {
    let num = |w: &str| p.value(w).and_then(|v| v.parse::<f64>().ok());
    let (files, budget, strict) = (p.operands.clone(), num("energy-budget"), p.has("strict"));
    let measure = p.has("measure");
    let given: Vec<&str> = ["measured", "reported", "measure"].into_iter().filter(|w| p.has(w)).collect();
    if given.len() > 1 {
        eprintln!("test: {} are exclusive; give one", given.join(" and "));
        return 2;
    }
    // `measured J PROVENANCE.json`, `reported J PROVENANCE.json`: a figure
    // read elsewhere is a reading only with its D.11 provenance; without a
    // valid one it is refused before anything runs.
    let figure = match given.first().copied() {
        Some("measured") => Some(typed(Basis::Measured, p.all("measured")[0])),
        Some("reported") => Some(typed(Basis::Reported, p.all("reported")[0])),
        _ => None,
    };
    let figure = match figure.transpose() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
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
            (Some(r.term.j()), "reported: RAPL package counter, whole package")
        }
        Some(Err(u)) => {
            let _ = writeln!(out, "{}", u.line());
            (None, "unmeasured")
        }
        None => match &figure {
            Some(t) => {
                let _ = writeln!(out, "{}", t.line());
                (Some(t.j()), match t.basis() {
                    Basis::Measured => "measured: typed, with its provenance",
                    Basis::Reported => "reported: typed, with its provenance",
                })
            }
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
        let pv = provenance_file(&crate::reading::tests::prov().json());
        assert_eq!(test("rce", &["energy-budget".into(), "1".into(), "measured".into(), "2".into(), pv, p.clone()], &mut o), 1);
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
    fn provenance_file(text: &str) -> String {
        let d = tempfile::tempdir().unwrap().keep();
        let f = d.join("provenance.json");
        std::fs::write(&f, text).unwrap();
        f.to_string_lossy().into_owned()
    }
    #[test]
    fn typed_figure_needs_its_provenance() {
        let p = prog();
        let env = fake(&tempfile::tempdir().unwrap().path().join("no-powercap"));
        let good = crate::reading::tests::prov().json();
        let pv = provenance_file(&good);
        let receipt = |basis: &str| format!(r#"energy receipt: {{"amortization":{{"kind":"per_call"}},"basis":"{basis}","j":2.0,"provenance":{good}}}"#);
        // with a valid provenance: a reading, under the label given
        let (c, o) = run_at(&["energy-budget", "1", "measured", "2", &pv, &p], &env);
        assert_eq!(c, 1, "{o}");
        let lines: Vec<&str> = o.lines().filter(|l| l.starts_with("energy")).collect();
        assert_eq!(lines, [receipt("measured").as_str(), "energy OVER budget: 2.000e0 J > 1.000e0 J (measured: typed, with its provenance)"]);
        let (c, o) = run_at(&["energy-budget", "3", "reported", "2", &pv, &p], &env);
        assert_eq!(c, 0, "{o}");
        let lines: Vec<&str> = o.lines().filter(|l| l.starts_with("energy")).collect();
        assert_eq!(lines, [receipt("reported").as_str(), "energy ok: 2.000e0 J <= 3.000e0 J (reported: typed, with its provenance)"]);
        // without a valid one: refused, citing D.11 rule 1, and nothing runs
        let bad = provenance_file(r#"{"samples":4}"#);
        let missing = format!("{pv}.absent");
        for (prov, why) in [(&bad, "the provenance is not valid: unknown field `samples`"), (&missing, "the provenance cannot be read (")] {
            let w = vec!["test".to_string(), "energy-budget".into(), "1".into(), "measured".into(), "2".into(), prov.clone(), p.clone()];
            let (_, parsed) = rce_spec_check::args::read("rce", &[TEST], REMOVED, &w).unwrap();
            let e = typed(Basis::Measured, parsed.all("measured")[0]).unwrap_err();
            assert!(e.starts_with(&format!("test: measured 2 {prov}: ")) && e.contains(why), "{e}");
            assert!(e.ends_with("D.11 rule 1: a `measured` figure carries a valid provenance, and a figure without one is not a reading"), "{e}");
            let mut o = Vec::new();
            assert_eq!(gate(&parsed, &mut o, &env), 2);
            assert!(o.is_empty(), "{}", String::from_utf8_lossy(&o));
        }
        let e = typed(Basis::Reported, &["x".into(), pv.clone()]).unwrap_err();
        assert!(e.contains("J `x` is not a number") && e.contains("a `reported` figure"), "{e}");
        let e = typed(Basis::Measured, &["inf".into(), pv.clone()]).unwrap_err();
        assert!(e.contains("J inf is not a finite figure >= 0"), "{e}");
        // the old spelling `measured J FILE` has no provenance: the FILE is
        // read as one, and with no FILE left it is a usage error
        let mut o = Vec::new();
        assert_eq!(test("rce", &["measured".into(), "2".into(), p.clone()], &mut o), 2);
        assert!(o.is_empty());
        let mut o = Vec::new();
        assert_eq!(test("rce", &["measured".into(), "2".into(), p.clone(), p.clone()], &mut o), 2);
        assert!(o.is_empty());
        // one source of a figure at a time
        let (c, o) = run_at(&["measured", "2", &pv, "measure", &p], &env);
        assert_eq!((c, o.as_str()), (2, ""));
        let (c, _) = run_at(&["measured", "2", &pv, "reported", "2", &pv, &p], &env);
        assert_eq!(c, 2);
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
