//! The lowering of the one emitter on the conformance suite (G4 of
//! one-emitter.md ONE-19): every test of conformance/tests whose
//! expectation is a run or a trace, built as the emitter adapter builds it
//! (the test's events as an appended example; `fn main` at the test's fuel
//! when it has no events). For each test the lowering covers, evaluating
//! the core prints the reference interpreter's trace byte for byte; a test
//! it does not cover is refused, never printed differently.
//!
//! This file is a test binary of its own because it sets the fuel limit,
//! which every evaluator of the process reads.
use std::path::{Path, PathBuf};

/// The number of run and trace tests the lowering covers with identical
/// traces; a change that covers more raises it.
const COVERED_AT_LEAST: usize = 1435;

fn walk(d: &Path, out: &mut Vec<PathBuf>) {
    let mut es: Vec<PathBuf> = std::fs::read_dir(d).unwrap().map(|e| e.unwrap().path()).collect();
    es.sort();
    for p in es {
        if p.is_dir() {
            walk(&p, out)
        } else if p.extension().is_some_and(|x| x == "rce") && !p.to_string_lossy().contains("_FIXTURE") {
            out.push(p)
        }
    }
}

#[test]
fn conformance_run_and_trace_tests_lower_to_the_reference_trace() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../conformance/tests");
    let mut files = vec![];
    walk(&root, &mut files);
    let (mut tests, mut same, mut refused) = (0, 0, 0);
    let mut diffs = vec![];
    for f in files {
        let raw = std::fs::read_to_string(&f).unwrap();
        let Some(i) = raw.rfind("\n#--- rce-test") else { continue };
        let (src, meta) = (&raw[..i + 1], &raw[i + 1..]);
        let get = |k: &str| meta.lines().find_map(|l| l.strip_prefix(&format!("# {k}: ")).map(|x| x.trim().to_string()));
        let expect = get("expect").unwrap_or_default();
        if expect != "run" && expect != "trace" {
            continue;
        }
        tests += 1;
        assert!(rce_spec_check::cli::diagnostics(src, &Default::default()).is_empty(), "{} is not admitted", f.display());
        let events = get("events");
        let full = match &events {
            Some(e) => format!("{src}example {e} -> true\n"),
            None => src.to_string(),
        };
        let m = rce_spec_check::program(&full).unwrap();
        let fuel = match (&events, get("fuel").and_then(|x| x.parse::<u64>().ok())) {
            (None, Some(n)) => n,
            _ => rce_spec_check::interp::FUEL,
        };
        rce_spec_check::interp::FUEL_LIMIT.store(fuel, std::sync::atomic::Ordering::Relaxed);
        let want = rce_spec_check::one::trace::reference(&m);
        match rce_spec_check::one::run_core_of(&m) {
            Ok(got) if got == want => same += 1,
            Ok(_) => diffs.push(f.display().to_string()),
            Err(e) => {
                assert!(e.starts_with("not covered by the lowering: "), "{}: {e}", f.display());
                refused += 1;
            }
        }
        rce_spec_check::interp::FUEL_LIMIT.store(rce_spec_check::interp::FUEL, std::sync::atomic::Ordering::Relaxed);
    }
    eprintln!("conformance run and trace tests: {tests}; identical traces: {same}; not yet covered: {refused}");
    assert!(diffs.is_empty(), "traces that differ: {diffs:?}");
    assert_eq!(tests, 1459);
    assert!(same >= COVERED_AT_LEAST, "{same} covered, {COVERED_AT_LEAST} expected");
}
