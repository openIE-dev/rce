//! The execution bridge: run a real stdlib competence through the RCE runtime.
//!
//! A competence record names a stdlib operation (`math.sin`). To execute it on
//! an argument, the bridge generates a caller Form (`use math`, a state holding
//! the result, an event, a step calling the operation), runs it through the
//! reference checker (`rce run`), and parses the resulting state. The stdlib is
//! built into the checker, so this executes the *real* verified competence at
//! the interpreter floor — zero model calls.

use std::path::{Path, PathBuf};
use std::process::Command;

use rcevc_core::{Authority, Checker, Scope, State, Statement, Unit};

use crate::CompetenceRecord;

/// A bridge to the RCE reference runtime.
#[derive(Clone)]
pub struct RuntimeBridge {
    /// Path to the `rce` checker binary.
    pub rce_bin: PathBuf,
    /// Directory for generated caller programs.
    pub workdir: PathBuf,
}

impl RuntimeBridge {
    /// A bridge using the given checker binary and work directory.
    pub fn new(rce_bin: impl Into<PathBuf>, workdir: impl Into<PathBuf>) -> Self {
        RuntimeBridge { rce_bin: rce_bin.into(), workdir: workdir.into() }
    }

    /// Execute a unary `math` function (f64 -> f64) on `x`, returning the result.
    /// Generates a caller Form, runs it, parses the final state value.
    pub fn call_math_unary(&self, func: &str, x: f64) -> Result<f64, String> {
        let prog = format!(
            "module call_{func}\n\nuse math\n\nstate Out\n    v: f64 = 0.0\n\n\
             event Calc(x: f64)\n\nstep Out on Calc(x: a)\n    v = math.{func}(a)\n"
        );
        let path = self.workdir.join(format!("call_{func}.rce"));
        std::fs::create_dir_all(&self.workdir).map_err(|e| e.to_string())?;
        std::fs::write(&path, prog).map_err(|e| e.to_string())?;

        let out = Command::new(&self.rce_bin)
            .arg("run")
            .arg(&path)
            .arg(format!("Calc(x: {x})"))
            .output()
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() {
            return Err(format!("runtime refused: {}{}", stdout, String::from_utf8_lossy(&out.stderr)));
        }
        parse_state_value(&stdout).ok_or_else(|| format!("no value in output: {stdout}"))
    }

    /// Run an arbitrary caller Form through the reference runtime and return the
    /// final state's `v` field as text. This is the general G0 execution path:
    /// the caller `use`s the basis module and calls the owned operation.
    pub fn run_caller(&self, name: &str, prog: &str, event: &str) -> Result<String, String> {
        let path = self.workdir.join(format!("{name}.rce"));
        std::fs::create_dir_all(&self.workdir).map_err(|e| e.to_string())?;
        std::fs::write(&path, prog).map_err(|e| e.to_string())?;
        let out = Command::new(&self.rce_bin)
            .arg("run")
            .arg(&path)
            .arg(event)
            .output()
            .map_err(|e| e.to_string())?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() {
            return Err(format!("runtime refused: {}{}", stdout, String::from_utf8_lossy(&out.stderr)));
        }
        parse_v(&stdout).ok_or_else(|| format!("no value in output: {stdout}"))
    }

    /// Execute `hash.fnv1a32` (or any `(&[u8]) -> u32` hash) on bytes.
    pub fn call_hash_u8(&self, module: &str, func: &str, bytes: &[i64]) -> Result<String, String> {
        let lit: Vec<String> = bytes.iter().map(|b| b.to_string()).collect();
        let prog = format!(
            "module call_{func}\n\nuse {module}\n\nstate Out\n    v: u32 = 0\n\n\
             event Calc(xs: List[u8])\n\nstep Out on Calc(xs: a)\n    v = {module}.{func}(&a)\n"
        );
        self.run_caller(&format!("call_{func}"), &prog, &format!("Calc(xs: [{}])", lit.join(", ")))
    }

    /// Execute `col.search.binary_search` (`(&[i64], i64) -> i64`).
    pub fn call_binary_search(&self, xs: &[i64], target: i64) -> Result<String, String> {
        let lit: Vec<String> = xs.iter().map(|b| b.to_string()).collect();
        let prog = "module call_bs\n\nuse col.search\n\nstate Out\n    v: i64 = 0\n\n\
             event Calc(xs: List[i64], t: i64)\n\nstep Out on Calc(xs: a, t: b)\n    v = search.binary_search(&a, b)\n";
        self.run_caller("call_bs", prog, &format!("Calc(xs: [{}], t: {target})", lit.join(", ")))
    }

    /// Execute `text.glob.glob_match` (`(&text, &text) -> bool`).
    pub fn call_glob(&self, pat: &str, s: &str) -> Result<String, String> {
        let prog = "module call_glob\n\nuse text.glob\n\nstate Out\n    v: bool = false\n\n\
             event Calc(p: text, s: text)\n\nstep Out on Calc(p: a, s: b)\n    v = glob.glob_match(&a, &b)\n";
        self.run_caller("call_glob", prog, &format!("Calc(p: \"{pat}\", s: \"{s}\")"))
    }
}

/// Parse the final state's `v = <value>` field as text (any type).
pub fn parse_v(stdout: &str) -> Option<String> {
    for line in stdout.lines().rev() {
        if let Some(idx) = line.find("v = ") {
            let rest = &line[idx + 4..];
            let end = rest.find(" }").unwrap_or(rest.len());
            return Some(rest[..end].trim().to_string());
        }
    }
    None
}

/// Parse the final `Out { v = <value> }` line of `rce run` output.
fn parse_state_value(stdout: &str) -> Option<f64> {
    for line in stdout.lines().rev() {
        if let Some(idx) = line.find("v = ") {
            let rest = &line[idx + 4..];
            let end = rest.find(" }").unwrap_or(rest.len());
            return rest[..end].trim().parse::<f64>().ok();
        }
    }
    None
}

/// Verify a competence by replay: run the caller and check it matches a known
/// vector. This is the gate's evidence (Authority::Replay) for a real record.
pub fn verify_math_unary(bridge: &RuntimeBridge, func: &str, x: f64, expected: f64) -> bool {
    match bridge.call_math_unary(func, x) {
        Ok(got) => got == expected,
        Err(_) => false,
    }
}

/// The workdir helper for tests/examples.
pub fn ensure_dir(p: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(p)
}

/// A gate checker that verifies a real competence by replay through the RCE
/// reference runtime: it re-runs the record's stdlib operation on the probe
/// input and compares the declared spectest output. Authority is `Replay`;
/// the trusted base is the reference runtime + the stdlib built into it.
pub struct RuntimeChecker {
    /// The runtime used to re-execute the spectest.
    pub bridge: RuntimeBridge,
}

impl Checker for RuntimeChecker {
    fn statement(&self) -> Statement {
        Statement {
            name: "rce-runtime-spectest".into(),
            property_class: "the stdlib operation reproduces its spectest via the reference runtime".into(),
            linking_assumption: "re-executed by the RCE reference checker on the admitted post-state".into(),
            trusted_base: "rce reference runtime + stdlib built into the checker".into(),
            authority: Authority::Replay,
            scope: Scope::Unit,
            version: "1".into(),
        }
    }

    fn check_unit(&self, _s: &State, _slot: &str, u: &Unit) -> Result<(), String> {
        let r = CompetenceRecord::decode(&u.body).ok_or_else(|| "malformed record".to_string())?;
        let x: f64 = r.probe_input.parse().map_err(|_| "probe_input not f64".to_string())?;
        let expected: f64 = r.probe_output.parse().map_err(|_| "probe_output not f64".to_string())?;
        let got = self.bridge.call_math_unary(&r.handler_kind, x)?;
        if got != expected {
            return Err(format!(
                "spectest failed: {}({}) = {}, declared {}",
                r.handler_kind, x, got, expected
            ));
        }
        Ok(())
    }
}