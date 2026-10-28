//! rce-adapter/1 for the Lean core semantics (S6 of
//! spec/proposals/soundness.md). Entry (parse, check, admission) is the
//! reference front end's; the run is the Lean interpreter's: the module is
//! elaborated into the core (`core_elab`) and evaluated by the
//! Lean `CoreMain` (command `RCE_LEAN_CORE`, run in `RCE_LEAN_DIR`). Programs outside
//! the core subset answer `unsupported`.
use rce_spec_check::{cli, core_elab, interp, parse};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::Ordering::SeqCst;

struct Lean {
    _c: Child,
    i: ChildStdin,
    o: BufReader<ChildStdout>,
}

impl Lean {
    fn start() -> Result<Lean, String> {
        // RCE_LEAN_CORE: a command line (default: interpret CoreMain.lean,
        // since compiled Lean 4.12 binaries do not load on current macOS);
        // RCE_LEAN_DIR: the lake package it runs in (spec/lean)
        let cmd = std::env::var("RCE_LEAN_CORE").unwrap_or_else(|_| "lake env lean --run CoreMain.lean".into());
        let argv: Vec<&str> = cmd.split_whitespace().collect();
        let mut k = Command::new(argv[0]);
        k.args(&argv[1..]).stdin(Stdio::piped()).stdout(Stdio::piped());
        if let Ok(d) = std::env::var("RCE_LEAN_DIR") {
            k.current_dir(d);
        }
        let mut c = k.spawn().map_err(|e| format!("{cmd}: {e}"))?;
        let i = c.stdin.take().unwrap();
        let o = BufReader::new(c.stdout.take().unwrap());
        Ok(Lean { _c: c, i, o })
    }
    fn ask(&mut self, v: &Value) -> Result<Value, String> {
        writeln!(self.i, "{v}").map_err(|e| e.to_string())?;
        self.i.flush().map_err(|e| e.to_string())?;
        let mut l = String::new();
        self.o.read_line(&mut l).map_err(|e| e.to_string())?;
        serde_json::from_str(&l).map_err(|e| format!("lean: {e}: {l}"))
    }
}

fn ensure(lean: &mut Option<Lean>) -> Result<&mut Lean, Value> {
    if lean.is_none() {
        *lean = Some(Lean::start().map_err(|e| json!({"error": e}))?);
    }
    Ok(lean.as_mut().unwrap())
}

/// S4: an event run, δ by the Lean core (RCE/Core/Events.lean `deltaS`);
/// the lines are the reference's (cli.rs `run`), built from Lean's states.
fn run_events(lean: &mut Option<Lean>, src: &str, evs: &str) -> Value {
    interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
    let d = cli::diagnostics(src, &Default::default());
    if !d.is_empty() {
        return json!({"verdict": "fit", "diagnostics": d.iter().map(|x| json!({"line": x.0, "message": x.1})).collect::<Vec<_>>()});
    }
    let m = match parse::parse(src) {
        Ok(m) => m,
        Err(e) => return json!({"verdict": "fit", "diagnostics": [{"message": e}]}),
    };
    let (core, names, evn, errs) = match core_elab::elaborate_events(&m, src, evs, interp::FUEL) {
        Ok(c) => c,
        Err(why) => return json!({"unsupported": format!("outside the core subset: {why}")}),
    };
    let l = match ensure(lean) {
        Ok(l) => l,
        Err(e) => return e,
    };
    let r = match l.ask(&core) {
        Ok(r) => r,
        Err(e) => {
            *lean = None;
            return json!({"error": e});
        }
    };
    if let Some(e) = r.get("error") {
        return json!({"error": e});
    }
    let show = |st: &Value| -> String {
        names
            .iter()
            .zip(st.as_array().into_iter().flatten())
            .map(|((rn, fs), vals)| {
                let mut v: Vec<&str> = vals.as_array().into_iter().flatten().filter_map(|x| x.as_str()).collect();
                v.reverse();
                let f: Vec<String> = fs.iter().zip(v).map(|(n, x)| format!("{n} = {x}")).collect();
                format!("{rn} {{ {} }}", f.join(", "))
            })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut lines = vec![format!("0 (initial) {}", show(&r["init"]))];
    for (k, st) in r["steps"].as_array().into_iter().flatten().enumerate() {
        let en = &evn[k];
        if let Some(rf) = st.get("kept") {
            let why = match (rf.get("user").and_then(|c| c.as_u64()), rf.get("trap").and_then(|t| t.as_str())) {
                (Some(c), _) => errs.get(c as usize - 1).cloned().unwrap_or_default(),
                (_, Some(t)) => format!("trap: {t}"),
                _ => "?".into(),
            };
            let outcome = if let Some(t) = why.strip_prefix("trap: ") { format!("trap:{t}") } else { format!("kept:{why}") };
            return json!({"verdict": "ok", "outputs": lines, "outcome": outcome});
        }
        lines.push(format!("{} {} {}", k + 1, en, show(&st["state"])));
        for o in st["emits"].as_array().into_iter().flatten() {
            lines.push(format!("  {}", o.as_str().unwrap_or("")));
        }
    }
    json!({"verdict": "ok", "outputs": lines, "outcome": "done"})
}

fn run(lean: &mut Option<Lean>, src: &str, events: Option<&str>, fuel: Option<u64>) -> Value {
    if let Some(evs) = events {
        if fuel.is_some() {
            return json!({"unsupported": "fuel limits apply to fn main runs"});
        }
        return run_events(lean, src, evs);
    }
    interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
    let d = cli::diagnostics(src, &Default::default());
    if !d.is_empty() {
        return json!({"verdict": "fit", "diagnostics": d.iter().map(|x| json!({"line": x.0, "message": x.1})).collect::<Vec<_>>()});
    }
    let m = match parse::parse(src) {
        Ok(m) => m,
        Err(e) => return json!({"verdict": "fit", "diagnostics": [{"message": e}]}),
    };
    let core = match core_elab::elaborate(&m, fuel.unwrap_or(interp::FUEL)) {
        Ok(c) => c,
        Err(why) => return json!({"unsupported": format!("outside the core subset: {why}")}),
    };
    if lean.is_none() {
        match Lean::start() {
            Ok(l) => *lean = Some(l),
            Err(e) => return json!({"error": e}),
        }
    }
    match lean.as_mut().unwrap().ask(&core) {
        Ok(r) if r.get("error").is_some() => json!({"error": r["error"]}),
        Ok(r) if r["outcome"] == "stuck" => json!({"error": "lean core: stuck (soundness says unreachable)"}),
        Ok(r) => json!({"verdict": "ok", "outcome": r["outcome"], "outputs": r["outputs"], "ticks": r["ticks"]}),
        Err(e) => {
            *lean = None;
            json!({"error": e})
        }
    }
}

/// `check`: the core has no rules of section 7, so a program the reference
/// front end reports is `unsupported`. An accepted program in the core
/// subset is `ok` only when the Lean checker (`inf`/`checkProg`, proved sound
/// against `HasTy` in RCE/Core/Check.lean) shows the elaborated program well
/// typed; when it cannot, the answer is `unsupported`, never a verdict.
fn check(lean: &mut Option<Lean>, req: &Value, src: &str) -> Value {
    if req["files"].as_object().is_some_and(|o| o.len() > 1) {
        return json!({"unsupported": "the core models single-file programs"});
    }
    interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
    if !cli::diagnostics(src, &Default::default()).is_empty() {
        return json!({"unsupported": "rules of section 7 are not modelled by the core"});
    }
    let m = match parse::parse(src) {
        Ok(m) => m,
        Err(_) => return json!({"unsupported": "rules of section 7 are not modelled by the core"}),
    };
    let mut core = match core_elab::elaborate(&m, interp::FUEL) {
        Ok(c) => c,
        Err(why) => return json!({"unsupported": format!("outside the core subset: {why}")}),
    };
    core["check"] = json!(true);
    if lean.is_none() {
        match Lean::start() {
            Ok(l) => *lean = Some(l),
            Err(e) => return json!({"error": e}),
        }
    }
    match lean.as_mut().unwrap().ask(&core) {
        Ok(r) if r["typed"] == true => json!({"verdict": "ok", "ok": true}),
        Ok(r) => json!({"unsupported": format!("lean check: {}", r["why"].as_str().unwrap_or("not shown well typed"))}),
        Err(e) => {
            *lean = None;
            json!({"error": e})
        }
    }
}

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    let mut lean = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = writeln!(out, "{}", json!({"error": format!("bad request: {e}")}));
                continue;
            }
        };
        let src = req["source"].as_str().unwrap_or("");
        let mut resp = match req["op"].as_str().unwrap_or("") {
            "hello" => json!({"protocol": "rce-adapter/1", "implementation": "rce Lean core semantics",
                              "ops": ["check", "run"], "features": ["fuel"]}),
            "check" => check(&mut lean, &req, src),
            "run" => run(&mut lean, src, req["events"].as_str(), req["fuel"].as_u64()),
            op => json!({"unsupported": format!("op `{op}` is not modelled by the core")}),
        };
        resp["id"] = req["id"].clone();
        let _ = writeln!(out, "{resp}");
        let _ = out.flush();
    }
}
