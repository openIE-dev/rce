//! rce-conform: the reference implementation behind the rce-conformance
//! adapter protocol (rce-adapter/1). One JSON request per stdin line, one
//! JSON response per stdout line. Requests:
//!   {"op":"hello"}
//!   {"op":"check","files":{"main.rce":SRC,...}}
//!   {"op":"run","source":SRC,"events":"Inc, Dec"|null,"fuel":N|null,"schedule_seed":N|null}
//!   {"op":"trace","source":SRC,"events":"Inc, Dec"}
//! A trace answer is the whole trace: `head`, the hash of the last entry of
//! D as `rce trace` writes it, and `lines`, the block an emitted program run
//! with `--trace --fuel-trace` prints for the same events (`emit::trace_example`):
//! every entry of D, the state after each event, the outputs, the reason an
//! event was kept, and the fuel used.
//! Every response carries the request's "id". Diagnostics carry the rule
//! code that 7. requires every diagnostic to name.
use rce_spec_check::ast::Item;
use rce_spec_check::{cli, conc, emit, interp, parse};
use serde_json::{Value, json};
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering::SeqCst;

fn code_of(msg: &str) -> Option<String> {
    let b = msg.as_bytes();
    for i in 0..b.len() {
        if b[i] == b'W' && b.get(i + 1) == Some(&b'-') && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
            let mut j = i + 2;
            while j < b.len() && (b[j].is_ascii_uppercase() || b[j] == b'-' || b[j].is_ascii_digit()) {
                j += 1;
            }
            let s = msg[i..j].trim_end_matches('-');
            if s.rsplit('-')
                .next()
                .is_some_and(|d| !d.is_empty() && d.bytes().all(|c| c.is_ascii_digit()))
            {
                return Some(s.to_string());
            }
        }
    }
    None
}

fn diags(errs: Vec<String>) -> Value {
    Value::Array(
        errs.iter()
            .map(|e| json!({"code": code_of(e), "line": cli::line_of(e), "message": e}))
            .collect(),
    )
}

fn check(files: &[(String, String)]) -> Value {
    let mut all = vec![];
    for (_, v) in rce_spec_check::check_program(files, false) {
        if let Err(e) = v {
            all.extend(e.split("; ").map(String::from));
        }
    }
    if all.is_empty() {
        json!({"verdict": "accept", "diagnostics": []})
    } else {
        json!({"verdict": "change", "diagnostics": diags(all)})
    }
}

fn outcome_of(lines: &[String], kept: bool) -> String {
    if !kept {
        return "done".into();
    }
    let last = lines.last().cloned().unwrap_or_default();
    if last == "deadlock" {
        return "deadlock".into();
    }
    let r = last.split_once(" kept: ").or_else(|| last.strip_prefix("stopped: ").map(|x| ("", x))).map(|x| x.1).unwrap_or(&last).trim().to_string();
    trap_or_kept(&r)
}

/// One spelling for an outcome: `trap:KIND` for a trap however the evaluator
/// words it (`trap: K` from fn main, `trap(K)` from δ), else `kept:E`.
fn trap_or_kept(r: &str) -> String {
    if let Some(t) = r.strip_prefix("trap: ") {
        return format!("trap:{t}");
    }
    if let Some(t) = r.strip_prefix("trap(").and_then(|t| t.strip_suffix(')')) {
        return format!("trap:{t}");
    }
    format!("kept:{r}")
}

fn run(src: &str, events: Option<&str>, fuel: Option<u64>, seed: Option<u64>) -> Value {
    // Admission runs the module's examples at the default fuel; the limit
    // of the request applies to the run only (fuel is per run, 6.7).
    interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
    let d = cli::diagnostics(src, &Default::default());
    if !d.is_empty() {
        return json!({"verdict": "change", "diagnostics": diags(d.into_iter().map(|x| x.1).collect())});
    }
    if events.is_none() {
        let m = match parse::parse(src) {
            Ok(m) => m,
            Err(e) => return json!({"verdict":"change","diagnostics":diags(vec![e])}),
        };
        let mut cfg = conc::RunCfg::default();
        if let Some(s) = seed {
            cfg.par_mode = conc::ParMode::Permute(s);
        }
        interp::FUEL_LIMIT.store(fuel.unwrap_or(interp::FUEL), SeqCst);
        let r = conc::run_main(&m, &cfg);
        interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
        return match r {
            Ok(r) => {
                let outputs: Vec<String> = r.outputs.iter().map(|o| format!("emit {}", o.show())).collect();
                let outcome = match r.outcome {
                    conc::Outcome::Done => "done".to_string(),
                    conc::Outcome::Deadlock => "deadlock".to_string(),
                    conc::Outcome::Stopped(e) => trap_or_kept(&e),
                };
                json!({"verdict": "accept", "outputs": outputs, "outcome": outcome})
            }
            Err(e) if conc::is_no_task_threads(&e) => json!({"verdict": "accept", "error": e}),
            Err(e) => json!({"verdict": "accept", "error": format!("{e}: give events, or declare `fn main()`")}),
        };
    }
    if fuel.is_some() {
        return json!({"verdict": "accept", "error": "fuel limits apply to fn main runs"});
    }
    match cli::run(src, events) {
        Ok(r) => {
            // the outcome of a run is that of its last step: an earlier kept
            // step is a line of the outputs, and the run went on (6.9)
            let outcome = outcome_of(&r.lines, r.last_kept);
            let mut lines = r.lines.clone();
            if r.last_kept {
                lines.pop();
            }
            json!({"verdict": "accept", "outputs": lines, "outcome": outcome})
        }
        Err(e) => json!({"verdict": "accept", "error": e}),
    }
}

fn trace(src: &str, events: Option<&str>) -> Value {
    interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
    let head = match cli::trace("t.rce", src, events) {
        Ok(t) => {
            let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
            // the head of the last example's (or the events') log; the
            // trace of `fn main` (source "main") is not an example's
            v["traces"]
                .as_array()
                .and_then(|t| t.iter().rfind(|t| t["source"] != "main"))
                .and_then(|t| t["steps"].as_array())
                .and_then(|s| s.last())
                .map(|s| s["hash"].clone())
                .unwrap_or(v["genesis"].clone())
        }
        Err(e) => return json!({"verdict": "accept", "error": e}),
    };
    match whole_trace(src, events) {
        Ok(lines) => json!({"verdict": "accept", "trace": {"head": head, "lines": lines}}),
        Err(e) => json!({"verdict": "accept", "error": e}),
    }
}

/// The lines of the traced example: the events given, appended to the
/// module as one more example (as the emit adapter appends them), or with
/// no events the module's last example.
fn whole_trace(src: &str, events: Option<&str>) -> Result<Vec<String>, String> {
    let (text, line) = match events {
        Some(evs) => {
            let base = if src.ends_with('\n') { src.to_string() } else { format!("{src}\n") };
            let line = base.matches('\n').count() + 1;
            (format!("{base}example {evs} -> true\n"), Some(line))
        }
        None => (src.to_string(), None),
    };
    let m = rce_spec_check::program(&text)?;
    let mut last = None;
    for i in &m.items {
        match i {
            Item::Example(x) if line.is_none_or(|l| x.line == l) => last = Some(x),
            Item::Intent(_, xs, _) if line.is_none() => last = xs.last().or(last),
            _ => {}
        }
    }
    let x = last.ok_or("the module has no example to trace")?;
    Ok(emit::trace_example(&m, x))
}

/// `host`: run the steps on the host runtime of Appendix E over a fresh log
/// in a temporary directory, then replay that log (HC-8) and reopen it
/// (HC-17), and report all three state hashes, the entry kinds of D and
/// what each window admitted, acked, deduplicated and held.
fn host(src: &str, steps: &Value, tear: Option<u64>) -> Value {
    use rce_spec_check::ast::{Arg, Expr};
    use rce_spec_check::host::{Entry, Host, Sourced, replay};
    interp::FUEL_LIMIT.store(interp::FUEL, SeqCst);
    let d = cli::diagnostics(src, &Default::default());
    if !d.is_empty() {
        return json!({"verdict": "change", "diagnostics": diags(d.into_iter().map(|x| x.1).collect())});
    }
    let m = match parse::parse(src) {
        Ok(m) => m,
        Err(e) => return json!({"verdict":"change","diagnostics":diags(vec![e])}),
    };
    let dir = std::env::temp_dir().join(format!("rce-conform-host-{}-{}", std::process::id(), rand_suffix()));
    let _ = std::fs::create_dir_all(&dir);
    let log = dir.join("D");
    let mut admitted = vec![];
    let mut windows = vec![];
    let res: Result<String, String> = (|| {
        let mut h = Host::open_file(&m, &log)?;
        for st in steps.as_array().into_iter().flatten() {
            if let Some(b) = st["batch"].as_str() {
                for a in h.admit(src, b)? {
                    admitted.push(json!([
                        a.seq,
                        a.event,
                        match &a.outcome {
                            Ok(o) => json!({"ok": o}),
                            Err(e) => json!({"kept": e}),
                        }
                    ]));
                }
            } else if let Some(o) = st.get("offer") {
                let args: Vec<Arg> = o["args"]
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|(k, v)| Arg {
                        label: Some(k.clone()),
                        e: match v.as_i64() {
                            Some(i) => Expr::Int(i as i128, 0),
                            None => Expr::Str(v.as_str().unwrap_or("").into(), 0),
                        },
                    })
                    .collect();
                let x = Sourced::new(
                    o["source"].as_str().unwrap_or(""),
                    o["seq"].as_u64().unwrap_or(0),
                    o["tick"].as_u64().unwrap_or(0),
                    o["event"].as_str().unwrap_or(""),
                    args,
                );
                if let Some(w) = h.offer(x)? {
                    windows.push(win(&w));
                    for a in w.admitted {
                        admitted.push(json!([
                            a.seq,
                            a.event,
                            match &a.outcome {
                                Ok(o) => json!({"ok": o}),
                                Err(e) => json!({"kept": e}),
                            }
                        ]));
                    }
                }
            } else if let Some(u) = st["close"].as_u64() {
                let w = h.close_window(u)?;
                windows.push(win(&w));
                for a in w.admitted {
                    admitted.push(json!([
                        a.seq,
                        a.event,
                        match &a.outcome {
                            Ok(o) => json!({"ok": o}),
                            Err(e) => json!({"kept": e}),
                        }
                    ]));
                }
            }
        }
        Ok(h.state_hash().to_string())
    })();
    let out = match res {
        Err(e) => json!({"verdict": "accept", "error": e}),
        Ok(live) => {
            let text = std::fs::read_to_string(&log).unwrap_or_default();
            let kinds: Vec<String> = text.lines().filter_map(Entry::parse_line).map(|e| e.kind()).collect();
            let rep = replay(&m, &text).map(|(_, h, _)| h);
            let reopen = Host::open_file(&m, &log).map(|h| h.state_hash().to_string());
            // T38: a torn tail (the last `tear` bytes lost) must replay to the
            // state of the verified prefix, never fail or invent an entry
            let torn = match tear {
                Some(n) if (n as usize) < text.len() => {
                    let t = &text[..text.len() - n as usize];
                    let rep = match replay(&m, t) {
                        Ok((k, h, _)) => json!({"entries": k, "state_hash": h}),
                        Err(e) => json!({"error": e}),
                    };
                    // reopening a torn D keeps its verified prefix (T38, E.3)
                    let tl = dir.join("D.torn");
                    let _ = std::fs::write(&tl, t);
                    let reo = Host::open_file(&m, &tl)
                        .map(|h| h.state_hash().to_string())
                        .unwrap_or_else(|e| format!("error: {e}"));
                    json!({"replay": rep, "reopen_state_hash": reo})
                }
                _ => Value::Null,
            };
            json!({"verdict": "accept", "host": {"state_hash": live, "torn": torn, "replay_state_hash": rep.unwrap_or_else(|e| format!("error: {e}")),
                   "reopen_state_hash": reopen.unwrap_or_else(|e| format!("error: {e}")), "entries": kinds,
                   "admitted": admitted, "windows": windows}})
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    out
}

/// `replicate`: drive the replicated log of E.2b (HC-18 to HC-20) on an
/// in-memory cluster of `nodes` replicas with a scripted network, and report
/// each replica's committed entries, term and commit index.
fn replicate(nodes: usize, script: &Value) -> Value {
    use rce_spec_check::raft::Cluster;
    let mut c = Cluster::new(nodes.max(1));
    let mut leader: Option<usize> = None;
    let mut old: Option<usize> = None;
    let mut events = vec![];
    for st in script.as_array().into_iter().flatten() {
        if let Some(n) = st["elect"].as_u64() {
            leader = c.elect(n as usize);
            events.push(json!({"leader": leader, "term": leader.map(|l| c.node(l).term())}));
        } else if let Some(xs) = st["propose"].as_array() {
            let Some(l) = c.leader().or(leader) else {
                events.push(json!({"error": "no leader"}));
                continue;
            };
            for x in xs {
                c.node_mut(l).propose(x.as_str().unwrap_or(""));
            }
            if let Some(p) = c.peers[l].as_mut() {
                p.flush();
            }
        } else if let Some(xs) = st["propose_old"].as_array() {
            let Some(o) = old else { continue };
            for x in xs {
                c.node_mut(o).propose(x.as_str().unwrap_or(""));
            }
            if let Some(p) = c.peers[o].as_mut() {
                p.flush();
            }
        } else if let Some(n) = st["run"].as_u64() {
            c.run(n as usize);
        } else if let Some(k) = st["isolate_leader_with"].as_u64() {
            let Some(l) = c.leader().or(leader) else { continue };
            old = Some(l);
            let minority: Vec<usize> = (0..k as usize).map(|i| (l + i) % nodes).collect();
            let majority: Vec<usize> = (0..nodes).filter(|i| !minority.contains(i)).collect();
            c.sw.borrow_mut().partition(&minority, &majority);
            events.push(json!({"isolated": minority, "old_commit": c.node(l).commit, "old_term": c.node(l).term()}));
        } else if st["heal"].as_bool() == Some(true) {
            c.sw.borrow_mut().heal();
        } else if let Some(i) = st["crash"].as_u64() {
            c.crash(i as usize);
        } else if let Some(i) = st["restart"].as_u64() {
            c.restart(i as usize);
        } else if st["snapshot"].as_bool() == Some(true) {
            events.push(json!({"leader": c.leader(), "term": c.leader().map(|l| c.node(l).term()),
                               "old_commit": old.map(|o| c.node(o).commit), "old_term": old.map(|o| c.node(o).term())}));
        }
    }
    let replicas: Vec<Value> = (0..nodes)
        .map(|i| match c.peers[i].as_ref() {
            Some(_) => json!({"committed": c.node(i).committed(), "term": c.node(i).term(), "commit": c.node(i).commit}),
            None => Value::Null,
        })
        .collect();
    json!({"verdict": "accept", "replicate": {"replicas": replicas, "events": events}})
}

fn win(w: &rce_spec_check::host::WindowAck) -> Value {
    json!({"window": w.window, "acks": w.acks, "dups": w.dups, "held": w.held,
           "rejected": w.rejected.iter().map(|(s, q, _)| json!([s, q])).collect::<Vec<_>>()})
}

fn rand_suffix() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
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
        let src = req["source"].as_str().unwrap_or("").to_string();
        let events = req["events"].as_str();
        let mut resp = match req["op"].as_str().unwrap_or("") {
            "hello" => json!({"protocol": "rce-adapter/1", "implementation": "rce reference interpreter",
                              "commit": rce_spec_check::GIT_COMMIT, "ops": ["check", "run", "trace", "host", "replicate"],
                              "features": ["fuel", "schedule_seed", "trace"]}),
            "check" => {
                let files: Vec<(String, String)> = match req["files"].as_object() {
                    Some(o) => o.iter().map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string())).collect(),
                    None => vec![("main.rce".into(), src.clone())],
                };
                check(&files)
            }
            "run" => run(&src, events, req["fuel"].as_u64(), req["schedule_seed"].as_u64()),
            "trace" => trace(&src, events),
            "host" => host(&src, &req["steps"], req["tear"].as_u64()),
            "replicate" => replicate(req["nodes"].as_u64().unwrap_or(3) as usize, &req["script"]),
            op => json!({"error": format!("unknown op `{op}`")}),
        };
        resp["id"] = req["id"].clone();
        let _ = writeln!(out, "{resp}");
        let _ = out.flush();
    }
}
