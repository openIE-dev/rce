//! Frontier 2 — deterministic record/replay (rr / QEMU / Simics / ReEmu discipline).
//!
//! Whole-system determinism is solved engineering: record the non-deterministic inputs once,
//! replay the execution exactly, anywhere, any number of times. RCE's Form log D is the same
//! discipline at the competence level: a hash-chained record of acts whose replay is byte-identical.
//! This module makes it concrete and falsifiable: `record` executes deterministically and chains
//! hashes; `replay` re-executes and must reproduce every output and the chain; any tamper breaks
//! the chain and is detected. This hardens the Cleanroom fab and federation determinism claim.

use crate::lift::lift_expr;
use crate::repair::eval_expr;

/// One recorded act: the intent, its acceptance, the deterministic output, and the chain hash.
pub struct Event {
    pub intent: String,
    pub accept: i128,
    pub out: i128,
    pub hash: String,
}

/// A hash-chained recording of a deterministic run.
pub struct Recording {
    pub events: Vec<Event>,
}

fn chain(prev: &str, intent: &str, accept: i128, out: i128) -> String {
    let payload = format!("{prev}|{intent}|{accept}|{out}");
    rcevc_core::sha256(payload.as_bytes()).hex()
}

/// Record a deterministic run: execute each act, chain the hashes (rr-style event log).
pub fn record(tasks: &[(String, i128)]) -> Recording {
    let mut events = Vec::new();
    let mut prev = String::from("genesis");
    for (intent, accept) in tasks {
        let out = lift_expr(intent).ok().and_then(|e| eval_expr(&e)).unwrap_or(i128::MIN);
        let hash = chain(&prev, intent, *accept, out);
        prev = hash.clone();
        events.push(Event { intent: intent.clone(), accept: *accept, out, hash });
    }
    Recording { events }
}

/// Verify the hash chain is intact (any tamper to an event breaks it).
pub fn verify_chain(rec: &Recording) -> bool {
    let mut prev = String::from("genesis");
    for e in &rec.events {
        let expect = chain(&prev, &e.intent, e.accept, e.out);
        if expect != e.hash {
            return false;
        }
        prev = e.hash.clone();
    }
    true
}

/// Replay the recording: re-execute every act and require the identical output and an intact
/// chain. Returns the replayed outputs, or an error on any divergence (non-determinism/tamper).
pub fn replay(rec: &Recording) -> Result<Vec<i128>, String> {
    if !verify_chain(rec) {
        return Err("hash chain broken: recording tampered".into());
    }
    let mut outs = Vec::new();
    for e in &rec.events {
        let out = lift_expr(&e.intent).ok().and_then(|x| eval_expr(&x))
            .ok_or_else(|| format!("replay diverged: {} not evaluable", e.intent))?;
        if out != e.out {
            return Err(format!("replay diverged: {} gave {} recorded {}", e.intent, out, e.out));
        }
        outs.push(out);
    }
    Ok(outs)
}