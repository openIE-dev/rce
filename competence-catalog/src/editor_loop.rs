//! The competence-backed editor loop (prototype, Rust first). Every edit is lifted to a Form,
//! gated by replay BEFORE it is served; admitted competence is served by content key at the G0
//! lookup floor. Diagnostics are admission failures caught at edit time; debugging is replay
//! divergence localized to a probe/construct. The zero-error hypothesis: because nothing is
//! served unless it replays, errors cannot escape to runtime and post-hoc debug sessions are 0.
//!
//! This is a simulation of the edit→gate→admit→lookup loop using the real replay gate and real
//! deterministic execution; wiring it into a live editor (LSP diagnostics-on-type) is open.

use std::collections::BTreeMap;

use crate::lut_emit::{builtin_luts, render_stmt, LanguageLut, Stmt};
use crate::program::{eval_program, lift_program, program_key, replay_admit, Env, FnTable};

#[derive(Debug, Clone, Default)]
pub struct LoopStats {
    pub edits: u64,
    /// Errors caught at edit time (parse or replay admission failure) — never reach runtime.
    pub caught_at_edit: u64,
    /// Served from already-admitted competence (G0 lookup, ~1 µJ, 0 model calls).
    pub served_hits: u64,
    /// First-time admissions (replay-verified before serving).
    pub admitted: u64,
    /// Errors that escaped to runtime after being served (hypothesis: always 0).
    pub escaped_errors: u64,
    /// Post-hoc debug sessions needed (hypothesis: always 0).
    pub debug_sessions: u64,
    pub cost_uj: u64,
}

#[derive(Debug, Clone)]
pub enum EditOutcome {
    /// Admitted (or hit) and served; carries the content key and the Rust emission.
    Served { key: String, rust: String, hit: bool },
    /// Rejected at edit time with a diagnostic — the error never reaches runtime.
    Rejected { diagnostic: String },
}

pub struct EditorLoop {
    pub lut: LanguageLut,
    pub admitted: BTreeMap<String, Stmt>,
    pub stats: LoopStats,
}

impl EditorLoop {
    /// A loop targeting Rust (the exemplar: rust-analyzer + borrow checker).
    pub fn rust() -> Self {
        let lut = builtin_luts().into_iter().find(|l| l.name == "rust").expect("rust lut");
        EditorLoop { lut, admitted: BTreeMap::new(), stats: LoopStats::default() }
    }

    /// One edit: lift → replay-gate → admit-or-hit → serve. Errors are caught here, at edit time.
    pub fn edit(&mut self, src: &str, probes: &[(Env, i128)]) -> EditOutcome {
        self.stats.edits += 1;
        let stmt = match lift_program(src) {
            Ok(s) => s,
            Err(e) => {
                self.stats.caught_at_edit += 1;
                return EditOutcome::Rejected { diagnostic: format!("parse: {e}") };
            }
        };
        if !replay_admit(&stmt, probes) {
            self.stats.caught_at_edit += 1;
            let divergent = localize_divergence(&stmt, probes);
            return EditOutcome::Rejected {
                diagnostic: format!("replay divergence at probe {:?}", divergent),
            };
        }
        let key = match program_key(&stmt) {
            Ok(k) => k,
            Err(e) => {
                self.stats.caught_at_edit += 1;
                return EditOutcome::Rejected { diagnostic: format!("key: {e}") };
            }
        };
        let hit = self.admitted.contains_key(&key);
        if hit {
            self.stats.served_hits += 1;
            self.stats.cost_uj += 1; // G0 lookup floor
        } else {
            self.admitted.insert(key.clone(), stmt.clone());
            self.stats.admitted += 1;
            self.stats.cost_uj += 1000; // modelled admission/check cost
        }
        let rust = render_stmt(&stmt, &self.lut).unwrap_or_default();
        EditOutcome::Served { key, rust, hit }
    }

    /// Serve an admitted competence by key and RE-VERIFY by replay; any divergence would be an
    /// escaped error (hypothesis: none, because admission already replayed every probe).
    pub fn serve(&mut self, key: &str, probes: &[(Env, i128)]) -> Option<String> {
        let stmt = self.admitted.get(key)?.clone();
        if !replay_admit(&stmt, probes) {
            self.stats.escaped_errors += 1;
            self.stats.debug_sessions += 1;
            return None;
        }
        self.stats.served_hits += 1;
        self.stats.cost_uj += 1;
        render_stmt(&stmt, &self.lut).ok()
    }

    pub fn cost_per_act(&self) -> f64 {
        let acts = (self.stats.served_hits + self.stats.admitted).max(1);
        self.stats.cost_uj as f64 / acts as f64
    }
}

/// Localize the first probe whose deterministic execution diverges from expectation — the
/// construct-level debugging signal (replay divergence), instead of post-hoc runtime debugging.
pub fn localize_divergence(s: &Stmt, probes: &[(Env, i128)]) -> Option<usize> {
    probes.iter().position(|(seed, expected)| {
        let mut env = seed.clone();
        !matches!(eval_program(s, &mut env, &mut FnTable::new()), Ok(Some(v)) if v == *expected)
    })
}
