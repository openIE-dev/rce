//! The memory service: one receipt per recall, store, forget, repo map and
//! note sync.
//!
//! Receipt order for writes: the plan is computed, the receipt is appended
//! (its detail names the rows), the rows are stamped with the receipt hash
//! and committed. If the commit then fails, a second receipt records the
//! failure, so the chain never claims a write that did not land silently.
//! E_sim is predicted from the realized telemetry features (rows touched,
//! gear); E_meas is the meter delta over the action window when a meter
//! exists, else absent.
use crate::cascade::{self, Recall};
use crate::engine::{MemError, MemoryEngine, Plan};
use crate::fact::{AsOf, Fact, NewFact, Provenance};
use crate::form::{FormIndex, IndexCache, form_hash};
use crate::notes::{self, Notes, SyncReport};
use crate::redb_engine::RedbEngine;
use crate::vector::VectorIndex;
use ide_energy::meter::now_ns;
use ide_energy::shared::SharedMeter;
use ide_energy::sim::{Features, Simulator};
use ide_receipt::{ActionKind, Draft, Ledger, Measured, MeterSource, Receipt, Verdict};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

/// What a memory call needs to write its receipt (borrowed from the host,
/// so memory receipts join the host's single chain).
pub struct ReceiptCtx<'a> {
    pub ledger: &'a mut Ledger,
    pub sim: &'a dyn Simulator,
    pub meter: Option<&'a SharedMeter>,
    pub policy_hash: &'a str,
}

#[derive(Debug, thiserror::Error)]
pub enum ServiceError {
    #[error(transparent)]
    Mem(#[from] MemError),
    #[error(transparent)]
    Note(#[from] notes::NoteError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("receipt: {0}")]
    Receipt(String),
}

pub type SResult<T> = Result<T, ServiceError>;

fn source_tag(s: &str) -> Option<MeterSource> {
    match s {
        "rapl" => Some(MeterSource::Rapl),
        "nvml" => Some(MeterSource::Nvml),
        "powermetrics" => Some(MeterSource::Powermetrics),
        "ioreport" => Some(MeterSource::Ioreport),
        "smc" => Some(MeterSource::Smc),
        "hwmon" => Some(MeterSource::Hwmon),
        "power-supply" => Some(MeterSource::PowerSupply),
        "jetson" => Some(MeterSource::Jetson),
        _ => None,
    }
}

/// A started action: start time and meter sample.
struct Span {
    t0: u64,
    span: Option<ide_energy::shared::Span>,
}

fn begin(ctx: &mut ReceiptCtx) -> Span {
    Span {
        t0: now_ns(),
        span: ctx.meter.and_then(|m| m.begin()),
    }
}

/// Features for a memory action: rows touched go in `form_nodes` (the
/// size term the simulator already fits), the gear names the coefficient set.
pub fn features(gear: &str, touched: u64) -> Features {
    Features {
        form_nodes: touched,
        gear: format!("memory.{gear}"),
        ..Default::default()
    }
}

fn finish(
    ctx: &mut ReceiptCtx,
    sp: Span,
    action: &str,
    feats: Features,
    verdict: Verdict,
    detail: Value,
) -> SResult<Receipt> {
    let j1 = match (ctx.meter, sp.span) {
        (Some(m), Some(s)) => m.end(s),
        _ => None,
    };
    let mut d = Draft::new(
        ActionKind::Other(action.into()),
        feats.gear.clone(),
        ctx.policy_hash,
    );
    d.t_start_ns = sp.t0;
    d.t_end_ns = now_ns();
    d.e_sim_j = ctx.sim.predict(&feats).map(|p| p.joules);
    if let (Some(j), Some(src)) = (j1, ctx.meter.and_then(|m| source_tag(m.source()))) {
        d.e_meas = Some(Measured {
            joules: j,
            source: src,
        });
    }
    d.verdict = verdict;
    d.detail = json!({"action": action, "features": feats, "detail": detail});
    ctx.ledger
        .append(d)
        .map_err(|e| ServiceError::Receipt(e.to_string()))
}

/// Ask to remember something.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Remember {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub multi: bool,
    #[serde(default)]
    pub valid_from: Option<i64>,
    #[serde(default)]
    pub valid_to: Option<i64>,
    #[serde(default = "default_prov")]
    pub provenance: Provenance,
    /// Also write the fact as a Markdown note in the notes directory.
    #[serde(default)]
    pub note: bool,
}

fn default_prov() -> Provenance {
    Provenance::Inferred
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecallQuery {
    pub query: String,
    #[serde(default)]
    pub valid_at: Option<i64>,
    #[serde(default)]
    pub known_at: Option<i64>,
    #[serde(default)]
    pub exclude_model: bool,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    10
}

pub struct Memory<E: MemoryEngine> {
    pub engine: E,
    pub cache: IndexCache,
    pub notes: Notes,
    pub vector: Option<Box<dyn VectorIndex>>,
}

impl Memory<RedbEngine> {
    /// Default layout under a workspace root: `.rce-ide/memory.redb`,
    /// `.rce-ide/index/`, notes in `notes/`.
    pub fn open_default(root: &Path) -> SResult<Self> {
        let state = root.join(".rce-ide");
        std::fs::create_dir_all(&state)?;
        Ok(Memory {
            engine: RedbEngine::open(&state.join("memory.redb"))?,
            cache: IndexCache::new(&state.join("index"))?,
            notes: Notes::new(root, "notes")?,
            vector: None,
        })
    }
}

impl<E: MemoryEngine> Memory<E> {
    pub fn new(engine: E, cache: IndexCache, notes: Notes) -> Self {
        Memory {
            engine,
            cache,
            notes,
            vector: None,
        }
    }

    pub fn index(&self, program: &[(String, String)]) -> SResult<(FormIndex, bool)> {
        Ok(self.cache.load(program)?)
    }

    /// Apply a plan after its receipt; on failure write a failure receipt.
    fn land(
        &mut self,
        ctx: &mut ReceiptCtx,
        mut plan: Plan,
        r: &Receipt,
        action: &str,
    ) -> SResult<Plan> {
        plan.stamp(&r.hash);
        if plan.is_empty() {
            return Ok(plan);
        }
        if let Err(e) = self.engine.commit(&plan) {
            let sp = begin(ctx);
            finish(
                ctx,
                sp,
                action,
                features("commit", 0),
                Verdict::Rejected {
                    reason: format!("commit failed after receipt {}: {e}", r.hash),
                },
                json!({"failed_receipt": r.hash}),
            )?;
            return Err(e.into());
        }
        Ok(plan)
    }

    fn sync_inner(&mut self, fh: &str, now: i64) -> SResult<(Plan, SyncReport)> {
        Ok(notes::plan_sync(&self.notes, &mut self.engine, fh, now)?)
    }

    /// Make memory match the notes directory (files win). One receipt.
    pub fn sync_notes(
        &mut self,
        ctx: &mut ReceiptCtx,
        program: &[(String, String)],
        now: i64,
    ) -> SResult<(SyncReport, Receipt)> {
        let sp = begin(ctx);
        let fh = form_hash(program);
        let (plan, rep) = self.sync_inner(&fh, now)?;
        let touched = (plan.inserts.len() + plan.expires.len() + rep.unchanged) as u64;
        let r = finish(
            ctx,
            sp,
            "memory.sync",
            features("sync", touched),
            Verdict::NotRun,
            json!({"report": rep, "inserts": plan.inserts.iter().map(|f| f.id).collect::<Vec<_>>(), "expires": plan.expires}),
        )?;
        self.land(ctx, plan, &r, "memory.sync")?;
        Ok((rep, r))
    }

    /// Recall through the cascade. Notes are synced first, in the same
    /// action, so a hand edit is never shadowed by a stale fact.
    pub fn recall(
        &mut self,
        ctx: &mut ReceiptCtx,
        program: &[(String, String)],
        q: &RecallQuery,
        now: i64,
    ) -> SResult<(Recall, Receipt)> {
        let sp = begin(ctx);
        let (ix, cache_hit) = self.index(program)?;
        let (sync_plan, rep) = self.sync_inner(&ix.form_hash, now)?;
        // Facts visible after the sync: committed rows plus the sync plan.
        let mut rows = self.engine.scan()?;
        for &(id, t) in &sync_plan.expires {
            if let Some(f) = rows.iter_mut().find(|f| f.id == id) {
                f.expired_at = Some(t);
            }
        }
        rows.extend(sync_plan.inserts.iter().cloned());
        let asof = AsOf {
            valid_at: q.valid_at.unwrap_or(now),
            known_at: q.known_at.unwrap_or(now),
        };
        let facts: Vec<Fact> = rows
            .into_iter()
            .filter(|f| asof.matches(f) && !(q.exclude_model && f.provenance == Provenance::Model))
            .collect();
        let rc = cascade::recall(&ix, &facts, self.vector.as_deref(), &q.query, q.limit);
        let gear = rc.found.map(|g| g.as_str()).unwrap_or("not_found");
        let r = finish(
            ctx,
            sp,
            "memory.recall",
            features(gear, rc.touched()),
            Verdict::NotRun,
            json!({
                "query": q, "form_hash": ix.form_hash, "index_cache_hit": cache_hit,
                "found": rc.found, "tried": rc.tried, "escalate": rc.escalate,
                "hits": rc.hits.iter().map(|h| h.key.clone()).collect::<Vec<_>>(),
                "sync": rep,
            }),
        )?;
        self.land(ctx, sync_plan, &r, "memory.recall")?;
        Ok((rc, r))
    }

    /// Store a fact pinned to the current Form hash.
    pub fn remember(
        &mut self,
        ctx: &mut ReceiptCtx,
        program: &[(String, String)],
        req: &Remember,
        now: i64,
    ) -> SResult<(Plan, Receipt)> {
        let sp = begin(ctx);
        let fh = form_hash(program);
        let mut n = NewFact {
            subject: req.subject.clone(),
            predicate: req.predicate.clone(),
            object: req.object.clone(),
            text: req.text.clone(),
            multi: req.multi,
            valid_from: req.valid_from.unwrap_or(now),
            valid_to: req.valid_to,
            form_hash: fh,
            provenance: req.provenance,
            source: None,
        };
        if req.note {
            let rel = format!(
                "{}/{}.md",
                self.notes.dir,
                notes::slug(&format!("{} {}", req.subject, req.predicate))
            );
            if let Err(e) = notes::check_write_path(&rel) {
                let r = finish(
                    ctx,
                    sp,
                    "memory.remember",
                    features("store", 0),
                    Verdict::Rejected {
                        reason: e.to_string(),
                    },
                    json!({"request": req}),
                )?;
                return Ok((Plan::default(), r));
            }
            n.source = Some(rel);
        }
        let plan = match self.engine.plan_store(&n, now) {
            Ok(p) => p,
            Err(e) => {
                let r = finish(
                    ctx,
                    sp,
                    "memory.remember",
                    features("store", 0),
                    Verdict::Rejected {
                        reason: e.to_string(),
                    },
                    json!({"request": req}),
                )?;
                return Ok((Plan::default(), r));
            }
        };
        let touched = self.engine.health()?.rows;
        let r = finish(
            ctx,
            sp,
            "memory.remember",
            features("store", touched),
            Verdict::Pass,
            json!({
                "request": req, "form_hash": n.form_hash, "new_id": plan.new_id, "duplicate_of": plan.duplicate_of,
                "inserts": plan.inserts.iter().map(|f| f.id).collect::<Vec<_>>(), "expires": plan.expires,
            }),
        )?;
        let plan = self.land(ctx, plan, &r, "memory.remember")?;
        if let (Some(id), Some(rel)) = (plan.new_id, &n.source) {
            let f = self.engine.fetch(id)?;
            self.notes
                .write(rel, &notes::render_note(&notes::fact_note(&f)))?;
        }
        Ok((plan, r))
    }

    /// Forget = expire. `ended_at` closes world time; `None` retracts.
    /// A fact mirrored in a note gets `status: forgotten` in its file (the
    /// file is kept, never deleted).
    pub fn forget(
        &mut self,
        ctx: &mut ReceiptCtx,
        id: u64,
        ended_at: Option<i64>,
        now: i64,
    ) -> SResult<(Plan, Receipt)> {
        let sp = begin(ctx);
        let plan = match self.engine.plan_forget(id, ended_at, now) {
            Ok(p) => p,
            Err(e) => {
                let r = finish(
                    ctx,
                    sp,
                    "memory.forget",
                    features("forget", 0),
                    Verdict::Rejected {
                        reason: e.to_string(),
                    },
                    json!({"id": id}),
                )?;
                return Ok((Plan::default(), r));
            }
        };
        let r = finish(
            ctx,
            sp,
            "memory.forget",
            features("forget", 1),
            Verdict::Pass,
            json!({"id": id, "ended_at": ended_at, "expires": plan.expires, "inserts": plan.inserts.iter().map(|f| f.id).collect::<Vec<_>>()}),
        )?;
        let plan = self.land(ctx, plan, &r, "memory.forget")?;
        let old = self.engine.fetch(id)?;
        if let Some(rel) = &old.source
            && ended_at.is_none()
        {
            self.notes
                .write(rel, &notes::render_note(&notes::fact_note(&old)))?;
        }
        Ok((plan, r))
    }

    /// Ranked repo map, personalized to `focus` def names.
    pub fn repo_map(
        &mut self,
        ctx: &mut ReceiptCtx,
        program: &[(String, String)],
        focus: &[String],
        budget_chars: usize,
    ) -> SResult<(String, Vec<String>, Receipt)> {
        let sp = begin(ctx);
        let (ix, hit) = self.index(program)?;
        let f: Vec<usize> = focus.iter().flat_map(|n| ix.find(n)).collect();
        let (text, picked) = ix.repo_map(&f, budget_chars);
        let names: Vec<String> = picked.iter().map(|&i| ix.defs[i].name.clone()).collect();
        let touched = (ix.defs.len() + ix.refs.len()) as u64;
        let r = finish(
            ctx,
            sp,
            "memory.repo_map",
            features("repo_map", touched),
            Verdict::NotRun,
            json!({"form_hash": ix.form_hash, "index_cache_hit": hit, "focus": focus, "budget_chars": budget_chars, "defs": names.len()}),
        )?;
        Ok((text, names, r))
    }
}
