//! The storage trait and the bitemporal semantics on top of it.
//!
//! A backend implements five primitives ([`MemoryEngine::next_id`],
//! [`MemoryEngine::commit`], [`MemoryEngine::get`], [`MemoryEngine::scan`],
//! [`MemoryEngine::health`]). `commit` is atomic: a [`Plan`] lands whole or
//! not at all. Everything else (store with contradiction handling, forget,
//! fetch, list, as-of queries, history) is provided once here, so every
//! backend has the same meaning; [`crate::conformance`] checks it.
use crate::fact::{AsOf, Fact, NewFact};
use std::collections::BTreeMap;

#[derive(Debug, thiserror::Error)]
pub enum MemError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("no fact with id {0}")]
    NotFound(u64),
    #[error("invalid: {0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, MemError>;

/// An atomic change: new rows, and rows to expire at a system time.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Plan {
    pub inserts: Vec<Fact>,
    pub expires: Vec<(u64, i64)>,
    /// For `store`: the id of the new fact's row (None for a duplicate).
    pub new_id: Option<u64>,
    /// For `store`: an existing live row that already says the same thing.
    pub duplicate_of: Option<u64>,
}

impl Plan {
    pub fn is_empty(&self) -> bool {
        self.inserts.is_empty() && self.expires.is_empty()
    }
    /// Stamp every inserted row with the receipt that produced it.
    pub fn stamp(&mut self, receipt: &str) {
        for f in &mut self.inserts {
            f.receipt_id = Some(receipt.to_string());
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Health {
    pub backend: &'static str,
    pub rows: u64,
}

pub trait MemoryEngine {
    /// Allocate a fresh row id (monotonic, never reused).
    fn next_id(&mut self) -> Result<u64>;
    /// Apply a plan atomically. Expiring an already expired row or a missing
    /// row is an error and nothing is applied.
    fn commit(&mut self, plan: &Plan) -> Result<()>;
    fn get(&self, id: u64) -> Result<Option<Fact>>;
    /// Every row, ascending id.
    fn scan(&self) -> Result<Vec<Fact>>;
    fn health(&self) -> Result<Health>;

    /// Plan storing `n` at system time `now`.
    ///
    /// Contradiction (same subject and predicate, a different object,
    /// overlapping world time, `multi` false):
    /// * an existing live row that began before the new fact is expired and
    ///   replaced by a successor ending at the new `valid_from`;
    /// * an existing row that began at or after the new fact stays, and the
    ///   new fact is closed at that row's `valid_from` (the newer world-time
    ///   claim wins regardless of arrival order).
    ///
    /// The same object with overlapping time is a duplicate: nothing is written.
    fn plan_store(&mut self, n: &NewFact, now: i64) -> Result<Plan> {
        if n.subject.is_empty() || n.predicate.is_empty() {
            return Err(MemError::Invalid(
                "subject and predicate are required".into(),
            ));
        }
        if n.valid_to.is_some_and(|t| t <= n.valid_from) {
            return Err(MemError::Invalid(
                "valid_to must be after valid_from".into(),
            ));
        }
        let live: Vec<Fact> = self
            .scan()?
            .into_iter()
            .filter(|f| f.live() && f.subject == n.subject && f.predicate == n.predicate)
            .collect();
        let mut plan = Plan::default();
        if let Some(d) = live.iter().find(|f| {
            f.object == n.object && f.text == n.text && f.overlaps(n.valid_from, n.valid_to)
        }) {
            plan.duplicate_of = Some(d.id);
            return Ok(plan);
        }
        let mut valid_to = n.valid_to;
        if !n.multi {
            for r in live.iter().filter(|f| !f.multi && f.object != n.object) {
                if !r.overlaps(n.valid_from, valid_to) {
                    continue;
                }
                if r.valid_from < n.valid_from {
                    plan.expires.push((r.id, now));
                    let mut s = r.clone();
                    s.id = self.next_id()?;
                    s.supersedes = Some(r.id);
                    s.valid_to = Some(n.valid_from);
                    s.recorded_at = now;
                    s.expired_at = None;
                    plan.inserts.push(s);
                } else {
                    valid_to = Some(valid_to.map_or(r.valid_from, |t| t.min(r.valid_from)));
                }
            }
        }
        let id = self.next_id()?;
        plan.inserts.push(Fact {
            id,
            lineage: id,
            supersedes: None,
            subject: n.subject.clone(),
            predicate: n.predicate.clone(),
            object: n.object.clone(),
            text: n.text.clone(),
            multi: n.multi,
            valid_from: n.valid_from,
            valid_to,
            recorded_at: now,
            expired_at: None,
            form_hash: n.form_hash.clone(),
            provenance: n.provenance,
            receipt_id: None,
            source: n.source.clone(),
        });
        plan.new_id = Some(id);
        Ok(plan)
    }

    /// Plan forgetting the live row `id` at system time `now`.
    /// `ended_at: None` retracts it (memory stops believing it was ever
    /// true; the row stays queryable with an earlier `known_at`).
    /// `ended_at: Some(t)` says it stopped being true at world time `t`: the
    /// row is expired and a successor closed at `t` is recorded.
    fn plan_forget(&mut self, id: u64, ended_at: Option<i64>, now: i64) -> Result<Plan> {
        let r = self.get(id)?.ok_or(MemError::NotFound(id))?;
        if !r.live() {
            return Err(MemError::Invalid(format!("fact {id} is already expired")));
        }
        let mut plan = Plan {
            expires: vec![(id, now)],
            ..Default::default()
        };
        if let Some(t) = ended_at {
            if t <= r.valid_from {
                return Err(MemError::Invalid("ended_at is not after valid_from".into()));
            }
            let mut s = r.clone();
            s.id = self.next_id()?;
            s.supersedes = Some(id);
            s.valid_to = Some(r.valid_to.map_or(t, |v| v.min(t)));
            s.recorded_at = now;
            plan.new_id = Some(s.id);
            plan.inserts.push(s);
        }
        Ok(plan)
    }

    fn fetch(&self, id: u64) -> Result<Fact> {
        self.get(id)?.ok_or(MemError::NotFound(id))
    }

    /// Rows matching a bitemporal query.
    fn as_of(&self, q: AsOf) -> Result<Vec<Fact>> {
        Ok(self.scan()?.into_iter().filter(|f| q.matches(f)).collect())
    }

    /// Live rows (currently believed), any world time.
    fn list_live(&self) -> Result<Vec<Fact>> {
        Ok(self.scan()?.into_iter().filter(Fact::live).collect())
    }

    /// Every row of a logical fact, oldest first.
    fn history(&self, lineage: u64) -> Result<Vec<Fact>> {
        Ok(self
            .scan()?
            .into_iter()
            .filter(|f| f.lineage == lineage)
            .collect())
    }

    /// Plan then commit; returns the plan that landed.
    fn store(&mut self, n: &NewFact, now: i64) -> Result<Plan> {
        let p = self.plan_store(n, now)?;
        if !p.is_empty() {
            self.commit(&p)?;
        }
        Ok(p)
    }

    fn forget(&mut self, id: u64, ended_at: Option<i64>, now: i64) -> Result<Plan> {
        let p = self.plan_forget(id, ended_at, now)?;
        self.commit(&p)?;
        Ok(p)
    }
}

/// Validate a plan against current rows (shared by backends).
pub fn check_plan(get: impl Fn(u64) -> Result<Option<Fact>>, plan: &Plan) -> Result<()> {
    for &(id, _) in &plan.expires {
        match get(id)? {
            None => return Err(MemError::NotFound(id)),
            Some(f) if !f.live() => {
                return Err(MemError::Invalid(format!("fact {id} is already expired")));
            }
            _ => {}
        }
    }
    for f in &plan.inserts {
        if get(f.id)?.is_some() {
            return Err(MemError::Invalid(format!("row {} already exists", f.id)));
        }
    }
    Ok(())
}

/// Reference engine: a map in memory. Used by tests and as the spec the
/// conformance suite is written against.
#[derive(Debug, Default)]
pub struct MemEngine {
    rows: BTreeMap<u64, Fact>,
    next: u64,
}

impl MemoryEngine for MemEngine {
    fn next_id(&mut self) -> Result<u64> {
        self.next += 1;
        Ok(self.next)
    }
    fn commit(&mut self, plan: &Plan) -> Result<()> {
        check_plan(|id| Ok(self.rows.get(&id).cloned()), plan)?;
        for &(id, t) in &plan.expires {
            if let Some(f) = self.rows.get_mut(&id) {
                f.expired_at = Some(t);
            }
        }
        for f in &plan.inserts {
            self.rows.insert(f.id, f.clone());
        }
        Ok(())
    }
    fn get(&self, id: u64) -> Result<Option<Fact>> {
        Ok(self.rows.get(&id).cloned())
    }
    fn scan(&self) -> Result<Vec<Fact>> {
        Ok(self.rows.values().cloned().collect())
    }
    fn health(&self) -> Result<Health> {
        Ok(Health {
            backend: "memory",
            rows: self.rows.len() as u64,
        })
    }
}
