//! Bitemporal facts.
//!
//! World time: `valid_from` .. `valid_to` (when the fact was true).
//! System time: `recorded_at` .. `expired_at` (when memory believed this row).
//! Rows are never deleted. Changing what memory believes expires a row (sets
//! `expired_at`) and, where the fact still held for part of the time, records
//! a successor row with a closed `valid_to`. All times are Unix milliseconds;
//! an open end is `None`. Intervals are half-open: `[from, to)`.
use serde::{Deserialize, Serialize};

/// Where a fact came from (Graphify-style tags, tied to receipts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Read directly from the Form or from a file (a Markdown note).
    Extracted,
    /// Derived by a deterministic rule.
    Inferred,
    /// Produced by a model call; recall can exclude these.
    Model,
}

/// One stored row version.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    /// Row id, unique per engine.
    pub id: u64,
    /// Logical fact id: the id of the first row of this fact.
    pub lineage: u64,
    /// The row this one replaced (a contradiction or a note edit), if any.
    pub supersedes: Option<u64>,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// Free text (a note body, a rationale) for keyword search.
    pub text: String,
    /// When true, many objects may hold at once for this subject and
    /// predicate, so a different object is not a contradiction.
    pub multi: bool,
    pub valid_from: i64,
    pub valid_to: Option<i64>,
    pub recorded_at: i64,
    pub expired_at: Option<i64>,
    /// Form version hash the fact is pinned to.
    pub form_hash: String,
    pub provenance: Provenance,
    /// Hash of the ide-receipt of the action that wrote this row.
    pub receipt_id: Option<String>,
    /// Workspace-relative Markdown path when the fact mirrors a note.
    pub source: Option<String>,
}

impl Fact {
    /// Was this row believed at system time `t`?
    pub fn known_at(&self, t: i64) -> bool {
        self.recorded_at <= t && self.expired_at.is_none_or(|e| e > t)
    }
    /// Does the fact claim to hold at world time `t`?
    pub fn valid_at(&self, t: i64) -> bool {
        self.valid_from <= t && self.valid_to.is_none_or(|e| e > t)
    }
    /// Is this row currently believed (not expired)?
    pub fn live(&self) -> bool {
        self.expired_at.is_none()
    }
    /// Do the world-time intervals overlap?
    pub fn overlaps(&self, from: i64, to: Option<i64>) -> bool {
        let a_end = self.valid_to.unwrap_or(i64::MAX);
        let b_end = to.unwrap_or(i64::MAX);
        self.valid_from < b_end && from < a_end
    }
}

/// What a caller asks to remember.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub multi: bool,
    pub valid_from: i64,
    #[serde(default)]
    pub valid_to: Option<i64>,
    pub form_hash: String,
    pub provenance: Provenance,
    #[serde(default)]
    pub source: Option<String>,
}

/// A bitemporal query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsOf {
    /// World time: what was true then.
    pub valid_at: i64,
    /// System time: according to what memory believed then.
    pub known_at: i64,
}

impl AsOf {
    pub fn now(t: i64) -> Self {
        AsOf {
            valid_at: t,
            known_at: t,
        }
    }
    pub fn matches(&self, f: &Fact) -> bool {
        f.known_at(self.known_at) && f.valid_at(self.valid_at)
    }
}
