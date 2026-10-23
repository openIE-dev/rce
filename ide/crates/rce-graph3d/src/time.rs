//! The time axis: Form versions, bitemporal facts and receipts, scrubbed
//! together. A snapshot at `t` is the latest Form version at or before
//! `t`, with the actions up to `t` applied and the facts that were both
//! true and known at `t`.
use crate::graph::Graph;
use crate::overlay::{self, Action};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Version {
    pub t_ms: i64,
    pub label: String,
    pub graph: Graph,
}

/// A bitemporal fact, flattened from rce-memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactIn {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub valid_from: i64,
    pub valid_to: Option<i64>,
    pub recorded_at: i64,
    pub expired_at: Option<i64>,
}

impl FactIn {
    pub fn at(&self, t: i64) -> bool {
        self.valid_from <= t
            && self.valid_to.is_none_or(|e| e > t)
            && self.recorded_at <= t
            && self.expired_at.is_none_or(|e| e > t)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Timeline {
    pub versions: Vec<Version>,
    pub facts: Vec<FactIn>,
    pub actions: Vec<Action>,
    /// Hardware merged into every snapshot.
    #[serde(default)]
    pub hardware: Graph,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Snapshot {
    pub t_ms: i64,
    pub version: Option<usize>,
    pub graph: Graph,
    /// Node ids new in this version / gone since the previous one.
    pub added: BTreeSet<String>,
    pub removed: BTreeSet<String>,
    /// Facts true and known at `t`, by subject.
    pub facts: BTreeMap<String, Vec<String>>,
    pub actions: usize,
}

impl Timeline {
    pub fn sort(&mut self) {
        self.versions.sort_by_key(|v| v.t_ms);
        self.actions.sort_by_key(|a| (a.t_ms, a.seq));
    }
    /// Every event time on the axis, ascending and unique.
    pub fn ticks(&self) -> Vec<i64> {
        let mut t: BTreeSet<i64> = self.versions.iter().map(|v| v.t_ms).collect();
        t.extend(self.actions.iter().map(|a| a.t_ms));
        for f in &self.facts {
            t.insert(f.valid_from);
            t.insert(f.recorded_at);
            t.extend(f.valid_to);
            t.extend(f.expired_at);
        }
        t.into_iter().collect()
    }
    pub fn range(&self) -> Option<(i64, i64)> {
        let t = self.ticks();
        Some((*t.first()?, *t.last()?))
    }
    /// Index of the version each node id first appears in.
    pub fn first_seen(&self) -> BTreeMap<String, u32> {
        let mut m = BTreeMap::new();
        for (i, v) in self.versions.iter().enumerate() {
            for n in &v.graph.nodes {
                m.entry(n.id.clone()).or_insert(i as u32);
            }
        }
        m
    }
    pub fn at(&self, t: i64, k_sigma: f64) -> Snapshot {
        let vi = self.versions.iter().rposition(|v| v.t_ms <= t);
        let mut g = vi
            .map(|i| self.versions[i].graph.clone())
            .unwrap_or_default();
        g.merge(&self.hardware);
        let ids = |g: &Graph| {
            g.nodes
                .iter()
                .filter(|n| !n.kind.is_hw())
                .map(|n| n.id.clone())
                .collect::<BTreeSet<_>>()
        };
        let now = ids(&g);
        let prev = match vi {
            Some(i) if i > 0 => ids(&self.versions[i - 1].graph),
            _ => BTreeSet::new(),
        };
        let acts: Vec<Action> = self
            .actions
            .iter()
            .filter(|a| a.t_ms <= t)
            .cloned()
            .collect();
        overlay::apply(&mut g, &acts, k_sigma);
        let mut facts: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for f in self.facts.iter().filter(|f| f.at(t)) {
            facts
                .entry(f.subject.clone())
                .or_default()
                .push(format!("{} {}", f.predicate, f.object));
        }
        Snapshot {
            t_ms: t,
            version: vi,
            added: now.difference(&prev).cloned().collect(),
            removed: prev.difference(&now).cloned().collect(),
            graph: g,
            facts,
            actions: acts.len(),
        }
    }
    /// Scrubber position in [0, 1] to a time on the axis.
    pub fn scrub(&self, u: f64) -> Option<i64> {
        let (a, b) = self.range()?;
        Some(a + ((b - a) as f64 * u.clamp(0.0, 1.0)).round() as i64)
    }
}
