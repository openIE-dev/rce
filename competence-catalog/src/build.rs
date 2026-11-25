//! The build/compilation layer — Blaze (Indexable Inc) ingested into the vertical
//! stack. RCE is all-encompassing: it does the build layer too, not just the
//! competence economy above it.
//!
//! Blaze's core design — "a result is a pure function of content, keyed by the
//! hash of exactly what it read, computed once, kept in one store" — applied to
//! Form checking. The build cache is content-addressed and incremental: a result
//! is keyed by the *canonical* (name-free) content hash, so "a rename, a move, a
//! comment, a reformat compile nothing" (the key holds behavior, not surface).
//! An edit reruns only the changed content; unchanged content is a cache hit.
//!
//! The evidence ladder (Blaze's, named): tested → simulated → model-checked →
//! proved. A fresh build is `Tested` (checked); assurance is promoted monotonically
//! as evidence accrues, never downgraded.

use std::collections::HashMap;

use crate::canonical::canonical_key;

/// The evidence ladder, weakest first (Blaze's assurance rungs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Evidence {
    /// Checked / spectest-passed.
    Tested,
    /// Deterministically simulated.
    Simulated,
    /// Model-checked.
    ModelChecked,
    /// Proved (Lean, independently rechecked).
    Proved,
}

impl Evidence {
    /// The rung's word.
    pub fn word(self) -> &'static str {
        match self {
            Evidence::Tested => "tested",
            Evidence::Simulated => "simulated",
            Evidence::ModelChecked => "model-checked",
            Evidence::Proved => "proved",
        }
    }
}

/// A content-addressed build result.
#[derive(Clone, Debug)]
pub struct BuildRecord {
    /// The name-free content key (canonical hash).
    pub key: String,
    /// Whether the content checks.
    pub ok: bool,
    /// The assurance rung reached.
    pub evidence: Evidence,
    /// The checker diagnostic, if any.
    pub diagnostic: String,
}

/// A content-addressed, incremental build cache (one store, keyed by content).
pub struct BuildCache {
    store: HashMap<String, BuildRecord>,
    /// Cache hits (content already built — nothing recompiled).
    pub hits: u64,
    /// Cache misses (new content built).
    pub misses: u64,
}

impl BuildCache {
    /// An empty store.
    pub fn new() -> Self {
        BuildCache { store: HashMap::new(), hits: 0, misses: 0 }
    }

    /// Build content: a cache hit reuses the result (nothing recompiled); a miss
    /// checks the content once and stores it. The key is canonical (name-free),
    /// so accidental representation changes hit the cache.
    pub fn build(&mut self, src: &str) -> BuildRecord {
        let key = canonical_key(src)
            .unwrap_or_else(|_| rcevc_core::sha256(src.as_bytes()).hex());
        if let Some(rec) = self.store.get(&key) {
            self.hits += 1;
            return rec.clone();
        }
        self.misses += 1;
        let (ok, diagnostic) = match rce_spec_check::check_one(src) {
            Ok(()) => (true, String::new()),
            Err(e) => (false, e),
        };
        let rec = BuildRecord { key: key.clone(), ok, evidence: Evidence::Tested, diagnostic };
        self.store.insert(key, rec.clone());
        rec
    }

    /// Promote a record's assurance monotonically up the evidence ladder (never
    /// downgraded). Returns the new rung.
    pub fn promote(&mut self, key: &str, to: Evidence) -> Option<Evidence> {
        let rec = self.store.get_mut(key)?;
        if to > rec.evidence {
            rec.evidence = to;
        }
        Some(rec.evidence)
    }

    /// The fraction of builds that were cache hits (reuse of compilation work).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    /// Distinct content keys stored.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}

impl Default for BuildCache {
    fn default() -> Self {
        Self::new()
    }
}

/// DBSP item-grain incremental build (Blaze v2 ingested): a record is valid by its
/// key or by its read-list; a body edit reruns that body's tasks and nothing else;
/// `why` explains reruns from the recorded reads, never by guessing.
pub struct IncrementalBuild {
    /// The content-addressed result store.
    pub cache: BuildCache,
    content: HashMap<String, String>,
    reads: HashMap<String, Vec<String>>,
}

impl IncrementalBuild {
    /// An empty incremental build.
    pub fn new() -> Self {
        IncrementalBuild { cache: BuildCache::new(), content: HashMap::new(), reads: HashMap::new() }
    }

    /// Register an item with its content and the items it reads.
    pub fn set_item(&mut self, name: &str, content: &str, reads: &[&str]) {
        self.content.insert(name.to_string(), content.to_string());
        self.reads.insert(name.to_string(), reads.iter().map(|s| s.to_string()).collect());
    }

    /// Change an item's content and return the rerun set: the item plus every item
    /// that transitively reads it. Nothing else reruns (item-grain incrementality).
    pub fn change(&mut self, name: &str, new_content: &str) -> Vec<String> {
        self.content.insert(name.to_string(), new_content.to_string());
        // Reverse dependency graph: for each item, who reads it.
        let mut reverse: HashMap<String, Vec<String>> = HashMap::new();
        for (reader, reads) in &self.reads {
            for r in reads {
                reverse.entry(r.clone()).or_default().push(reader.clone());
            }
        }
        let mut seen = std::collections::HashSet::new();
        seen.insert(name.to_string());
        let mut rerun = vec![name.to_string()];
        let mut queue = vec![name.to_string()];
        while let Some(cur) = queue.pop() {
            if let Some(dependents) = reverse.get(&cur) {
                for d in dependents {
                    if seen.insert(d.clone()) {
                        rerun.push(d.clone());
                        queue.push(d.clone());
                    }
                }
            }
        }
        rerun.sort();
        rerun
    }

    /// Explain an item's reruns from its recorded reads (Blaze's `blaze why`).
    pub fn why(&self, name: &str) -> String {
        match self.reads.get(name) {
            Some(reads) if !reads.is_empty() => {
                format!("{name} reads {}; it reruns when any of them change", reads.join(", "))
            }
            _ => format!("{name} reads nothing; it reruns only when its own content changes"),
        }
    }

    /// Build an item through the content-addressed cache.
    pub fn build_item(&mut self, name: &str) -> Option<BuildRecord> {
        let content = self.content.get(name)?.clone();
        Some(self.cache.build(&content))
    }
}

impl Default for IncrementalBuild {
    fn default() -> Self {
        Self::new()
    }
}
