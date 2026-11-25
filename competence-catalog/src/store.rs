//! The store layer — Blaze's hot/warm/cold tiers and trust-by-verification
//! ingested into the vertical stack.
//!
//! Blaze keeps one content-addressed store in hot/warm/cold tiers; a remote cache
//! is "the same store, the same keys", and trust is by verification — sampled hits
//! are recomputed, never blindly believed. This is that store for the competence
//! economy: entries promote to hot on access, demote by capacity (LRU), and a
//! sampled entry is recomputed and compared to detect corruption or drift. The
//! gate's discipline (verify-before-trust) applied to the store itself.

use std::collections::HashMap;

/// The storage tier (Blaze's hot/warm/cold).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// In fast reach (recently used).
    Hot,
    /// Near storage.
    Warm,
    /// Archive (cold).
    Cold,
}

#[derive(Clone)]
struct Entry<V> {
    value: V,
    tier: Tier,
    age: u64,
}

/// A content-addressed tiered store with sampled-recompute trust.
pub struct TieredStore<V: Clone + PartialEq> {
    entries: HashMap<String, Entry<V>>,
    /// Hot-tier capacity (overflow demotes to warm).
    pub hot_cap: usize,
    /// Warm-tier capacity (overflow demotes to cold).
    pub warm_cap: usize,
    clock: u64,
    /// How many sampled recomputes have run.
    pub recomputes: u64,
    /// How many sampled recomputes disagreed with the stored value.
    pub mismatches: u64,
}

impl<V: Clone + PartialEq> TieredStore<V> {
    /// A store with the given hot/warm capacities.
    pub fn new(hot_cap: usize, warm_cap: usize) -> Self {
        TieredStore {
            entries: HashMap::new(),
            hot_cap,
            warm_cap,
            clock: 0,
            recomputes: 0,
            mismatches: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Insert content into the hot tier; demote overflow by LRU.
    pub fn put(&mut self, key: &str, value: V) {
        let age = self.tick();
        self.entries.insert(key.to_string(), Entry { value, tier: Tier::Hot, age });
        self.demote();
    }

    /// Read content, promoting it to hot (access keeps competence in fast reach).
    pub fn get(&mut self, key: &str) -> Option<V> {
        let age = self.tick();
        let e = self.entries.get_mut(key)?;
        e.tier = Tier::Hot;
        e.age = age;
        let v = e.value.clone();
        self.demote();
        Some(v)
    }

    /// The tier an entry sits in.
    pub fn tier_of(&self, key: &str) -> Option<Tier> {
        self.entries.get(key).map(|e| e.tier)
    }

    /// Enforce capacities: oldest hot → warm, oldest warm → cold (LRU).
    fn demote(&mut self) {
        loop {
            let oldest = self.entries.iter()
                .filter(|(_, e)| e.tier == Tier::Hot)
                .min_by_key(|(_, e)| e.age)
                .map(|(k, _)| k.clone());
            let hot_count = self.entries.values().filter(|e| e.tier == Tier::Hot).count();
            match oldest {
                Some(k) if hot_count > self.hot_cap => {
                    if let Some(e) = self.entries.get_mut(&k) { e.tier = Tier::Warm; }
                }
                _ => break,
            }
        }
        loop {
            let oldest = self.entries.iter()
                .filter(|(_, e)| e.tier == Tier::Warm)
                .min_by_key(|(_, e)| e.age)
                .map(|(k, _)| k.clone());
            let warm_count = self.entries.values().filter(|e| e.tier == Tier::Warm).count();
            match oldest {
                Some(k) if warm_count > self.warm_cap => {
                    if let Some(e) = self.entries.get_mut(&k) { e.tier = Tier::Cold; }
                }
                _ => break,
            }
        }
    }

    /// Trust by verification: recompute a sampled entry and compare to the stored
    /// value. Returns whether they agree (None if the key or recompute is absent).
    /// A disagreement is counted — corruption/drift is detected, not trusted.
    pub fn sample_verify<F: FnOnce(&str) -> Option<V>>(&mut self, key: &str, recompute: F) -> Option<bool> {
        let stored = self.entries.get(key)?.value.clone();
        self.recomputes += 1;
        let fresh = recompute(key)?;
        let ok = fresh == stored;
        if !ok {
            self.mismatches += 1;
        }
        Some(ok)
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Count of entries in a tier.
    pub fn count_in(&self, tier: Tier) -> usize {
        self.entries.values().filter(|e| e.tier == tier).count()
    }
}
