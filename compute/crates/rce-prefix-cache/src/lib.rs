//! RadixAttention-style token-id-keyed prefix cache.
//!
//! When a runtime serves multiple completions that share a common
//! prefix (system prompt, chat history, RAG retrieval context), the
//! KV-cache state after the shared prefix is identical across them.
//! Re-computing the prefix forward pass each time wastes the bulk of
//! prefill compute — SGLang measures **75-90% hit rate on multi-turn
//! chat, ~6.4× speedup on RAG**.
//!
//! This crate is the substrate-side data structure. Each runtime
//! brings its own KV-state representation (its `BlockState` /
//! `Gemma4State` / etc.) and supplies a `Clone` impl plus an
//! "advance through `prompt[i]`" hook. The trie below stores
//! `(token_id_path, KvSnapshot)` and offers longest-prefix lookup
//! over an iterator of token ids.
//!
//! Single-stream local-first: this is **not** a PagedAttention pool.
//! KV snapshots are owned by the trie via clones — appropriate when
//! the runtime is one process serving a small number of concurrent
//! turns (chat tab + RAG worker, max ~16 in-flight). At higher
//! concurrency, switch to a paged backing store; this trait is the
//! same in both designs.
//!
//! # Example
//!
//! ```ignore
//! use rce_prefix_cache::{PrefixCache, KvSnapshot};
//!
//! let mut cache: PrefixCache<MyState> = PrefixCache::new(/*max_branches=*/ 64);
//!
//! // Existing state at prompt[..6] from a previous turn:
//! cache.insert(&prompt_ids[..6], my_state_at_pos_6.clone());
//!
//! // New turn shares the same prefix:
//! if let Some((matched_len, restored_state)) = cache.longest_prefix(&new_prompt_ids) {
//!     state = restored_state;        // skip prefill of [..matched_len]
//!     remaining = &new_prompt_ids[matched_len..];
//! }
//! // run normal prefill on `remaining`, then optionally cache the new
//! // suffix branch:
//! cache.insert(&new_prompt_ids[..matched_len + remaining.len()], state.clone());
//! ```

use std::collections::HashMap;

/// Any state a runtime wants to snapshot at a prefix boundary. The
/// runtime owns the type; this crate only requires it be `Clone`.
pub trait KvSnapshot: Clone {}
impl<T: Clone> KvSnapshot for T {}

/// One node in the radix trie. We store the FULL prefix path implicitly
/// (children are indexed by next token id) plus an OPTIONAL snapshot
/// at this node — only nodes that were once a generation start point
/// hold a snapshot.
struct Node<S: KvSnapshot> {
    snapshot: Option<S>,
    children: HashMap<i32, Node<S>>,
    /// Counts how many times this node was the longest match — used
    /// by the LRU-ish eviction policy when the trie hits its cap.
    hit_count: u64,
}

impl<S: KvSnapshot> Node<S> {
    fn new() -> Self {
        Self { snapshot: None, children: HashMap::new(), hit_count: 0 }
    }

    /// Total number of nodes in this subtree (used for cap enforcement).
    fn count(&self) -> usize {
        1 + self.children.values().map(|c| c.count()).sum::<usize>()
    }
}

/// The radix trie + cap.
pub struct PrefixCache<S: KvSnapshot> {
    root: Node<S>,
    /// Soft cap on the total number of trie nodes; when exceeded,
    /// the LEAST-recently-hit branch is dropped on insert.
    max_nodes: usize,
    /// Total cache hits across all `longest_prefix` calls (telemetry).
    pub hits: u64,
    /// Total misses (top-level `longest_prefix` returning matched_len=0).
    pub misses: u64,
}

impl<S: KvSnapshot> PrefixCache<S> {
    /// Create a cache with a soft `max_nodes` ceiling. A reasonable
    /// chat-shaped value is ~256: that's about 16 active conversation
    /// turns × 16-token branch depth.
    pub fn new(max_nodes: usize) -> Self {
        Self { root: Node::new(), max_nodes, hits: 0, misses: 0 }
    }

    /// Insert a (prefix, snapshot) pair. Replaces any existing snapshot
    /// at that exact prefix. Creates intermediate nodes as needed.
    pub fn insert(&mut self, prefix: &[i32], snapshot: S) {
        if prefix.is_empty() {
            self.root.snapshot = Some(snapshot);
            return;
        }
        // Walk / create.
        let mut cur = &mut self.root;
        for &tok in prefix {
            cur = cur.children.entry(tok).or_insert_with(Node::new);
        }
        cur.snapshot = Some(snapshot);

        if self.root.count() > self.max_nodes {
            self.evict_least_hit();
        }
    }

    /// Longest prefix of `tokens` that has a snapshot. Returns
    /// `(matched_len, snapshot)` where `matched_len > 0` on hit; `None`
    /// on miss (no snapshot anywhere on the path).
    pub fn longest_prefix(&mut self, tokens: &[i32]) -> Option<(usize, S)> {
        let mut cur = &mut self.root;
        let mut best: Option<(usize, S)> = None;
        // Root snapshot is the "no prefix" baseline — we ignore it
        // (caller already has whatever they had at start).
        for (i, &tok) in tokens.iter().enumerate() {
            let Some(child) = cur.children.get_mut(&tok) else { break };
            cur = child;
            if let Some(snap) = &cur.snapshot {
                best = Some((i + 1, snap.clone()));
                cur.hit_count = cur.hit_count.saturating_add(1);
            }
        }
        match &best {
            Some(_) => self.hits = self.hits.saturating_add(1),
            None => self.misses = self.misses.saturating_add(1),
        }
        best
    }

    /// Returns the current count of nodes in the trie. Useful for
    /// monitoring eviction pressure.
    pub fn len(&self) -> usize {
        self.root.count()
    }

    pub fn is_empty(&self) -> bool {
        self.root.children.is_empty() && self.root.snapshot.is_none()
    }

    pub fn clear(&mut self) {
        self.root = Node::new();
        self.hits = 0;
        self.misses = 0;
    }

    /// Drop the subtree of the LEAST-hit immediate child of the root.
    /// Coarse but bounded — fine for chat-shaped workloads where the
    /// hit distribution is heavily skewed toward the system prompt.
    fn evict_least_hit(&mut self) {
        if self.root.children.is_empty() {
            return;
        }
        let victim: Option<i32> = self.root.children.iter()
            .min_by_key(|(_, c)| c.hit_count)
            .map(|(t, _)| *t);
        if let Some(t) = victim {
            self.root.children.remove(&t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lightweight stand-in for a runtime's KV state. Carries enough
    /// content to verify clone-correctness across cache hits.
    #[derive(Clone, Debug, PartialEq)]
    struct DummyState {
        pos: usize,
        scratch: Vec<f32>,
    }

    #[test]
    fn pc_a_empty_cache_misses() {
        let mut c: PrefixCache<DummyState> = PrefixCache::new(64);
        assert!(c.longest_prefix(&[1, 2, 3]).is_none());
        assert_eq!(c.misses, 1);
        assert_eq!(c.hits, 0);
    }

    #[test]
    fn pc_b_exact_match() {
        let mut c: PrefixCache<DummyState> = PrefixCache::new(64);
        let s = DummyState { pos: 5, scratch: vec![1.0, 2.0, 3.0] };
        c.insert(&[10, 20, 30, 40, 50], s.clone());
        let (n, got) = c.longest_prefix(&[10, 20, 30, 40, 50, 60, 70]).expect("hit");
        assert_eq!(n, 5);
        assert_eq!(got, s);
        assert_eq!(c.hits, 1);
    }

    #[test]
    fn pc_c_longest_prefix_among_multiple() {
        let mut c: PrefixCache<DummyState> = PrefixCache::new(64);
        let s2 = DummyState { pos: 2, scratch: vec![2.0] };
        let s4 = DummyState { pos: 4, scratch: vec![4.0] };
        c.insert(&[1, 2], s2.clone());
        c.insert(&[1, 2, 3, 4], s4.clone());
        // Query [1,2,3,4,5] → longest prefix is the depth-4 entry.
        let (n, got) = c.longest_prefix(&[1, 2, 3, 4, 5]).expect("hit");
        assert_eq!(n, 4);
        assert_eq!(got, s4);
        // Query [1,2,9] → longest is depth-2 (depth-4 path diverges at idx 2).
        let (n, got) = c.longest_prefix(&[1, 2, 9]).expect("hit");
        assert_eq!(n, 2);
        assert_eq!(got, s2);
    }

    #[test]
    fn pc_d_overlapping_branches() {
        let mut c: PrefixCache<DummyState> = PrefixCache::new(64);
        c.insert(&[1, 2, 3], DummyState { pos: 3, scratch: vec![] });
        c.insert(&[1, 2, 4], DummyState { pos: 4, scratch: vec![] });
        // [1,2,5] → no snapshot at depth 3 along that branch; we don't
        // have a snapshot at [1,2], so this is a miss.
        assert!(c.longest_prefix(&[1, 2, 5]).is_none());
        // Add a snapshot at the shared [1,2] prefix:
        c.insert(&[1, 2], DummyState { pos: 2, scratch: vec![] });
        let (n, _) = c.longest_prefix(&[1, 2, 5]).expect("hit at depth 2");
        assert_eq!(n, 2);
    }

    #[test]
    fn pc_e_eviction_caps_growth() {
        let mut c: PrefixCache<DummyState> = PrefixCache::new(8);
        // Insert 4 branches of length 5 = 20 nodes worth, well over cap.
        for branch in 0..4 {
            let prompt: Vec<i32> = (0..5).map(|i| branch * 100 + i).collect();
            c.insert(&prompt, DummyState { pos: 5, scratch: vec![] });
        }
        assert!(c.len() <= 16, "cap softly enforced, got {}", c.len());
    }
}
