//! `rce-federation` — distributed RCE worker primitives.
//!
//! Port of the load-bearing pieces of mesh-core's design for a federated
//! placement layer: each worker runs its own `PlacementQmmAdapter` and
//! reports its dispatch-transcript digest (the verifiable energy
//! receipt from `rce-placement-adapter::trace`); the federation
//! detects byzantine workers by digest disagreement.
//!
//! Three modules, each one a tight transplant of a mesh-core idea:
//!
//! 1. **`Membership` (OR-Set)** — port of mesh-core's monotone-admit /
//!    observed-remove-evict membership lattice. Add-wins semantics on
//!    re-admit after eviction. Coordination-free under any partition;
//!    converges by `join()`.
//! 2. **`Authenticator` trait + `TokenAuth` stand-in** — port of
//!    mesh-core's admission gate. Real deployments swap in
//!    certificate-based auth; the federation only needs yes/no.
//! 3. **`DigestRegistry`** — RCE's addition. Each worker's
//!    last-known transcript digest is recorded; `consensus()` returns
//!    `Some(d)` iff every live member reports `d`, `None` if any
//!    disagree. Byzantine workers (whose digest diverges from the
//!    honest majority) are caught by exactly the same trust primitive
//!    `rce-placement-adapter::trace::DispatchTranscript::digest`
//!    enables on a single worker — re-execute, compare.
//!
//! The intentional non-goal: this crate **does not** ship a gossip
//! transport, sequencer, or ledger. Those are mesh-core's full sim
//! responsibilities; the federation primitive sits one layer down,
//! providing the trust primitives that any transport (in-process,
//! gRPC, WebSockets, file drop) can use.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

// ── 1. Authenticator (admission gate) ────────────────────────────

/// Gate on who may be admitted. A real implementation verifies a
/// signature or certificate; the federation only needs the yes/no.
pub trait Authenticator {
    fn verify(&self, worker: u64, credential: u64) -> bool;
}

/// Stand-in authenticator — port of mesh-core's `TokenAuth`. A
/// worker's valid credential is a fixed function of its id and a
/// shared secret. Swap for real asymmetric crypto in deployment.
pub struct TokenAuth {
    pub secret: u64,
}

impl TokenAuth {
    pub fn credential_for(&self, worker: u64) -> u64 {
        worker.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ self.secret
    }
}

impl Authenticator for TokenAuth {
    fn verify(&self, worker: u64, credential: u64) -> bool {
        credential == self.credential_for(worker)
    }
}

// ── 2. Membership (OR-Set) ──────────────────────────────────────

/// Port of mesh-core's `Membership` OR-Set, specialised to worker
/// ids. Same semantics: monotone `admit` (coordination-free), observed-
/// remove `evict` (re-admit with a fresh tag wins).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Membership {
    adds: BTreeMap<u64, BTreeSet<u64>>, // worker -> live + dead add-tags
    removes: BTreeSet<u64>,             // tombstoned tags
}

impl Membership {
    pub fn new() -> Self { Self::default() }

    /// Monotone, coordination-free: record an admission with a unique tag.
    pub fn admit(&mut self, worker: u64, tag: u64) {
        self.adds.entry(worker).or_default().insert(tag);
    }

    /// Authenticated admission — returns false (no state change) on bad
    /// credential. Same shape as mesh-core's
    /// `admit_authenticated`.
    pub fn admit_authenticated<A: Authenticator>(
        &mut self,
        auth: &A,
        worker: u64,
        credential: u64,
        tag: u64,
    ) -> bool {
        if auth.verify(worker, credential) {
            self.admit(worker, tag);
            true
        } else {
            false
        }
    }

    /// Non-monotone: tombstone every add-tag currently observed for
    /// `worker`. Concurrent re-admission with a fresh tag rejoins
    /// (add-wins OR-Set).
    pub fn evict(&mut self, worker: u64) {
        if let Some(tags) = self.adds.get(&worker) {
            for &t in tags { self.removes.insert(t); }
        }
    }

    pub fn is_member(&self, worker: u64) -> bool {
        self.adds.get(&worker)
            .map(|tags| tags.iter().any(|t| !self.removes.contains(t)))
            .unwrap_or(false)
    }

    pub fn members(&self) -> Vec<u64> {
        self.adds.iter()
            .filter(|(_, tags)| tags.iter().any(|t| !self.removes.contains(t)))
            .map(|(w, _)| *w)
            .collect()
    }

    pub fn count(&self) -> usize { self.members().len() }

    /// Lattice join — converges with any other replica's view under any
    /// delivery order / partition. The CRDT operation mesh-core's whole
    /// design rests on.
    pub fn join(&mut self, other: &Self) {
        for (w, tags) in &other.adds {
            let e = self.adds.entry(*w).or_default();
            for &t in tags { e.insert(t); }
        }
        for &t in &other.removes { self.removes.insert(t); }
    }
}

// ── 3. DigestRegistry (cross-replica transcript agreement) ──────

/// Per-worker latest transcript digest + consensus detection. Each
/// honest worker running the same model + cost table + dispatch cache
/// produces the same `DispatchTranscript::digest`; a byzantine worker
/// (different backend, dropped event, cost-drift, dtype lie) produces
/// a different digest. `consensus()` reports the agreed value when
/// every live member agrees, or `None` (with `disagreements()` for
/// the divergence detail) when at least one diverges.
///
/// This is the federation-scale extension of the single-worker trust
/// primitive in `rce-placement-adapter::trace::DispatchTranscript`:
/// the transcript digest is what makes a single worker's energy
/// receipt verifiable; comparing digests across workers extends the
/// same verifiability to the whole fleet.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct DigestRegistry {
    digests: HashMap<u64, u64>, // worker_id → last reported transcript digest
}

impl DigestRegistry {
    pub fn new() -> Self { Self::default() }

    /// Worker reports its current transcript digest. Overwrites any
    /// prior value for that worker — the registry tracks the **latest**
    /// digest, not the full history.
    pub fn report(&mut self, worker: u64, digest: u64) {
        self.digests.insert(worker, digest);
    }

    /// `Some(d)` iff every member of `live` has reported the same
    /// digest `d`; `None` otherwise. `live` is typically
    /// `membership.members()` — the federation cross-checks workers
    /// that are currently part of the fleet.
    pub fn consensus(&self, live: &[u64]) -> Option<u64> {
        let mut iter = live.iter()
            .filter_map(|w| self.digests.get(w).copied());
        let first = iter.next()?;
        if iter.all(|d| d == first) { Some(first) } else { None }
    }

    /// Workers whose reported digest disagrees with the modal value
    /// among `live`. Empty when there's consensus or when no worker
    /// has reported. Useful for telemetry / alerting on byzantines.
    pub fn disagreements(&self, live: &[u64]) -> Vec<(u64, u64)> {
        let mut counts: HashMap<u64, usize> = HashMap::new();
        let reported: Vec<(u64, u64)> = live.iter()
            .filter_map(|w| self.digests.get(w).map(|d| (*w, *d))).collect();
        for (_, d) in &reported { *counts.entry(*d).or_default() += 1; }
        let modal = counts.iter().max_by_key(|(_, c)| **c).map(|(d, _)| *d);
        match modal {
            None => Vec::new(),
            Some(m) => reported.into_iter().filter(|(_, d)| *d != m).collect(),
        }
    }

    /// Latest digest reported by `worker`, if any.
    pub fn get(&self, worker: u64) -> Option<u64> {
        self.digests.get(&worker).copied()
    }

    pub fn len(&self) -> usize { self.digests.len() }
    pub fn is_empty(&self) -> bool { self.digests.is_empty() }
}

// ── 4. Federation (the three together) ─────────────────────────

/// Convenience aggregate — a `Membership` + `DigestRegistry` paired so
/// callers reach for one struct. Each worker's `report_digest` is gated
/// by membership: non-members are silently ignored. Eviction
/// preserves the worker's last-known digest (matches mesh-core's
/// "evicted node's data persists in the fabric" property — useful for
/// post-incident audit).
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
pub struct Federation {
    pub members: Membership,
    pub digests: DigestRegistry,
}

impl Federation {
    pub fn new() -> Self { Self::default() }

    /// Worker reports a digest. Ignored if the worker isn't a current
    /// member — prevents non-admitted workers from polluting the
    /// consensus view.
    pub fn report(&mut self, worker: u64, digest: u64) -> bool {
        if !self.members.is_member(worker) { return false; }
        self.digests.report(worker, digest);
        true
    }

    /// `Some(d)` iff every live member that has reported a digest
    /// agrees on `d`. Same shape as mesh-core's `members_consensus`,
    /// but on transcript digests rather than membership views.
    pub fn consensus(&self) -> Option<u64> {
        self.digests.consensus(&self.members.members())
    }

    pub fn disagreements(&self) -> Vec<(u64, u64)> {
        self.digests.disagreements(&self.members.members())
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Membership tests (mirror of mesh-core's) ──

    #[test]
    fn admit_is_monotone_and_coordination_free() {
        let mut a = Membership::new();
        a.admit(1, 0x11);
        a.admit(2, 0x22);
        let mut b = Membership::new();
        b.admit(3, 0x33);
        b.admit(2, 0x22);

        let mut a_m = a.clone(); a_m.join(&b);
        let mut b_m = b.clone(); b_m.join(&a);
        assert_eq!(a_m.members(), b_m.members());
        assert_eq!(a_m.members(), vec![1, 2, 3]);
    }

    #[test]
    fn eviction_is_observed_remove_and_rejoin_wins() {
        let mut m = Membership::new();
        m.admit(5, 0xA);
        assert!(m.is_member(5));
        m.evict(5);
        assert!(!m.is_member(5));
        m.admit(5, 0xB); // fresh tag → rejoin wins
        assert!(m.is_member(5));
    }

    #[test]
    fn authentication_gates_admission() {
        let auth = TokenAuth { secret: 0xDEAD_BEEF };
        let mut m = Membership::new();
        let good = auth.credential_for(7);
        assert!(m.admit_authenticated(&auth, 7, good, 0x1));
        assert!(m.is_member(7));
        assert!(!m.admit_authenticated(&auth, 8, 0xBAD, 0x2));
        assert!(!m.is_member(8));
    }

    // ── DigestRegistry: the federation-scale trust primitive ──

    #[test]
    fn honest_workers_reach_consensus_on_their_digest() {
        let mut r = DigestRegistry::new();
        r.report(1, 0xABCD);
        r.report(2, 0xABCD);
        r.report(3, 0xABCD);
        assert_eq!(r.consensus(&[1, 2, 3]), Some(0xABCD));
        assert!(r.disagreements(&[1, 2, 3]).is_empty());
    }

    #[test]
    fn byzantine_worker_caught_by_digest_disagreement() {
        let mut r = DigestRegistry::new();
        r.report(1, 0xABCD); // honest
        r.report(2, 0xABCD); // honest
        r.report(3, 0xABCD); // honest
        r.report(4, 0xDEAD); // byzantine
        assert_eq!(r.consensus(&[1, 2, 3, 4]), None,
            "any disagreement breaks consensus");
        let bad = r.disagreements(&[1, 2, 3, 4]);
        assert_eq!(bad, vec![(4, 0xDEAD)], "byzantine identified by id+digest");
    }

    #[test]
    fn no_reports_means_no_consensus_but_also_no_disagreement() {
        let r = DigestRegistry::new();
        assert!(r.consensus(&[1, 2, 3]).is_none());
        assert!(r.disagreements(&[1, 2, 3]).is_empty());
    }

    // ── Federation (the integrated view) ──

    #[test]
    fn federation_ignores_reports_from_non_members() {
        let mut f = Federation::new();
        f.members.admit(1, 0x1);
        f.members.admit(2, 0x2);

        assert!(f.report(1, 0xABCD), "member's report accepted");
        assert!(f.report(2, 0xABCD), "member's report accepted");
        assert!(!f.report(99, 0xBAD), "non-member's report rejected");
        // Consensus excludes the non-member entirely.
        assert_eq!(f.consensus(), Some(0xABCD));
    }

    #[test]
    fn federation_consensus_under_membership_churn() {
        // The integrated property: workers join/leave, digests evolve;
        // consensus tracks the live members, not the historical set.
        let mut f = Federation::new();
        f.members.admit(1, 0x1); f.members.admit(2, 0x2);
        f.report(1, 0xAA); f.report(2, 0xAA);
        assert_eq!(f.consensus(), Some(0xAA));

        // New worker joins with a different (stale) digest — consensus
        // breaks until it catches up.
        f.members.admit(3, 0x3);
        f.report(3, 0xBB); // stale
        assert_eq!(f.consensus(), None);
        assert_eq!(f.disagreements().len(), 1);

        // Worker 3 catches up and reports the agreed digest.
        f.report(3, 0xAA);
        assert_eq!(f.consensus(), Some(0xAA));

        // Worker 1 is evicted; consensus continues among 2 + 3.
        f.members.evict(1);
        assert_eq!(f.consensus(), Some(0xAA));
        assert!(!f.members.is_member(1));
    }

    #[test]
    fn evicted_worker_digest_persists_for_audit() {
        // Mesh-core's "evicted node's data persists in the fabric"
        // applied to transcripts: after eviction we can still inspect
        // what the worker last reported. Useful for post-incident audit.
        let mut f = Federation::new();
        f.members.admit(1, 0x1);
        f.report(1, 0xABCD);
        f.members.evict(1);
        assert!(!f.members.is_member(1));
        // The digest the now-evicted worker last reported is still
        // queryable on the registry directly (not via consensus,
        // which filters by membership).
        assert_eq!(f.digests.get(1), Some(0xABCD));
    }

    #[test]
    fn membership_join_converges_across_replicas() {
        // Two federation replicas independently admit workers and
        // collect digest reports, then exchange membership via join.
        // (Digests don't have a CRDT here — they're caller-chosen
        // values; for a full design we'd lattice-fold them too. This
        // test focuses on the membership convergence.)
        let mut a = Federation::new();
        a.members.admit(1, 0x1);
        a.members.admit(2, 0x2);
        let mut b = Federation::new();
        b.members.admit(2, 0x2); // idempotent same tag
        b.members.admit(3, 0x3);

        a.members.join(&b.members);
        b.members.join(&a.members);
        assert_eq!(a.members.members(), b.members.members());
        assert_eq!(a.members.members(), vec![1, 2, 3]);
    }
}
