//! Verifiable dispatch trace — mesh-core's `Command` + `transcript_digest`
//! pattern applied to RCE's per-call backend selection.
//!
//! Each `PlacementQmmAdapter::qmm_1d` call appends one `DispatchEvent` to a
//! `DispatchTranscript`. The transcript folds into an FNV-1a digest that
//! is **a pure function of the dispatch sequence**: two honest replicas
//! running the same model with the same cost table produce the same
//! digest. A replica that lies about which backend ran a dispatch — or
//! drops / reorders events — diverges on the digest and is caught.
//!
//! Why this matters: the digest is a function of the dispatch sequence,
//! so two runs with the same cost table can be compared. The figure
//! 1.865 mJ in earlier notes on this file is illustrative, not a reading.
//!
//! Determinism rules (the same discipline mesh-core enforces):
//! - Field encoding is canonical (lengths-prefixed strings, little-endian
//!   u64s, fixed-tag separators).
//! - Floats are encoded via `f64::to_bits().to_le_bytes()` so NaN and
//!   precision quirks can't perturb the digest.
//! - No wall-clock, no RNG, no I/O — only data passed through the API.
//! - Events are folded in append order; the position is implicit.

use crate::QMM_DTYPE_UNKNOWN;
use rce_placement::EnergyBasis;

/// One per `PlacementQmmAdapter::qmm_1d` call. Captures every input to
/// the placement decision plus the chosen backend's cost estimate — what
/// a future replay needs to verify the dispatch was honest.
#[derive(Clone, Debug)]
pub struct DispatchEvent {
    /// The op the scheduler dispatched. Always `"qmm"` for this adapter
    /// today; kept generic so future rce-placement-adapter variants
    /// (e.g. an attention adapter) share the event shape.
    pub op: &'static str,
    /// Activation shape — `[k]` for qmm.
    pub a_shape: Vec<usize>,
    /// Weight shape: `[n_rows, n_cols, weight_bytes, dtype_id]`. The
    /// dtype_id distinguishes Q8_0 / Q4_K / Q5_K / Q6K / unknown so a
    /// replica that swapped a Q8_0 dispatch for a Q4_K one is caught
    /// even if the (n_rows, n_cols) match.
    pub w_shape: Vec<usize>,
    /// Backend id that won `pick_priced` (e.g. `"cpu:op-qmm"`,
    /// `"metal:op-qmm"`).
    pub backend_id: String,
    /// Estimated joules-of-this-dispatch the chosen backend's cost cell
    /// reported. Two honest replicas with the same cost table produce
    /// the same number; a replica running on a different cost table
    /// diverges here.
    pub estimated_picojoules: f64,
    /// D.11 basis of `estimated_picojoules`. Flows from `CostTable.lookup`.
    /// The digest byte is [`EnergyBasis::digest_code`]: 0 est, 1 reported
    /// (the previous `measured = true`), 2 measured.
    pub basis: EnergyBasis,
}

/// Ordered log of dispatch events + a folded digest. Single-source-of-
/// truth for "what dispatches happened on this adapter."
#[derive(Default, Clone, Debug)]
pub struct DispatchTranscript {
    events: Vec<DispatchEvent>,
}

impl DispatchTranscript {
    pub fn new() -> Self { Self::default() }

    /// Append a new event. Returns its position in the log (the seq id).
    pub fn append(&mut self, ev: DispatchEvent) -> usize {
        self.events.push(ev);
        self.events.len() - 1
    }

    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }
    pub fn events(&self) -> &[DispatchEvent] { &self.events }

    /// FNV-1a fingerprint of the full transcript. Mirrors mesh-core's
    /// `transcript_digest`: a deterministic fold over the events,
    /// reproducible across all honest replicas.
    pub fn digest(&self) -> u64 {
        let mut d = DigestBuf::new();
        for (i, ev) in self.events.iter().enumerate() {
            d.u64(i as u64);
            encode_event(&mut d, ev);
        }
        d.finish()
    }
}

// ── Encoding helpers (mesh-core's `Digest` builder, ported) ────────

const TAG_OP:        u8 = 0x01;
const TAG_A_SHAPE:   u8 = 0x02;
const TAG_W_SHAPE:   u8 = 0x03;
const TAG_BACKEND:   u8 = 0x04;
const TAG_EST_PJ:    u8 = 0x05;
const TAG_MEASURED:  u8 = 0x06;

fn encode_event(d: &mut DigestBuf, ev: &DispatchEvent) {
    d.tag(TAG_OP);
    d.str(ev.op);
    d.tag(TAG_A_SHAPE);
    d.u64(ev.a_shape.len() as u64);
    for x in &ev.a_shape { d.u64(*x as u64); }
    d.tag(TAG_W_SHAPE);
    d.u64(ev.w_shape.len() as u64);
    for x in &ev.w_shape { d.u64(*x as u64); }
    d.tag(TAG_BACKEND);
    d.str(&ev.backend_id);
    d.tag(TAG_EST_PJ);
    d.u64(ev.estimated_picojoules.to_bits());
    d.tag(TAG_MEASURED);
    d.u8(ev.basis.digest_code());
}

/// FNV-1a fold over a canonical byte buffer. Hand-rolled rather than
/// using `std`'s default hasher (which has a randomized seed) so the
/// digest is a function of bytes alone — mesh-core's same rationale.
struct DigestBuf { buf: Vec<u8> }

impl DigestBuf {
    fn new() -> Self { Self { buf: Vec::new() } }
    fn tag(&mut self, t: u8) { self.buf.push(t); }
    fn u8(&mut self, v: u8)   { self.buf.push(v); }
    fn u64(&mut self, v: u64) { self.buf.extend_from_slice(&v.to_le_bytes()); }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn finish(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &b in &self.buf {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

/// Convenience: build a `DispatchEvent` from the data the adapter has at
/// dispatch time. Kept as a free function so it stays trivially testable
/// without spinning up the full adapter.
pub fn make_event(
    op: &'static str,
    a_shape: &[usize],
    w_shape: &[usize],
    backend_id: &str,
    estimated_picojoules: f64,
    basis: EnergyBasis,
) -> DispatchEvent {
    DispatchEvent {
        op,
        a_shape: a_shape.to_vec(),
        w_shape: w_shape.to_vec(),
        backend_id: backend_id.to_string(),
        estimated_picojoules,
        basis,
    }
}

/// Sanity helper for callers wanting to assert "I priced a known-format
/// op." Returns true iff `w_shape`'s dtype slot is something other than
/// `QMM_DTYPE_UNKNOWN`.
pub fn w_shape_has_known_dtype(w_shape: &[usize]) -> bool {
    w_shape.get(3).copied().unwrap_or(QMM_DTYPE_UNKNOWN) != QMM_DTYPE_UNKNOWN
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(op: &'static str, a: &[usize], w: &[usize], be: &str, pj: f64, basis: EnergyBasis) -> DispatchEvent {
        make_event(op, a, w, be, pj, basis)
    }

    #[test]
    fn empty_transcript_has_a_stable_digest() {
        let t1 = DispatchTranscript::new();
        let t2 = DispatchTranscript::new();
        assert_eq!(t1.digest(), t2.digest());
    }

    #[test]
    fn honest_replicas_with_same_sequence_agree() {
        // Two transcripts built by appending the same events in the
        // same order should produce bit-identical digests — the
        // "honest replicas agree" property mesh-core asserts.
        let build = || {
            let mut t = DispatchTranscript::new();
            t.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est));
            t.append(ev("qmm", &[2048], &[5632, 2048, 64, 1], "metal:op-qmm", 5.0e5, EnergyBasis::Est));
            t
        };
        let a = build();
        let b = build();
        assert_eq!(a.digest(), b.digest());
        assert_eq!(a.len(), 2);
    }

    #[test]
    fn byzantine_backend_swap_is_caught() {
        // A replica that ran the same dispatch but lies about which
        // backend it routed to diverges on the digest.
        let mut honest = DispatchTranscript::new();
        honest.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "metal:op-qmm", 1.0e6, EnergyBasis::Est));

        let mut byz = DispatchTranscript::new();
        byz.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est));

        assert_ne!(honest.digest(), byz.digest(),
            "byzantine backend-swap should diverge on the trace digest");
    }

    #[test]
    fn dropped_event_is_caught() {
        // A replica that drops one dispatch event from the log
        // diverges from the full log's digest.
        let mut full = DispatchTranscript::new();
        full.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est));
        full.append(ev("qmm", &[2048], &[5632, 2048, 64, 1], "metal:op-qmm", 5.0e5, EnergyBasis::Est));

        let mut dropped = DispatchTranscript::new();
        dropped.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est));

        assert_ne!(full.digest(), dropped.digest(),
            "byzantine drop should diverge on the trace digest");
    }

    #[test]
    fn cost_drift_is_caught() {
        // A replica running on a cost table whose coefficients differ
        // computes a different estimated_picojoules per dispatch and
        // diverges — exactly the "trace catches calibration drift"
        // property the mesh-core controller pattern needs.
        let mut a = DispatchTranscript::new();
        a.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est));
        let mut b = DispatchTranscript::new();
        b.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.1e6, EnergyBasis::Est));
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn basis_is_load_bearing_in_digest() {
        // The basis byte distinguishes an analytical prior (`est`, code 0)
        // from a counter (`reported`, code 1, the previous true flag).
        // Two replicas, same dispatch, different bases, different digests.
        let mut prior = DispatchTranscript::new();
        prior.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est));
        let mut measured = DispatchTranscript::new();
        measured.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Reported));
        assert_ne!(prior.digest(), measured.digest());
    }

    #[test]
    fn order_matters() {
        // Two transcripts holding the same multiset of events in
        // different orders produce different digests — the bound
        // fraction of the trace.
        let e1 = ev("qmm", &[2048], &[2048, 2048, 32, 1], "cpu:op-qmm", 1.0e6, EnergyBasis::Est);
        let e2 = ev("qmm", &[2048], &[5632, 2048, 64, 1], "metal:op-qmm", 5.0e5, EnergyBasis::Est);

        let mut ab = DispatchTranscript::new();
        ab.append(e1.clone());
        ab.append(e2.clone());

        let mut ba = DispatchTranscript::new();
        ba.append(e2);
        ba.append(e1);

        assert_ne!(ab.digest(), ba.digest());
    }

    #[test]
    fn dtype_swap_is_caught() {
        // Q8_0 → Q4_K swap with the same (n_rows, n_cols) — a real
        // byzantine attack vector since the n_rows/n_cols match. The
        // dtype_id slot in w_shape catches it.
        let mut q8  = DispatchTranscript::new();
        q8.append(ev("qmm", &[2048], &[2048, 2048, 32, 1], "metal:op-qmm", 5.0e5, EnergyBasis::Est));
        let mut q4  = DispatchTranscript::new();
        q4.append(ev("qmm", &[2048], &[2048, 2048, 32, 2], "metal:op-qmm", 5.0e5, EnergyBasis::Est));
        assert_ne!(q8.digest(), q4.digest());
    }
}
