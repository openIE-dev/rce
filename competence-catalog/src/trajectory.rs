//! Robot-trajectory ingestion — provenance at scale (the 1M-vs-350k dispute).
//!
//! A dataset-size claim is answered by distinct admitted content keys, not by a
//! press release: episodes are content-addressed (ticks only, source kept as
//! provenance), deduped across sources, and malformed episodes refused. Admission
//! here is integrity + dedup; physics replay of a trajectory stays the open twin edge.

use rcevc_core::sha256;

/// One control tick: observation and action.
#[derive(Clone, Debug, PartialEq)]
pub struct Tick {
    pub obs: f64,
    pub act: f64,
}

/// An episode from a named source (teleop rig, sim farm, fleet robot).
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub source: String,
    pub ticks: Vec<Tick>,
}

/// The name-free content key: canonical encoding of the ticks, source excluded.
pub fn content_key(t: &Trajectory) -> String {
    let body: String = t
        .ticks
        .iter()
        .map(|k| format!("{:.17e},{:.17e}", k.obs, k.act))
        .collect::<Vec<_>>()
        .join(";");
    sha256(body.as_bytes()).hex()
}

/// The ingestion receipt.
#[derive(Clone, Debug)]
pub struct IngestReport {
    pub offered: usize,
    pub admitted: usize,
    pub dedup: usize,
    pub refused: usize,
    pub keys: Vec<String>,
}

/// Ingest episodes: refuse empty/non-finite, dedup by content key across sources.
pub fn ingest(eps: &[Trajectory]) -> IngestReport {
    let mut seen = std::collections::BTreeSet::new();
    let mut keys = Vec::new();
    let (mut admitted, mut dedup, mut refused) = (0, 0, 0);
    for e in eps {
        if e.ticks.is_empty() || e.ticks.iter().any(|t| !t.obs.is_finite() || !t.act.is_finite()) {
            refused += 1;
            continue;
        }
        let k = content_key(e);
        if seen.insert(k.clone()) {
            admitted += 1;
            keys.push(k);
        } else {
            dedup += 1;
        }
    }
    IngestReport { offered: eps.len(), admitted, dedup, refused, keys }
}

/// The provenance answer to a dataset-size claim: (distinct admitted, unbacked claim).
pub fn dispute(claim: usize, r: &IngestReport) -> (usize, usize) {
    (r.admitted, claim.saturating_sub(r.admitted))
}
