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
    /// Episodes refused because their trace does not replay against the plant twin.
    pub refused_twin: usize,
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
    IngestReport { offered: eps.len(), admitted, dedup, refused, refused_twin: 0, keys }
}

/// Gated ingestion: an episode is admitted only if it is integral, distinct, AND its
/// recorded trace replays against the plant twin within tol (sim2real admission).
pub fn ingest_gated(eps: &[Trajectory], plant: &crate::twin::Plant, tol: f64) -> IngestReport {
    let mut seen = std::collections::BTreeSet::new();
    let mut keys = Vec::new();
    let (mut admitted, mut dedup, mut refused, mut refused_twin) = (0, 0, 0, 0);
    for e in eps {
        if e.ticks.is_empty() || e.ticks.iter().any(|t| !t.obs.is_finite() || !t.act.is_finite()) {
            refused += 1;
            continue;
        }
        let acts: Vec<f64> = e.ticks.iter().map(|t| t.act).collect();
        let recorded: Vec<f64> = e.ticks.iter().map(|t| t.obs).collect();
        if !crate::twin::admit_controller(plant, &acts, &recorded, tol) {
            refused_twin += 1;
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
    IngestReport { offered: eps.len(), admitted, dedup, refused, refused_twin, keys }
}

/// The provenance answer to a dataset-size claim: (distinct admitted, unbacked claim).
pub fn dispute(claim: usize, r: &IngestReport) -> (usize, usize) {
    (r.admitted, claim.saturating_sub(r.admitted))
}

/// Standards-style episode manifest (India consortium / AAIF interop seed): name-free
/// content keys, counts, and refusal reasons — the receipt a dataset standard can verify.
pub fn manifest(r: &IngestReport) -> serde_json::Value {
    serde_json::json!({
        "schema": "rce-episode-manifest/1",
        "offered": r.offered,
        "admitted": r.admitted,
        "dedup": r.dedup,
        "refused": r.refused,
        "refused_twin": r.refused_twin,
        "keys": r.keys,
    })
}
