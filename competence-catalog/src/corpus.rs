//! Corpus ingestion — the world's repos made ingestable.
//!
//! The bootstrap-corpus pipeline: take a code snippet from a searchable repo,
//! **lift** it into the canonical Form IR, **repair** it if it fails its spec,
//! and content-address it as competence ready for the gate. This is the import
//! half of the competence economy — humanity's verified code mined into the
//! substrate, then owned, reused, and circulated.
//!
//! Ingest → lift → (repair) → content key. The key is name-free (canonical
//! behavior), so the same competence from any repo/language deduplicates. The
//! admission gate (replay) is applied downstream; ingestion prepares the record.

use crate::canonical::{emit, normalize};
use crate::lift::lift_expr;
use crate::repair::repair_to_value;

/// The result of ingesting a code snippet.
pub struct IngestReport {
    /// The original source.
    pub original: String,
    /// The lifted canonical Form source.
    pub form: String,
    /// The repaired source, if a defect was corrected.
    pub repaired: Option<String>,
    /// The name-free content key of the (corrected) competence.
    pub key: String,
}

/// Ingest a snippet: lift to Form, repair to a target spec if given and defective,
/// and content-address the result. The final form is what the gate would admit.
pub fn ingest(source: &str, target: Option<i128>) -> Result<IngestReport, String> {
    // Lift, then normalize to canonical form so the content key is name-free and
    // order-independent (a+b and b+a are one competence).
    let form = emit(&normalize(lift_expr(source)?));
    let mut repaired = None;
    let final_form = match target {
        Some(t) => {
            if let Some(r) = repair_to_value(source, t) {
                repaired = Some(r.repaired.clone());
                r.repaired
            } else {
                form.clone()
            }
        }
        None => form.clone(),
    };
    let key = rcevc_core::sha256(final_form.as_bytes()).hex();
    Ok(IngestReport {
        original: source.to_string(),
        form,
        repaired,
        key,
    })
}

/// Ingest a batch (a repo's snippets), deduplicating by content key — the same
/// competence appearing in many repos collapses to one entry (the clone economy).
pub fn ingest_batch(snippets: &[(&str, Option<i128>)]) -> Vec<IngestReport> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (src, target) in snippets {
        if let Ok(report) = ingest(src, *target) {
            if seen.insert(report.key.clone()) {
                out.push(report);
            }
        }
    }
    out
}
