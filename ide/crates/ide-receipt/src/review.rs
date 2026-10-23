//! Reviewed-by-human marks that expire when the reviewed code changes.
//! A mark pins the BLAKE3 hash of the item's text at review time and is
//! recorded as a signed receipt (`Other("review")`), so the review is on
//! the same hash chain as every agent action. Any change to the item makes
//! the mark `Expired`; nothing is silently carried over.
use crate::chain::LedgerError;
use crate::{ActionKind, Draft, Ledger, Receipt, Verdict};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Mark {
    pub node: String,
    pub reviewer: String,
    pub item_hash: String,
    pub receipt: String,
    pub t_ns: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status { Valid, Expired { was: String, now: String } }

pub fn item_hash(text: &str) -> String {
    // Normalise trailing whitespace so a reformat of blank space alone
    // does not expire a review; any other change does.
    let t: Vec<&str> = text.lines().map(str::trim_end).collect();
    blake3::hash(t.join("\n").trim_end().as_bytes()).to_hex().to_string()
}

pub fn mark(ledger: &mut Ledger, node: &str, text: &str, reviewer: &str, policy_hash: &str, t_ns: u64)
    -> Result<(Mark, Receipt), LedgerError> {
    let h = item_hash(text);
    let mut d = Draft::new(ActionKind::Other("review".into()), "human", policy_hash);
    d.t_start_ns = t_ns;
    d.t_end_ns = t_ns;
    d.verdict = Verdict::Pass;
    d.detail = serde_json::json!({"node": node, "reviewer": reviewer, "item_hash": h});
    let r = ledger.append(d)?;
    Ok((Mark { node: node.into(), reviewer: reviewer.into(), item_hash: h, receipt: r.hash.clone(), t_ns }, r))
}

pub fn status(m: &Mark, current_text: &str) -> Status {
    let now = item_hash(current_text);
    if now == m.item_hash { Status::Valid } else { Status::Expired { was: m.item_hash.clone(), now } }
}

/// A mark is trustworthy only if its receipt is in the chain and names the
/// same node and hash.
pub fn backed(m: &Mark, receipts: &[Receipt]) -> bool {
    receipts.iter().any(|r| r.hash == m.receipt
        && r.body.draft.detail["item_hash"] == m.item_hash.as_str()
        && r.body.draft.detail["node"] == m.node.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigningKey;
    #[test]
    fn expires_on_change_and_is_receipted() {
        let mut l = Ledger::new(SigningKey::from_bytes(&[7; 32]));
        let src = "fn f() -> i64\n    return 1\n";
        let (m, r) = mark(&mut l, "code:a.rce:f", src, "david", "p", 1).unwrap();
        assert_eq!(status(&m, src), Status::Valid);
        assert_eq!(status(&m, "fn f() -> i64\n    return 1   \n\n"), Status::Valid);
        assert!(matches!(status(&m, "fn f() -> i64\n    return 2\n"), Status::Expired { .. }));
        assert!(backed(&m, std::slice::from_ref(&r)));
        let mut forged = m.clone();
        forged.item_hash = item_hash("other");
        assert!(!backed(&forged, &[r]));
    }
}
