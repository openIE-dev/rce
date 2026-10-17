//! Caretakers and revocation (spec F.5.1, W-CAP-4 and W-CAP-5), as a function
//! of the log. A revoke is an entry of the log like any other, so whether a use
//! is before or after it is the log order, and replay is exact (T3).
//!
//! This is the semantic model the reference checker implements. The language
//! surface that would write `host.revoke(id)` needs the host log, which the
//! reference checker does not have, so that call is refused by version
//! (`v09::gate`). Nothing here proves T51; it is stated and unproved.
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Entry {
    /// the granter hands a capability to another component as a caretaker
    Grant { id: u64 },
    /// a component uses the capability behind a caretaker
    Use { id: u64 },
    /// `host.revoke(id)`
    Revoke { id: u64 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// the entry is admitted
    Admitted,
    /// a revoke of a caretaker already revoked: a no-op entry (still admitted)
    NoOp,
    /// W-CAP-5: the use is refused with the fixed record
    Refused(String),
}

/// The fixed record of a use of a revoked caretaker (canonical JSON, 6.1).
pub fn refusal_record(id: u64) -> String {
    format!("{{\"id\":{id},\"kind\":\"capability\",\"reason\":\"revoked\"}}")
}

/// The verdict of every entry of a log, in log order.
pub fn admit(log: &[Entry]) -> Vec<Verdict> {
    let mut granted = BTreeSet::new();
    let mut revoked = BTreeSet::new();
    let mut out = vec![];
    for e in log {
        out.push(match e {
            Entry::Grant { id } => {
                // an id is granted once; a granted id is never valid again once revoked
                granted.insert(*id);
                Verdict::Admitted
            }
            Entry::Use { id } => {
                if revoked.contains(id) || !granted.contains(id) {
                    Verdict::Refused(refusal_record(*id))
                } else {
                    Verdict::Admitted
                }
            }
            Entry::Revoke { id } => {
                if revoked.insert(*id) { Verdict::Admitted } else { Verdict::NoOp }
            }
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use Entry::*;

    #[test]
    fn use_before_revoke_is_admitted_and_after_is_refused() {
        let v = admit(&[Grant { id: 1 }, Use { id: 1 }, Revoke { id: 1 }, Use { id: 1 }]);
        assert_eq!(v[1], Verdict::Admitted);
        assert_eq!(v[3], Verdict::Refused("{\"id\":1,\"kind\":\"capability\",\"reason\":\"revoked\"}".into()));
    }

    #[test]
    fn revoking_twice_is_a_no_op_and_there_is_no_unrevoke() {
        let v = admit(&[Grant { id: 7 }, Revoke { id: 7 }, Revoke { id: 7 }, Grant { id: 7 }, Use { id: 7 }]);
        assert_eq!(v[1], Verdict::Admitted);
        assert_eq!(v[2], Verdict::NoOp);
        assert!(matches!(v[4], Verdict::Refused(_)), "a re-grant of a revoked id does not revive it");
    }

    #[test]
    fn idempotent_revoke_leaves_every_other_verdict_unchanged() {
        let a = admit(&[Grant { id: 1 }, Grant { id: 2 }, Revoke { id: 1 }, Use { id: 2 }, Use { id: 1 }]);
        let b = admit(&[Grant { id: 1 }, Grant { id: 2 }, Revoke { id: 1 }, Revoke { id: 1 }, Use { id: 2 }, Use { id: 1 }]);
        let strip = |v: Vec<Verdict>| v.into_iter().filter(|x| *x != Verdict::NoOp).collect::<Vec<_>>();
        assert_eq!(strip(a), strip(b));
    }

    #[test]
    fn a_use_of_an_ungranted_caretaker_is_refused_and_replay_is_exact() {
        let log = [Use { id: 9 }, Grant { id: 9 }, Use { id: 9 }];
        let v = admit(&log);
        assert!(matches!(v[0], Verdict::Refused(_)));
        assert_eq!(v[2], Verdict::Admitted);
        assert_eq!(admit(&log), v);
    }
}
