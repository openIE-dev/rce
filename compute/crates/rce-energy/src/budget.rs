//! Energy budget verdicts. A gate may refuse a claim whose basis is
//! `reported` or `measured` when the figure is over the budget. A
//! simulated figure is not joules: `check` returns [`Verdict::Simulated`]
//! and `failed` is false for it. A missing reading is [`Verdict::NoMeasurement`],
//! not a pass and not 0.

/// D.11 basis of the figure passed to [`check`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimBasis {
    /// OS or on-chip counter.
    Reported,
    /// External wall or shunt meter.
    Measured,
    /// Model, table, fallback or simulation. Not a joule reading.
    Est,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass { j: f64, budget: f64, basis: ClaimBasis },
    Over { j: f64, budget: f64, basis: ClaimBasis },
    NoMeasurement,
    /// `est_j` is a model figure. It is not presented as joules.
    Simulated { est_j: f64 },
}

impl Verdict {
    pub fn failed(&self, strict: bool) -> bool {
        match self {
            Verdict::Over { basis: ClaimBasis::Reported | ClaimBasis::Measured, .. } => true,
            Verdict::Over { basis: ClaimBasis::Est, .. } | Verdict::Simulated { .. } | Verdict::Pass { .. } => false,
            Verdict::NoMeasurement => strict,
        }
    }
}

/// `j` of `None` is absent. `ClaimBasis::Est` is labelled simulated and
/// does not fail as joules, whether or not it is over `budget`.
pub fn check(j: Option<f64>, budget: f64, basis: ClaimBasis) -> Verdict {
    match (j, basis) {
        (None, _) => Verdict::NoMeasurement,
        (Some(est_j), ClaimBasis::Est) => Verdict::Simulated { est_j },
        (Some(j), basis) if j > budget => Verdict::Over { j, budget, basis },
        (Some(j), basis) => Verdict::Pass { j, budget, basis },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verdicts() {
        assert!(!check(Some(1.0), 2.0, ClaimBasis::Reported).failed(true));
        assert!(check(Some(3.0), 2.0, ClaimBasis::Measured).failed(false));
        assert!(!check(None, 2.0, ClaimBasis::Reported).failed(false));
        assert!(check(None, 2.0, ClaimBasis::Reported).failed(true));
        let sim = check(Some(3.0), 2.0, ClaimBasis::Est);
        assert!(!sim.failed(true));
        assert_eq!(sim, Verdict::Simulated { est_j: 3.0 });
    }
}
