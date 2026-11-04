//! Energy budgets that fail a test run or CI job (lesson from CodeGreen's
//! budget gate). A missing measurement is its own outcome, not a pass.

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass { joules: f64, budget: f64 },
    Over { joules: f64, budget: f64 },
    NoMeasurement,
}

impl Verdict {
    pub fn failed(&self, strict: bool) -> bool {
        matches!(self, Verdict::Over { .. }) || (strict && *self == Verdict::NoMeasurement)
    }
}

pub fn check(joules: Option<f64>, budget: f64) -> Verdict {
    match joules {
        None => Verdict::NoMeasurement,
        Some(j) if j > budget => Verdict::Over { joules: j, budget },
        Some(j) => Verdict::Pass { joules: j, budget },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verdicts() {
        assert!(!check(Some(1.0), 2.0).failed(true));
        assert!(check(Some(3.0), 2.0).failed(false));
        assert!(!check(None, 2.0).failed(false));
        assert!(check(None, 2.0).failed(true));
    }
}
