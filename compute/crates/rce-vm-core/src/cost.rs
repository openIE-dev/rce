/// Cost estimate produced by `Backend::cost(plan)` and consumed by the
/// scheduler's placement pass. Energy is denominated in picojoules to match
/// the existing `inv-ai-codegraph::rce::energy_estimator` convention.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    pub energy: EnergyEstimate,
    /// Wall-clock time estimate in nanoseconds.
    pub time_ns: u64,
    /// Peak memory in bytes.
    pub peak_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyEstimate {
    pub picojoules: f64,
    /// `true` if the estimate is derived from a measurement on this hardware
    /// (e.g. NVML, IOReport, RAPL); `false` if from the static estimator.
    pub measured: bool,
}

impl EnergyEstimate {
    pub const fn estimated(picojoules: f64) -> Self {
        Self {
            picojoules,
            measured: false,
        }
    }

    pub const fn measured(picojoules: f64) -> Self {
        Self {
            picojoules,
            measured: true,
        }
    }
}
