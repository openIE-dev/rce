/// Cost estimate produced by `Backend::cost(plan)` and consumed by the
/// scheduler's placement pass. Energy is denominated in picojoules so the
/// figure lines up with the placement cost table.
///
/// D.11 basis (`spec/RCE-SPEC.md`): `measured` is an external wall or shunt
/// meter on a named rail; `reported` is an OS or on-chip counter (RAPL,
/// NVML, IOReport, macmon, powermetrics, SMC, hwmon); `est` is a model,
/// table, fallback or simulation. A missing figure is absent, not 0.

/// Where an energy figure came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyBasis {
    /// A model, table, fallback or simulation.
    Est,
    /// An OS or on-chip counter (RAPL, NVML, IOReport, macmon, powermetrics, SMC, hwmon).
    Reported,
    /// An external wall or shunt meter on a named rail.
    Measured,
}

impl EnergyBasis {
    pub const fn word(self) -> &'static str {
        match self {
            EnergyBasis::Est => "est",
            EnergyBasis::Reported => "reported",
            EnergyBasis::Measured => "measured",
        }
    }

    /// Byte stored in a placement transcript.
    /// `0` is `est` (the previous `measured = false`).
    /// `1` is `reported` (the previous `measured = true`, which named a counter).
    /// `2` is `measured` (a wall or shunt meter).
    pub const fn digest_code(self) -> u8 {
        match self {
            EnergyBasis::Est => 0,
            EnergyBasis::Reported => 1,
            EnergyBasis::Measured => 2,
        }
    }

    /// Weaker of two bases. `est` is weaker than `reported`, which is weaker than `measured`.
    pub const fn weaker(self, other: Self) -> Self {
        match (self, other) {
            (EnergyBasis::Est, _) | (_, EnergyBasis::Est) => EnergyBasis::Est,
            (EnergyBasis::Reported, _) | (_, EnergyBasis::Reported) => EnergyBasis::Reported,
            (EnergyBasis::Measured, EnergyBasis::Measured) => EnergyBasis::Measured,
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().trim_matches('"') {
            "est" => Some(EnergyBasis::Est),
            "reported" => Some(EnergyBasis::Reported),
            "measured" => Some(EnergyBasis::Measured),
            _ => None,
        }
    }

    /// A cost-table file from before `basis` existed stored `measured = true`
    /// for a counter or an observation. D.11 calls that `reported`.
    /// `measured = false` is `est`.
    pub const fn from_legacy_flag(measured: bool) -> Self {
        if measured { EnergyBasis::Reported } else { EnergyBasis::Est }
    }
}

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
    /// D.11 basis of `picojoules`.
    pub basis: EnergyBasis,
}

impl EnergyEstimate {
    pub const fn estimated(picojoules: f64) -> Self {
        Self { picojoules, basis: EnergyBasis::Est }
    }

    pub const fn reported(picojoules: f64) -> Self {
        Self { picojoules, basis: EnergyBasis::Reported }
    }

    /// A wall or shunt meter. A counter is [`EnergyEstimate::reported`].
    pub const fn measured(picojoules: f64) -> Self {
        Self { picojoules, basis: EnergyBasis::Measured }
    }
}
