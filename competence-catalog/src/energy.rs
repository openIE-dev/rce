//! Measured joules for competence execution — D.11 provenance made real.
//!
//! The catalog's cost figures are `Modelled` (the ≈1 µJ G0 interpreter floor)
//! unless a hardware meter backs them. This opens the platform meter (macOS
//! IOReport package rails, no root; Linux RAPL) and measures the energy of an
//! execution window as the delta of the cumulative counter. A source that
//! cannot measure errors — it never returns a fabricated 0. The reading is
//! whole-package: concurrent work inside the window is included, which is the
//! honest caveat a `Measured` figure carries against the `Modelled` floor.

use rce_energy::meter::{Meter, MeterError};

/// A hardware energy meter for the catalog.
pub struct EnergyMeter {
    meter: Box<dyn Meter>,
}

impl EnergyMeter {
    /// Open the platform meter, or `Err` if none can measure here.
    pub fn open() -> Result<Self, MeterError> {
        #[cfg(target_os = "macos")]
        let meter: Box<dyn Meter> = Box::new(rce_energy::ioreport::IoReportPackageMeter::open()?);
        #[cfg(target_os = "linux")]
        let meter: Box<dyn Meter> = Box::new(rce_energy::meter::RaplMeter::open()?);
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        return Err(MeterError::Unsupported("energy", "no meter on this platform".into()));
        Ok(EnergyMeter { meter })
    }

    /// The meter source tag (names the counter in the receipt's provenance).
    pub fn source(&self) -> &'static str {
        self.meter.source()
    }

    /// Microjoules drawn while running `f`, from the cumulative-counter delta.
    pub fn measure<F: FnOnce()>(&mut self, f: F) -> Result<u64, MeterError> {
        let before = self.meter.sample()?;
        f();
        let after = self.meter.sample()?;
        let dj = (after.joules - before.joules).max(0.0);
        Ok((dj * 1e6) as u64)
    }

    /// Measure `n` repetitions of `f` and return the mean microjoules per act.
    /// A single interpreter-floor act is below the meter's resolution; averaging
    /// a batch gives a stable `Measured` cost per competent act.
    pub fn measure_mean<F: FnMut()>(&mut self, n: usize, mut f: F) -> Result<u64, MeterError> {
        if n == 0 {
            return Err(MeterError::Read("energy", "n must be positive".into()));
        }
        let before = self.meter.sample()?;
        for _ in 0..n {
            f();
        }
        let after = self.meter.sample()?;
        let dj = (after.joules - before.joules).max(0.0);
        Ok((dj * 1e6 / n as f64) as u64)
    }
}
