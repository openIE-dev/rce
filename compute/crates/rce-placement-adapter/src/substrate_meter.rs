//! Bridge: [`crate::EnergyMeter`] ↔ [`rce_energy::EnergyTelemetry`].
//!
//! `EnergyMeter` is the small start/end-token shape the placement
//! adapter brackets each dispatch with. `EnergyTelemetry` is the
//! production cross-platform measurement trait — macOS IOReport
//! (~1 mJ resolution), Linux RAPL (1 µJ), universal TDP fallback —
//! consumed by `rce-energy-xover`, `joule-pipeline`, `graph-ai`,
//! and `ask-coding-web`. This module ties them together so the
//! self-calibration loop runs against real hardware on whatever host
//! the adapter is on.
//!
//! Snapshot semantics differ per backend, and the trait docstring is
//! actually wrong for the macOS backend:
//!   - **macOS IOReport**: `snapshot()` internally calls
//!     `IOReportCreateSamplesDelta(prior, current)` and returns the
//!     **delta since the previous snapshot** (`macos_ioreport.rs`
//!     line 277-287). The Total field is the interval energy, not a
//!     running total.
//!   - **Linux RAPL**: `snapshot()` reads
//!     `/sys/class/powercap/.../energy_uj` directly — monotonically
//!     accumulating since the kernel counter was zeroed (typically
//!     boot). Total grows across snapshots.
//!   - **Fallback TDP**: `snapshot()` computes
//!     `process_cpu_time × per_core_TDP` — monotonically
//!     accumulating since the process started.
//!
//! For interval energy across a `start`/`end` window: macOS reads
//! `now.channels[Total]` directly; the cumulative backends compute
//! `now.total_uj_since(&prior)`. We switch on `backend.name()`.
//!
//! Threading: `EnergyTelemetry::snapshot` is `&mut self`, the trait
//! object is `Send` but not `Sync`. We wrap it in a `Mutex`; the
//! placement adapter's dispatch path is sequential per-call so
//! contention is on the order of "one lock per dispatch".

use std::collections::HashMap;
use std::sync::Mutex;

use rce_energy::{Channel, EnergySnapshot, EnergyTelemetry, open_default_backend};

use crate::EnergyMeter;

/// `EnergyMeter` impl that delegates to a `Box<dyn EnergyTelemetry>`.
/// On `start()` takes a snapshot, hands back a counter token, and
/// stashes the snapshot keyed by that token. On `end(token)` takes
/// another snapshot, recovers the prior by token, and reports
/// `total_uj_since` × 1_000_000 (µJ → pJ) so the picojoule integer
/// can flow straight into `CostTable::observe`.
pub struct SubstrateEnergyMeter {
    inner: Mutex<Box<dyn EnergyTelemetry>>,
    /// In-flight measurement windows. A window is opened by `start()`
    /// and closed by `end()`; cardinality at any instant is bounded by
    /// the number of concurrent dispatches the adapter has issued,
    /// which is 1 in the production single-threaded path.
    windows: Mutex<HashMap<u64, EnergySnapshot>>,
    /// Monotonic token allocator. Tokens are NOT recycled across
    /// `end()` calls — they only need uniqueness within the lifetime
    /// of one meter, and even a packed-burst dispatch loop won't
    /// exhaust 2^64 windows.
    next_token: Mutex<u64>,
}

impl SubstrateEnergyMeter {
    /// Open the platform default backend. Returns the meter unless
    /// **all three** backends fail (which `open_default_backend` never
    /// allows — `fallback_tdp` is always available), so in practice
    /// this never returns None. Kept Option-typed for symmetry with
    /// `MetalQmmContext::new` and to leave room for a future
    /// `try_open_with(backend_name)` that may legitimately fail.
    pub fn open_default() -> Option<Self> {
        let backend = open_default_backend();
        Some(Self {
            inner: Mutex::new(backend),
            windows: Mutex::new(HashMap::new()),
            next_token: Mutex::new(0),
        })
    }

    /// Snapshot count of in-flight measurement windows. Useful for
    /// asserting "all dispatches that started have ended" in tests.
    pub fn windows_open(&self) -> usize {
        self.windows.lock().unwrap().len()
    }
}

impl EnergyMeter for SubstrateEnergyMeter {
    fn start(&self) -> Option<u64> {
        let snap = self.inner.lock().unwrap().snapshot().ok()?;
        let token = {
            let mut t = self.next_token.lock().unwrap();
            *t = t.wrapping_add(1);
            *t
        };
        self.windows.lock().unwrap().insert(token, snap);
        Some(token)
    }

    fn end(&self, token: u64) -> Option<u64> {
        let prior = self.windows.lock().unwrap().remove(&token)?;
        let mut be = self.inner.lock().unwrap();
        let now = be.snapshot().ok()?;
        // Branch on backend semantics (see module docstring): macOS
        // IOReport's snapshot is already a since-previous interval,
        // so `now.channels[Total]` *is* this window's energy. The
        // cumulative backends (Linux RAPL, TDP fallback) need a
        // subtraction.
        let delta_uj = if be.name() == "macos-ioreport" {
            now.channels.get(&Channel::Total).copied().unwrap_or(0)
        } else {
            now.total_uj_since(&prior)
        };
        // µJ → pJ. `saturating_mul` because a zero or near-zero delta
        // is the realistic case for tiny dispatches; we'd rather under-
        // report than panic / overflow.
        Some(delta_uj.saturating_mul(1_000_000))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_default_returns_a_meter_on_every_platform() {
        let m = SubstrateEnergyMeter::open_default()
            .expect("open_default_backend always succeeds via fallback_tdp");
        assert_eq!(m.windows_open(), 0);
    }

    #[test]
    fn start_end_returns_a_picojoule_delta() {
        let m = SubstrateEnergyMeter::open_default().expect("backend");
        let t = m.start().expect("start");
        // Do *some* work so the backend has something to attribute
        // energy to. ~1 ms of CPU keeps the test fast while giving a
        // real backend (RAPL / IOReport) a measurable interval.
        let mut x = 0u64;
        for i in 0..200_000u64 { x = x.wrapping_add(i.wrapping_mul(13)); }
        std::hint::black_box(x);
        let pj = m.end(t).expect("end");
        // Don't assert a lower bound on pj — `fallback_tdp` on a
        // sandbox-throttled CI runner can legitimately report 0 for
        // such a short interval. We only assert (a) end() didn't
        // fail, and (b) the window closed.
        let _ = pj;
        assert_eq!(m.windows_open(), 0, "end() must close the window");
    }

    #[test]
    fn end_of_unknown_token_returns_none() {
        let m = SubstrateEnergyMeter::open_default().expect("backend");
        // Token never opened by start() — should fail-safe to None,
        // not panic or return a garbage estimate.
        assert!(m.end(99_999).is_none());
    }
}
