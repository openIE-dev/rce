//! `rce-energy`: RCE's one meter library. Measured and reported joules,
//! with the provenance D.11 asks of a reading, simulated joules for the
//! IDE's actions, and the cumulative snapshots the placement crates read.
//! It joins what were two libraries, the compute crates' `rce-energy` and
//! the IDE's `ide-energy`; the IDE names it `ide_energy`.
//!
//! - [`sim`]: E_sim from telemetry features, with per-feature coefficients
//!   fitted by least squares to measured runs. An uncalibrated gear/model
//!   predicts nothing (`None`), never a default number.
//! - [`meter`]: the `Meter` trait, Linux RAPL (powercap sysfs), NVML behind
//!   the `nvml` feature. Other sources: [`ioreport`] (macOS rails), [`smc`]
//!   (macOS system power), [`powermetrics`] (macOS, root via setup),
//!   [`hwmon`] (Linux GPU/CPU/board drivers incl. Jetson INA3221),
//!   [`supply`] (battery discharge, Linux/Android). [`power`] integrates
//!   polled watts. A source that cannot measure errors; it never returns 0.
//! - [`discover`]: probe every source, list what works and what is tracked.
//! - [`setup`] and [`cli`]: `rce-ide telemetry probe|setup`, consent per step.
//!   See docs/TELEMETRY.md.
//! - [`attrib`]: measured energy over an action's time window, from a meter's
//!   cumulative samples.
//! - [`calib`]: run fixed workloads under the discovered meters and fit
//!   the simulator (`rce-ide telemetry calibrate`).
//! - [`anomaly`]: flag |delta| > k sigma.
//! - [`telemetry`]: cumulative microjoules per channel ([`EnergyTelemetry`]):
//!   the RAPL counters through [`meter::wrap_delta`], or [`FallbackTdp`], an
//!   estimate (modelled, never a reading) where no counter can be read.
pub mod anomaly;
pub mod apportion;
pub mod budget;
pub mod perfetto;
pub mod attrib;
pub mod calib;
pub mod cli;
pub mod discover;
pub mod gpu_launch;
pub mod hwmon;
pub mod ioreport;
pub mod meter;
pub mod power;
pub mod powermetrics;
pub mod setup;
pub mod shared;
pub mod sim;
pub mod smc;
pub mod supply;
pub mod telemetry;

pub use telemetry::{open_default_backend, Calibration, Channel, EnergyError, EnergySnapshot, EnergyTelemetry, FallbackTdp, Rapl};
