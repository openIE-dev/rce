//! rce cross-backend conformance suite.
//!
//! Every supported execution path — CPU sync, CPU async, wgpu sync,
//! wgpu async — runs the same corpus of graphs and produces outputs
//! that are compared:
//!
//! - **bit-equal** for integer / boolean dtypes
//! - **within tolerance** for floating-point dtypes (relative `1e-5` for
//!   f32, `1e-12` for f64; non-finite values must match exactly)
//!
//! ## Why this is load-bearing
//!
//! Without a conformance corpus, every change to a kernel — adding a
//! new wgpu shader, tweaking a CPU broadcasting fast path, fusing two
//! ops — is a coin flip on whether outputs still agree across backends.
//! With it, divergence is caught at PR time and the offending change
//! either documents the divergence (and tightens tolerance) or doesn't
//! merge.
//!
//! ## Corpus generation
//!
//! Cases are built programmatically: per-op cases on representative
//! shapes (scalars, 1-D, 2-D, with broadcasting), plus end-to-end
//! graph cases (MLP, attention). Every case carries:
//!
//! - a name (stable identifier for diffing reports)
//! - a `FunctionGraph`
//! - bound `Inputs`
//! - the dtype family the case exercises
//! - an optional expected output (for cases where a reference is known)
//!
//! ## Output
//!
//! - Per-case pass/fail with concrete numeric divergence on failure
//! - A summary count by backend pair
//! - Markdown table written to stdout; suitable for piping into release
//!   notes or a CI artifact

pub mod compare;
pub mod corpus;
pub mod report;
pub mod runner;

pub use compare::{compare, ToleranceMode};
pub use corpus::{ConformanceCase, DtypeFamily};
pub use report::{ReportLine, Verdict};
pub use runner::{run_corpus, BackendId, RunOptions};
