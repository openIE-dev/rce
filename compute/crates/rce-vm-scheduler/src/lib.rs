//! rce async scheduler.
//!
//! Drives a [`FunctionGraph`] with **parallel node execution** instead of
//! the strict topological loop in [`rce_vm_backend_cpu::interpret`].
//! Independent branches run concurrently on a tokio multi-thread runtime;
//! per-wire `oneshot::channel`s carry values between nodes; sync backends
//! (CPU, wgpu) are wrapped in `spawn_blocking` so the scheduler is never
//! starved by a long-running kernel.
//!
//! ## Why
//!
//! The synchronous interpreter in `rce-vm-backend-cpu` is correct and
//! deterministic, but it never overlaps — even when a graph has two
//! parallel matmul branches, they run sequentially. The scheduler is the
//! piece the rest of the architecture was designed against:
//!
//! - [`OpDispatch`](rce_vm_backend_cpu::OpDispatch) returning
//!   `Unsupported` is what lets the scheduler route a node to *any*
//!   backend that claims it; this stays the same.
//! - Wires already carry [`WireKind`](rce_graph::graph::WireKind)
//!   information (Move / Stream / Channel / Feedback) — v0.1 scheduler
//!   honours `Move` (the default); v0.2 will honour the others properly.
//!
//! ## v0.1 scope
//!
//! - Multi-threaded tokio runtime; native-only (the wasm32 target uses
//!   the synchronous interpreter — browsers don't have threads in the
//!   way the scheduler needs).
//! - Each node spawns one task; data flows on per-wire `oneshot::Sender`.
//! - Sync backends wrapped in `spawn_blocking`. wgpu compute submissions
//!   queue without awaiting, so multiple GPU dispatches overlap on the
//!   driver side.
//! - No backpressure, no cost-based placement (uses the same
//!   first-claim-wins logic as the synchronous interpreter). Cost-based
//!   placement is v0.2 and depends on the benchmark scaffold.
//! - Returns the same `Value` shape as the synchronous interpreter; no
//!   semantics change.

pub mod error;
pub mod placement;
pub mod scheduler;

pub use error::{Result, ScheduleError};
pub use placement::{pick_priced, PricedBackend};
pub use scheduler::{run, run_async, Backends};
