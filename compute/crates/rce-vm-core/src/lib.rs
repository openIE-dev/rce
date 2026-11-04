//! rce Graph VM core.
//!
//! This crate is intentionally minimal and synchronous. It defines the
//! `Backend` trait that backends (CPU, wgpu, CUDA, Metal, Wasm-leaf, NPU,
//! silicon) implement, and the `Tensor`, `Region`, `Plan`, and `Cost`
//! primitives the scheduler operates on. Async work-stealing dispatch and
//! tokio integration live in `rce-vm-scheduler`. Memory regions and
//! device allocators live in `rce-vm-mem`. Capability resolution and
//! enforcement live in `rce-vm-cap`.
//!
//! This crate has no I/O and no async; it is `no_std`-friendly under the
//! `no_std` feature (TODO).
//!
//! Status: phase-0 skeleton. Backend trait is settled in shape; concrete
//! backends are stubbed.

pub mod backend;
pub mod cost;
pub mod plan;
pub mod region;
pub mod tensor;
pub mod value;

pub use backend::{Backend, BackendId, BackendKind, Capabilities};
pub use cost::{Cost, EnergyEstimate};
pub use plan::{Plan, PlanError, Placement};
pub use region::{Region, RegionKind};
pub use tensor::{DType, Tensor, TensorHandle};
pub use value::{Value, ValueError};
