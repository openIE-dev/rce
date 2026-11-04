//! rce wgpu backend.
//!
//! Implements the [`rce_vm_backend_cpu::OpDispatch`] trait by routing
//! ML primitives to WebGPU compute shaders. The same code compiles for
//! native (Metal / Vulkan / DX12) and browser (WebGPU via
//! `wasm32-unknown-unknown`) — wgpu unifies both targets, which is the
//! whole reason rce picks it over a vendor-specific backend.
//!
//! ## Status (v0.3)
//!
//! Custom ops (via `OpDispatch::execute`):
//! - `matmul`: 2D and batched 3D, f32, equal batch dims
//! - `softmax`: last-axis only, f32 (other axes → CPU)
//! - `transpose`: 2D `[1,0]` only, f32
//! - `relu`, `sigmoid`, `tanh`, `gelu`: element-wise, f32
//! - `matmul_scale_softmax`: fused single-kernel `softmax((a@b)*scale, axis=-1)`,
//!   2D, N ≤ 256 (workgroup-shared budget)
//!
//! Primitive ops (via `OpDispatch::execute_primitive`):
//! - `Add` / `Sub` / `Mul` / `Div` on f32 tensors with three dispatch paths:
//!   (1) same-shape, (2) scalar broadcast, (3) full N-D broadcast for rank ≤ 4.
//!
//! ## Precision
//!
//! WGSL is **f32-only by spec**. There is no portable f64 path in WebGPU
//! shaders — some native vendors expose double-precision via extensions,
//! but no browser does, and wgpu does not surface them in stable. This
//! backend therefore claims only f32 ops and returns
//! [`InterpretError::Unsupported`] for f64 inputs. The interpreter then
//! falls through to the CPU kernel, which is f64-correct.
//!
//! Users who want GPU acceleration on f64-typed graphs should pass
//! `--cast-f32` to the CLI (or otherwise cast before interpretation).
//! There is no silent precision loss anywhere in this backend.
//!
//! ## Falling through
//!
//! Every wgpu kernel returns `Unsupported(_)` when its sweet spot doesn't
//! match (rank, dtype, shape pattern, workgroup-shared limits). The
//! interpreter consults the next backend in `cfg.ops` and finally the
//! CPU `ml::dispatch_custom`. This is how the architecture stays graceful
//! across heterogeneous hardware.

pub mod error;
pub mod executor;
pub mod kernels;

pub use error::{Result, WgpuError};
pub use executor::WgpuExecutor;
