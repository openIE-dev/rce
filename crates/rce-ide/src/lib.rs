//! `rce-ide`: the RCE IDE app.
//!
//! - [`vm`]: view models (editor + language-server diagnostics, MEI
//!   proposals with accept/reject through rce-mcp, receipts and energy,
//!   memory recall, network approvals). No rendering; tested headless.
//! - [`view`]: the photon `ViewNode` tree for those models, and tap dispatch.
//! - [`scene`]: photon draw commands flattened to quads and text runs.
//! - [`buffer`]: the editor's rope buffer (cursor, selection, undo/redo, IME preedit, scrolling).
//! - [`clip`]: system clipboard (arboard, pure Rust backends) behind a trait.
//! - [`lsp`]: LSP client for the RCE language server.
//! - `gpu` (feature `gpu`): the wgpu + Parley renderer, with an offscreen path.
//! - `graph_panel` (feature `gpu`): the 3D graph mode (rce-graph3d) and the
//!   telemetry dashboard, composed with the 2D UI; linked to the editor.
#[cfg(not(target_arch = "wasm32"))]
pub mod app;
pub mod buffer;
#[cfg(not(target_arch = "wasm32"))]
pub mod calibrate;
#[cfg(not(target_arch = "wasm32"))]
pub mod ci;
pub mod clip;
pub mod cost;
#[cfg(not(target_arch = "wasm32"))]
pub mod demo;
pub mod graph_edit;
pub mod holes;
pub mod lsp;
pub mod model;
pub mod png;
pub mod scene;
pub mod view;
#[cfg(not(target_arch = "wasm32"))]
pub mod vm;
#[cfg(all(target_arch = "wasm32", feature = "gpu"))]
pub mod web;

#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "gpu")]
pub mod graph_panel;
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod gpu_launch;
#[cfg(all(feature = "gpu", target_arch = "wasm32"))]
pub mod graph_web;
