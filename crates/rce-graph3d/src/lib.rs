//! `rce-graph3d`: code out of the text era.
//!
//! The RCE Form is a graph; text is one view of it. This crate is the 3D
//! view, the time axis and the telemetry overlay:
//!
//! - [`graph`] - nodes (state, event, step, fn, type, error, and hardware)
//!   and edges (refs, calls, data flow, code -> hardware). Built from
//!   rce-memory's `FormIndex` with feature
//!   `native`; plain serde data otherwise (the browser receives it as JSON).
//! - [`hw`] - hardware topology discovered at runtime (Linux sysfs, macOS
//!   sysctl), never invented: what cannot be probed is `declared` or absent.
//! - [`layout`] - deterministic 3D force-directed layout (Barnes-Hut octree),
//!   warm-started from the previous version, with layered modes.
//! - [`camera`] - orbit camera, projection and picking.
//! - [`time`] - Form versions, bitemporal facts and receipts on one time
//!   axis, with a scrubber.
//! - [`overlay`] - energy per node (E_meas, else E_sim, marked), traffic per
//!   edge, |delta| > k sigma flags.
//! - [`dash`] - the telemetry dashboard as 2D primitives (sparklines, W, J,
//!   source tag, accuracy, status; never a zero for a source that is not read).
//! - `render` (feature `gpu`): instanced nodes and edges on wgpu.
pub mod camera;
pub mod dash;
pub mod graph;
#[cfg(feature = "native")]
pub mod history;
pub mod hw;
pub mod hwloc;
pub mod layout;
pub mod lens;
pub mod math;
pub mod neighbors;
#[cfg(feature = "native")]
pub mod native;
pub mod overlay;
pub mod query;
#[cfg(feature = "gpu")]
pub mod render;
pub mod scene3d;
pub mod scip;
pub mod time;

pub use graph::{Edge, EdgeKind, Graph, Kind, Node};
