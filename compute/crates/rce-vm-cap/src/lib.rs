//! rce capability runtime.
//!
//! Closes the security gap that has existed since v0.1: `GraphNode::CapabilityRef`
//! parses cleanly, makes it through serialization and the type system, but
//! the interpreter has been erroring out on it as `Unsupported`. This crate
//! adds the table, the handles, the delegation-chain validation, and the
//! `LeafDispatch` adapter that makes capability-ref dispatch actually work.
//!
//! ## Model
//!
//! A **capability** is a host-provided function callable from inside a
//! rce graph through a `CapabilityRef { capability_id, version, slot,
//! method, output_type, effects }` node. Examples: `joulesperbit.pay.ledger.transfer`,
//! `crypto.constant_time.aes_gcm.encrypt`, `io.fs.read`.
//!
//! - **Slots** are namespaced: capabilities live under `capability_id`
//!   keys with a slot-name + method-name addressing pattern.
//! - **Effects** are declared per node (`effects: Vec<String>`) and
//!   allowlisted per host policy. A host that doesn't permit
//!   `regulatory:financial` can refuse the dispatch even if the
//!   capability is registered.
//! - **Handles** are 16-byte unforgeable IDs returned by the host;
//!   passing a handle through wires lets a graph propagate authority
//!   without re-resolving the capability table.
//! - **Delegation chains** are Ed25519-signed grants: A signs "B may
//!   exercise capability X with effects E"; B can then sub-delegate to
//!   C; the final dispatch validates the chain end-to-end.
//!
//! ## What this v0.1 implementation does
//!
//! - Defines `CapTable` (in-memory mapping of capability_id → handler)
//! - Defines `CapabilityHandle` (16-byte unforgeable ID)
//! - Defines `Delegation` and `DelegationChain` with Ed25519 signing
//! - Provides `CapabilityDispatcher` that implements
//!   `rce_vm_backend_cpu::LeafDispatch` for the synthetic language
//!   `"capability"` — the interpreter's existing fall-through machinery
//!   routes `CapabilityRef` nodes to it via the same trait that handles
//!   Wasm leaves.
//! - Validates effects against a per-call allow-list at dispatch time
//!   (not at graph construction). Dynamic capability passing works.
//! - Validates Ed25519 delegation chains end-to-end before invoking the
//!   target capability handler.
//!
//! ## What this v0.1 implementation does **not** do (yet)
//!
//! - Persisted capabilities. CapTable is in-memory only; v0.2 adds
//!   on-disk capability indices read from `.fg`'s
//!   `SectionKind::CapabilityIndex` block.
//! - Sigstore / Rekor transparency-log integration. Planned for v0.2
//!   alongside SLSA-3 provenance.
//! - HMAC-protected handle issuance for cross-process capability
//!   passing. Currently handles are validated by chain-presence only.

pub mod chain;
pub mod dispatcher;
pub mod error;
pub mod handle;
pub mod table;
pub mod wire;

pub use chain::{Delegation, DelegationChain, EffectsAllow};
pub use dispatcher::CapabilityDispatcher;
pub use error::{CapError, Result};
pub use handle::CapabilityHandle;
pub use table::{CapHandler, CapTable};
pub use wire::{WireError, WireRequest, WireResponse, WIRE_SCHEMA_VERSION};
