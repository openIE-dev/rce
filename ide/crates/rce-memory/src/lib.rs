//! `rce-memory`: memory for the MEI agent in the RCE IDE.
//!
//! * [`form`]: the RCE Form is the code graph. Definitions, references, call
//!   edges and a ranked repo map are derived caches keyed by the Form hash.
//! * [`fact`]: bitemporal facts (world time and system time). A contradiction
//!   ends a fact; nothing is deleted.
//! * [`engine`]: the [`engine::MemoryEngine`] trait, an in-memory reference
//!   engine, and [`conformance`] tests any backend must pass.
//! * [`redb_engine`]: the default embedded backend (redb, pure Rust, no server).
//! * [`notes`]: Markdown + YAML frontmatter notes, synced both ways, files win.
//! * [`bm25`], [`vector`], [`cascade`]: retrieval gears, cheapest first.
//! * [`service`]: ties it together and writes one receipt per recall/store.
pub mod bm25;
pub mod cascade;
pub mod conformance;
pub mod engine;
pub mod fact;
pub mod form;
pub mod notes;
pub mod redb_engine;
pub mod service;
pub mod vector;

/// Unix milliseconds now.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
