//! `mei-agent`: the MEI agent for the RCE IDE.
//!
//! A turn is a mei-machine statechart with four parallel regions:
//! `turn` (plan, propose, check, accept, escalate, failed), `gears` (gear0..gear4,
//! each invoking its own child chart), `verdict` (none, done, not_done, reject)
//! and `budget` (ok, out). Guards read only the configuration: accept needs
//! `v_done`; escalate needs `v_change` or `v_not_done`, `budget_ok`, and a gear
//! below 4. Anything else fails the turn. So the agent climbs a gear only on a
//! checker reject or a not-done decision, and only while joules remain.
//!
//! Gears: G0 checker and tests, G1 rules lookup, G2 System-One decision model
//! (`POST /v1/systemone`, the Jev/Laya/Kev protocol), G3 a local
//! OpenAI-compatible model, G4 a frontier model from the provider catalog.
//! Every change goes through `rce-mcp` `propose_edit`, so the checker decides.
//! Every step writes an `ide-receipt`.
pub mod agent;
pub mod catalog;
pub mod chart;
pub mod http;

pub use agent::{Acceptance, Agent, AgentConfig, ExternalG4, ExternalTurn, Outcome, Rule, Task};
pub use catalog::{Catalog, Provider};
