//! Sovereignty layer — code correction isolated from AI and other "listeners".
//!
//! Data sovereignty / software sovereignty / privacy: the deterministic correction path must not
//! leak code to a model, the network, or any telemetry "listener". Correction runs locally,
//! air-gapped, with no model calls; AI is confined to the airlock and never observes raw code unless
//! explicitly permitted. A `Policy` defaults to fully sovereign (no model / no network / no
//! listeners); listeners are blocked unless allowed; every isolated correction is audited.

use crate::repair::repair_to_value;

/// Sovereignty policy. Default: fully isolated (local, no model, no network, no listeners).
#[derive(Clone, Debug)]
pub struct Policy {
    pub allow_model: bool,
    pub allow_network: bool,
    pub allow_listeners: bool,
    pub residency: String,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { allow_model: false, allow_network: false, allow_listeners: false, residency: "local".into() }
    }
}

/// A sovereign execution context: enforces isolation and audits it.
pub struct Sovereign {
    pub policy: Policy,
    pub listeners: Vec<String>,
    pub audit: Vec<String>,
}

impl Sovereign {
    pub fn new(policy: Policy) -> Self {
        Sovereign { policy, listeners: Vec::new(), audit: Vec::new() }
    }

    /// Register a telemetry/AI "listener". Blocked unless the policy explicitly allows it.
    pub fn register_listener(&mut self, name: &str) -> Result<(), String> {
        if !self.policy.allow_listeners {
            return Err(format!("listener '{name}' blocked by sovereignty policy"));
        }
        self.listeners.push(name.into());
        Ok(())
    }

    /// Assert the context is fully isolated (no model, network, or listeners).
    pub fn assert_isolated(&self) -> Result<(), String> {
        if self.policy.allow_model { return Err("model access not sovereign".into()); }
        if self.policy.allow_network { return Err("network access not sovereign".into()); }
        if self.policy.allow_listeners && !self.listeners.is_empty() {
            return Err("active listeners not sovereign".into());
        }
        Ok(())
    }

    /// Correct code in isolation: deterministic, local, no model/network/listeners. Audited.
    pub fn correct_isolated(&mut self, src: &str, accept: i128) -> Result<Option<String>, String> {
        self.assert_isolated()?;
        let out = repair_to_value(src, accept).map(|r| r.repaired);
        self.audit.push(format!("correction isolated (residency={}): no model/network/listeners", self.policy.residency));
        Ok(out)
    }
}