//! In-memory capability table.

use crate::chain::EffectsAllow;
use crate::error::{CapError, Result};
use rce_vm_backend_cpu::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A capability handler — host-provided closure that takes positional
/// inputs and returns a `Value`. Errors propagate as `CapError::Handler`.
pub type CapHandler =
    Arc<dyn Fn(&str, &[Value]) -> Result<Value> + Send + Sync + 'static>;

#[derive(Default)]
pub struct CapTable {
    /// `capability_id` → `method_name` → handler
    handlers: HashMap<String, HashMap<String, CapHandler>>,
    /// Effect allow-list for the host. Any capability requesting effects
    /// outside this set is denied at dispatch time.
    pub effects_allow: EffectsAllow,
}

impl CapTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_effects_allow(mut self, allow: EffectsAllow) -> Self {
        self.effects_allow = allow;
        self
    }

    pub fn register<F>(mut self, capability_id: impl Into<String>, method: impl Into<String>, f: F) -> Self
    where
        F: Fn(&str, &[Value]) -> Result<Value> + Send + Sync + 'static,
    {
        let cap = capability_id.into();
        self.handlers
            .entry(cap)
            .or_default()
            .insert(method.into(), Arc::new(f));
        self
    }

    pub fn lookup(
        &self,
        capability_id: &str,
        method: &str,
    ) -> Result<&CapHandler> {
        let methods = self
            .handlers
            .get(capability_id)
            .ok_or_else(|| CapError::Unregistered(capability_id.into()))?;
        methods.get(method).ok_or_else(|| CapError::UnknownMethod {
            capability: capability_id.into(),
            method: method.into(),
        })
    }
}
