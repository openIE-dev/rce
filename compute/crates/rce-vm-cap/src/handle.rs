//! Unforgeable 16-byte capability handles.
//!
//! Issued by the host when a graph receives a capability via input
//! binding or via delegation. Passed across wires (as `Value::Capability(_)`)
//! to flow authority through a graph without re-resolving the cap table.

use rand_core::{OsRng, RngCore};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CapabilityHandle(pub [u8; 16]);

impl CapabilityHandle {
    pub fn new_random() -> Self {
        let mut bytes = [0u8; 16];
        OsRng.fill_bytes(&mut bytes);
        Self(bytes)
    }

    pub fn from_bytes(b: [u8; 16]) -> Self {
        Self(b)
    }

    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    pub fn hex(&self) -> String {
        hex::encode(self.0)
    }
}

impl std::fmt::Display for CapabilityHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cap:{}", self.hex())
    }
}
