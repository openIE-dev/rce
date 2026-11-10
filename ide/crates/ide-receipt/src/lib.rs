//! `ide-receipt`: one receipt per RCE IDE action.
//!
//! Every receipt carries the simulated energy (`e_sim_j`), a hardware reading
//! (`e_meas`, when a meter produced it, with its source tag), their
//! difference, the checker verdict, the hash of the policy in force, and the
//! BLAKE3 hash of the previous receipt. The body is signed with ed25519.
//! Nothing in this crate invents an energy number: an absent reading is
//! `None`, and `delta_j` is computed when the source is hardware.
//! The wire field stays `e_meas`. Its basis is the source tag's D.11 basis,
//! not the field name: counters are `reported`, a shunt is `measured`,
//! and `EstimatedRemote` is `est`.

pub mod chain;
pub mod map;
pub mod review;

pub use chain::{Ledger, VerifyError, verify_chain};
use ed25519_dalek::{Signature, Signer, Verifier};
pub use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

/// Schema tag written into every body.
pub const SCHEMA: &str = "rce-ide.receipt.v1";
/// `prev_hash` of the first receipt in a chain.
pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// What the action was.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    ReadForm,
    ProposeEdit,
    Check,
    Test,
    Emit,
    Run,
    ModelCall,
    /// An action imported from another receipt format (mol, mei), by name.
    Other(String),
}

/// D.11 basis of a [`MeterSource`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyBasis {
    /// External wall or shunt meter on a named rail.
    Measured,
    /// OS or on-chip counter or sensor.
    Reported,
    /// Model, table, fallback or remote prior.
    Est,
}

/// Where a hardware or tagged-remote energy figure came from.
/// The serde names are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeterSource {
    /// Linux powercap RAPL counters (`energy_uj`). Basis `reported`.
    Rapl,
    /// NVIDIA NVML total energy counter. Basis `reported`.
    Nvml,
    /// macOS powermetrics. Basis `reported`.
    Powermetrics,
    /// macOS IOReport counters read in-process (no root), for example the
    /// Energy Model group. Basis `reported`.
    Ioreport,
    /// Apple SMC system power key (PSTR), polled and integrated. Basis `reported`.
    Smc,
    /// Linux hwmon energy or power attributes (amdgpu, i915/xe, and others).
    /// Basis `reported`.
    Hwmon,
    /// Battery discharge power from the power_supply class (Linux/Android).
    /// Basis `reported`.
    PowerSupply,
    /// NVIDIA Jetson INA shunt on board rails. Basis `measured`.
    Jetson,
    /// A remote model whose provider cannot be metered. Basis `est`.
    EstimatedRemote,
}

impl MeterSource {
    /// D.11 basis of this tag.
    pub fn basis(self) -> EnergyBasis {
        match self {
            MeterSource::Jetson => EnergyBasis::Measured,
            MeterSource::EstimatedRemote => EnergyBasis::Est,
            MeterSource::Rapl
            | MeterSource::Nvml
            | MeterSource::Powermetrics
            | MeterSource::Ioreport
            | MeterSource::Smc
            | MeterSource::Hwmon
            | MeterSource::PowerSupply => EnergyBasis::Reported,
        }
    }

    /// True for a counter or a shunt. False for [`MeterSource::EstimatedRemote`].
    pub fn is_hardware(self) -> bool {
        !matches!(self, MeterSource::EstimatedRemote)
    }
}

/// An energy figure with the meter tag that names its D.11 basis.
/// The JSON object is `{joules, source}`, the same bytes as the previous
/// struct name `Measured`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Reading {
    pub joules: f64,
    pub source: MeterSource,
}

/// The checker's word on the action.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Verdict {
    /// `rce check` admitted the result.
    Pass,
    /// `rce check` reported the result as requiring change; the change was
    /// not applied. The serde tag below is the stored one, so signed chains
    /// verify.
    #[serde(rename = "refused")]
    RequiresChange { diagnostics: Vec<String> },
    /// The action was rejected before the checker ran (bad path, policy).
    Rejected { reason: String },
    /// No checker involved (a read, a model call).
    NotRun,
}

/// Everything about the action except the chain fields. Callers fill this.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Draft {
    pub action: ActionKind,
    /// Gear of the MEI cascade (e.g. `checker`, `lookup`, `local-small`, `frontier`).
    pub gear: String,
    pub model: Option<String>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    /// Predicted joules. `None` when the simulator is not calibrated for this
    /// gear and model: no default number is substituted.
    pub e_sim_j: Option<f64>,
    /// Hardware or tagged-remote figure. Absent is `None`, not 0.
    /// The JSON name `e_meas` is kept. The basis is `source.basis()`.
    pub e_meas: Option<Reading>,
    pub verdict: Verdict,
    /// Hex BLAKE3 of the policy in force.
    pub policy_hash: String,
    /// Action start and end, Unix nanoseconds.
    pub t_start_ns: u64,
    pub t_end_ns: u64,
    /// Free-form, action-specific detail (paths, edit summary, origin format).
    #[serde(default)]
    pub detail: serde_json::Value,
}

impl Draft {
    pub fn new(
        action: ActionKind,
        gear: impl Into<String>,
        policy_hash: impl Into<String>,
    ) -> Self {
        Draft {
            action,
            gear: gear.into(),
            model: None,
            tokens_in: 0,
            tokens_out: 0,
            e_sim_j: None,
            e_meas: None,
            verdict: Verdict::NotRun,
            policy_hash: policy_hash.into(),
            t_start_ns: 0,
            t_end_ns: 0,
            detail: serde_json::Value::Null,
        }
    }
}

/// The signed part of a receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Body {
    pub schema: String,
    pub seq: u64,
    pub draft: Draft,
    /// `e_meas - e_sim`, only when both exist and the source is hardware.
    pub delta_j: Option<f64>,
    pub prev_hash: String,
}

impl Body {
    /// Canonical bytes: serde_json of the body. Field order is fixed by the
    /// struct and `detail` maps are sorted (serde_json without
    /// `preserve_order`), so the same body always gives the same bytes.
    pub fn canonical(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("receipt body serializes")
    }
    pub fn hash(&self) -> String {
        blake3::hash(&self.canonical()).to_hex().to_string()
    }
}

/// `e_meas - e_sim` when it means something.
pub fn delta(e_sim: Option<f64>, e_meas: Option<Reading>) -> Option<f64> {
    match (e_sim, e_meas) {
        (Some(s), Some(m)) if m.source.is_hardware() && s.is_finite() && m.joules.is_finite() => {
            Some(m.joules - s)
        }
        _ => None,
    }
}

/// A signed receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Receipt {
    pub body: Body,
    /// Hex BLAKE3 of `body.canonical()`.
    pub hash: String,
    /// Hex ed25519 public key.
    pub signer: String,
    /// Hex ed25519 signature over the 32 raw bytes of the BLAKE3 hash.
    pub signature: String,
}

impl Receipt {
    pub fn sign(body: Body, key: &SigningKey) -> Receipt {
        let h = blake3::hash(&body.canonical());
        let sig = key.sign(h.as_bytes());
        Receipt {
            hash: h.to_hex().to_string(),
            signer: hex::encode(key.verifying_key().to_bytes()),
            signature: hex::encode(sig.to_bytes()),
            body,
        }
    }

    /// Recompute the hash and check the signature against `key`.
    pub fn verify(&self, key: &VerifyingKey) -> Result<(), VerifyError> {
        let h = blake3::hash(&self.body.canonical());
        if h.to_hex().as_str() != self.hash {
            return Err(VerifyError::Hash(self.body.seq));
        }
        if hex::encode(key.to_bytes()) != self.signer {
            return Err(VerifyError::Signer(self.body.seq));
        }
        let raw =
            hex::decode(&self.signature).map_err(|_| VerifyError::Signature(self.body.seq))?;
        let arr: [u8; 64] = raw
            .try_into()
            .map_err(|_| VerifyError::Signature(self.body.seq))?;
        key.verify(h.as_bytes(), &Signature::from_bytes(&arr))
            .map_err(|_| VerifyError::Signature(self.body.seq))
    }
}

/// A fresh signing key from the OS random source.
pub fn generate_key() -> Result<SigningKey, getrandom::Error> {
    let mut seed = [0u8; 32];
    getrandom::fill(&mut seed)?;
    Ok(SigningKey::from_bytes(&seed))
}

/// Hex BLAKE3 of a policy document's canonical JSON.
pub fn policy_hash(policy: &serde_json::Value) -> String {
    blake3::hash(&serde_json::to_vec(policy).expect("json"))
        .to_hex()
        .to_string()
}
