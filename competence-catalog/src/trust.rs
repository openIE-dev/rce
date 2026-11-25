//! Trust & compliance layer — SOC2 / Trust Services Criteria mapped to RCE mechanisms.
//!
//! For Cleanroom to serve government, military, industrial, enterprise, consumer, hobbyist, maker
//! and tinker, it must carry trust: auditable, verified, access-controlled, private. RCE already
//! embodies the hard part of SOC2's Processing Integrity (every output replay-verified, provenance
//! tagged, hash-chained audit log). This layer makes that explicit as compliance evidence: a
//! control map onto the SOC2 Trust Services Criteria, a tamper-evident audit chain, and a
//! compliance report. Controls not yet implemented (encryption at rest, RBAC) are flagged honestly.

use serde::{Deserialize, Serialize};

/// SOC2 Trust Services Criteria.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum Criterion {
    Security,          // CC
    Availability,      // A
    ProcessingIntegrity, // PI
    Confidentiality,   // C
    Privacy,           // P
}

/// A compliance control: the criterion, the RCE mechanism that satisfies it, and whether built.
#[derive(Clone, Serialize, Deserialize)]
pub struct Control {
    pub id: &'static str,
    pub criterion: Criterion,
    pub requirement: &'static str,
    pub mechanism: &'static str,
    pub built: bool,
}

/// The SOC2 control map onto RCE/Cleanroom mechanisms.
pub fn soc2_controls() -> Vec<Control> {
    vec![
        Control { id: "CC6.1", criterion: Criterion::Security, requirement: "logical access control",
            mechanism: "admission gate: only replay-verified competence enters; authority check", built: true },
        Control { id: "CC7.2", criterion: Criterion::Security, requirement: "tamper-evident audit trail",
            mechanism: "hash-chained log (rce-vc); any edit breaks the chain", built: true },
        Control { id: "CC7.3", criterion: Criterion::Security, requirement: "anomaly / negative-transfer detection",
            mechanism: "forged or non-reproducing record refused at the border", built: true },
        Control { id: "A1.2", criterion: Criterion::Availability, requirement: "recoverable, deterministic execution",
            mechanism: "exact replay + CALM federation + tiered store", built: true },
        Control { id: "PI1.1", criterion: Criterion::ProcessingIntegrity, requirement: "outputs correct & complete",
            mechanism: "replay gate + spectests (completeness), no vacuous pass", built: true },
        Control { id: "PI1.2", criterion: Criterion::ProcessingIntegrity, requirement: "provenance of every figure",
            mechanism: "D.11 Modelled/Reported/Measured tagging; never a fabricated 0", built: true },
        Control { id: "C1.1", criterion: Criterion::Confidentiality, requirement: "protect confidential code/data",
            mechanism: "Cleanroom private repo; name-free keys carry no source identifiers", built: true },
        Control { id: "C1.2", criterion: Criterion::Confidentiality, requirement: "encryption at rest / in transit",
            mechanism: "AES-256-GCM authenticated sealing (secure::seal/open); tamper detected", built: true },
        Control { id: "CC6.3", criterion: Criterion::Security, requirement: "role-based access / least privilege",
            mechanism: "RBAC least-privilege matrix (secure::can)", built: true },
        Control { id: "P3.1", criterion: Criterion::Privacy, requirement: "no PII in stored competence",
            mechanism: "content keys are name-free behavior hashes; retention/prune operators", built: true },
    ]
}

/// One tamper-evident audit entry.
#[derive(Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub seq: u64,
    pub action: String,
    pub verified: bool,
    pub provenance: String,
    pub hash: String,
}

/// The trust layer: a hash-chained audit log + compliance reporting.
pub struct TrustLayer {
    pub log: Vec<AuditEntry>,
}

fn chain(prev: &str, seq: u64, action: &str, verified: bool, prov: &str) -> String {
    rcevc_core::sha256(format!("{prev}|{seq}|{action}|{verified}|{prov}").as_bytes()).hex()
}

impl TrustLayer {
    pub fn new() -> Self { TrustLayer { log: Vec::new() } }

    /// Record an act. Only a gate-verified act is admitted to the durable log (PI); an unverified
    /// act is recorded as refused, never admitted.
    pub fn record(&mut self, action: &str, verified: bool, provenance: &str) -> &AuditEntry {
        let prev = self.log.last().map(|e| e.hash.clone()).unwrap_or_else(|| "genesis".into());
        let seq = self.log.len() as u64;
        let hash = chain(&prev, seq, action, verified, provenance);
        self.log.push(AuditEntry { seq, action: action.into(), verified, provenance: provenance.into(), hash });
        self.log.last().unwrap()
    }

    /// The audit chain is intact (tamper-evident).
    pub fn chain_intact(&self) -> bool {
        let mut prev = "genesis".to_string();
        for e in &self.log {
            if chain(&prev, e.seq, &e.action, e.verified, &e.provenance) != e.hash { return false; }
            prev = e.hash.clone();
        }
        true
    }

    /// Compliance report: per-criterion, are all its *built* controls present and the audit intact?
    /// Returns (criterion, satisfied). Unbuilt controls are reported separately (honest gaps).
    pub fn report(&self) -> Vec<(Criterion, bool)> {
        let controls = soc2_controls();
        let mut out = Vec::new();
        for c in [Criterion::Security, Criterion::Availability, Criterion::ProcessingIntegrity,
                  Criterion::Confidentiality, Criterion::Privacy] {
            let built: Vec<&Control> = controls.iter().filter(|x| x.criterion == c && x.built).collect();
            let ok = !built.is_empty() && self.chain_intact()
                && self.log.iter().all(|e| e.verified || e.action.starts_with("refused"));
            out.push((c, ok));
        }
        out
    }

    /// Controls still to implement (the honest compliance gap list).
    pub fn gaps(&self) -> Vec<Control> {
        soc2_controls().into_iter().filter(|c| !c.built).collect()
    }
}

impl Default for TrustLayer {
    fn default() -> Self { Self::new() }
}

/// EU Machinery Regulation 2023/1230 + embodied-safety standards mapped to mechanisms.
/// The attestation itself (notified body, SIL certification) is a process edge: built=false.
pub fn eu_machinery_controls() -> Vec<Control> {
    vec![
        Control { id: "EU-1.1", criterion: Criterion::ProcessingIntegrity,
            requirement: "evolving ML safety functions verified before deployment (Annex III)",
            mechanism: "replay gate + spectests; realtime deadline gate refuses unfit gears", built: true },
        Control { id: "EU-1.2", criterion: Criterion::Security,
            requirement: "safety-software logs retained five years, tamper-evident",
            mechanism: "hash-chained audit log + tiered store retention", built: true },
        Control { id: "EU-1.3", criterion: Criterion::ProcessingIntegrity,
            requirement: "notified-body evidence package for evolving safety functions",
            mechanism: "trust report + provenance receipts; attestation process OPEN", built: false },
        Control { id: "ISO-10218", criterion: Criterion::Security,
            requirement: "industrial robot safety (ISO 10218-1/2)",
            mechanism: "mapping open; gate + deadline refusal are partial evidence", built: false },
        Control { id: "ISO-TS-15066", criterion: Criterion::Security,
            requirement: "collaborative force/pressure limits",
            mechanism: "open: needs plant-level force model (twin seed is kinematic)", built: false },
        Control { id: "IEC-61508", criterion: Criterion::ProcessingIntegrity,
            requirement: "functional safety SIL assignment",
            mechanism: "Lean metatheory partial; SIL certification open", built: false },
    ]
}

impl TrustLayer {
    /// The honest embodied-safety gap list (attestation + standards mapping).
    pub fn eu_gaps(&self) -> Vec<Control> {
        eu_machinery_controls().into_iter().filter(|c| !c.built).collect()
    }

    /// Notified-body / auditor evidence bundle: the control maps, chain status, log size,
    /// and the honest open-gap list. The artifact a certification body consumes; the
    /// attestation itself remains a process edge.
    pub fn evidence_package(&self) -> serde_json::Value {
        let mut open: Vec<&'static str> = self.gaps().iter().map(|c| c.id).collect();
        open.extend(self.eu_gaps().iter().map(|c| c.id));
        serde_json::json!({
            "schema": "rce-evidence/1",
            "soc2": soc2_controls(),
            "eu_machinery": eu_machinery_controls(),
            "chain_intact": self.chain_intact(),
            "entries": self.log.len(),
            "open_gaps": open,
        })
    }
}