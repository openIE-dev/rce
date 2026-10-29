//! Panel models: plain data, no I/O. Shared by the native IDE ([`crate::vm`])
//! and the browser build.
pub use crate::buffer::{EditOp, Motion, Visible};
use crate::lsp::Diagnostic;
use ide_receipt::{ActionKind, Receipt, Verdict};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EditorVm {
    pub path: String,
    pub text: String,
    pub diagnostics: Vec<Diagnostic>,
    /// Source of the diagnostics (`rce` language server name, and transport).
    pub server: String,
    /// What the editor shows: visible lines, cursor, selection, preedit.
    #[serde(default)]
    pub view: Visible,
    /// The buffer differs from the file on disk.
    #[serde(default)]
    pub dirty: bool,
    #[serde(default)]
    pub can_undo: bool,
    #[serde(default)]
    pub can_redo: bool,
}

impl EditorVm {
    /// Lines with their diagnostics, for the gutter.
    pub fn lines(&self) -> Vec<(usize, &str, Vec<&str>)> {
        self.text
            .split('\n')
            .enumerate()
            .map(|(i, l)| {
                let d = self
                    .diagnostics
                    .iter()
                    .filter(|d| d.line == i + 1)
                    .map(|d| d.message.as_str())
                    .collect();
                (i + 1, l, d)
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProposalStatus {
    Pending,
    Accepted {
        receipt: String,
    },
    /// The checker refused it on accept; nothing was written.
    RequiresChange {
        diagnostics: Vec<String>,
        receipt: Option<String>,
    },
    Rejected {
        receipt: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Proposal {
    pub id: u64,
    pub path: String,
    /// Who proposed it (gear / agent) and why.
    pub origin: String,
    pub summary: String,
    /// The edits, as rce-mcp `propose_edit` JSON.
    pub edits: Vec<Value>,
    pub status: ProposalStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReceiptRow {
    pub seq: u64,
    pub action: String,
    pub gear: String,
    pub e_sim_j: Option<f64>,
    pub e_meas_j: Option<f64>,
    /// `rapl`, `nvml`, `powermetrics`, `jetson`, `estimated-remote`.
    pub source: Option<String>,
    /// Only for hardware measurements with a prediction.
    pub delta_j: Option<f64>,
    pub verdict: String,
    pub hash: String,
}

impl ReceiptRow {
    pub fn of(r: &Receipt) -> Self {
        let d = &r.body.draft;
        let source = d.e_meas.map(|m| {
            serde_json::to_value(m.source)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default()
        });
        ReceiptRow {
            seq: r.body.seq,
            action: match &d.action {
                ActionKind::Other(s) => s.clone(),
                a => serde_json::to_value(a)
                    .ok()
                    .and_then(|v| v.as_str().map(str::to_string))
                    .unwrap_or_else(|| format!("{a:?}")),
            },
            gear: d.gear.clone(),
            e_sim_j: d.e_sim_j,
            e_meas_j: d.e_meas.map(|m| m.joules),
            source,
            delta_j: r.body.delta_j,
            verdict: match &d.verdict {
                Verdict::Pass => "pass".into(),
                Verdict::RequiresChange { .. } => "requires change".into(),
                Verdict::Rejected { .. } => "rejected".into(),
                v => format!("{v:?}").to_lowercase(),
            },
            hash: r.hash.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyTotals {
    pub actions: usize,
    pub e_sim_j: f64,
    pub with_sim: usize,
    pub e_meas_hw_j: f64,
    pub with_hw: usize,
    pub e_est_remote_j: f64,
    pub delta_j: f64,
    pub with_delta: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReceiptsVm {
    pub rows: Vec<ReceiptRow>,
    pub totals: EnergyTotals,
    /// `verify_chain` result over the whole ledger.
    pub chain_ok: Option<bool>,
}

impl ReceiptsVm {
    pub fn from_receipts(rs: &[Receipt], key: Option<&ide_receipt::VerifyingKey>) -> Self {
        let rows: Vec<ReceiptRow> = rs.iter().map(ReceiptRow::of).collect();
        let mut t = EnergyTotals {
            actions: rows.len(),
            ..Default::default()
        };
        for r in &rows {
            if let Some(s) = r.e_sim_j {
                t.e_sim_j += s;
                t.with_sim += 1;
            }
            match (r.e_meas_j, r.source.as_deref()) {
                (Some(m), Some("estimated-remote")) => t.e_est_remote_j += m,
                (Some(m), Some(_)) => {
                    t.e_meas_hw_j += m;
                    t.with_hw += 1;
                }
                _ => {}
            }
            if let Some(d) = r.delta_j {
                t.delta_j += d;
                t.with_delta += 1;
            }
        }
        let chain_ok = key.map(|k| ide_receipt::verify_chain(rs, k, ide_receipt::GENESIS).is_ok());
        ReceiptsVm {
            rows,
            totals: t,
            chain_ok,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryHit {
    pub key: String,
    pub score: f64,
    pub detail: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryVm {
    pub query: String,
    pub hits: Vec<MemoryHit>,
    pub gear: Option<String>,
    pub escalate: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NetVm {
    /// Declared endpoints waiting for a first decision.
    /// `(endpoint key, reason)`.
    pub pending: Vec<(String, String)>,
    pub decided: Vec<(String, bool)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Action {
    Accept(u64),
    Reject(u64),
    Recall(String),
    Net {
        key: String,
        allow: bool,
    },
    /// Replace the editor text, re-run diagnostics.
    SetText(String),
    /// An editing command on the buffer (keys, IME, mouse).
    Edit(EditOp),
    /// Turn the buffer into a pending `set_file` proposal; the file changes
    /// only when that proposal is accepted (and passes the checker).
    Save,
    Focus(Focus),
    Refresh,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Focus {
    #[default]
    Editor,
    Memory,
}

/// Everything the view draws.
pub struct Panels<'a> {
    pub editor: &'a EditorVm,
    pub proposals: &'a [Proposal],
    pub receipts: &'a ReceiptsVm,
    pub memory: &'a MemoryVm,
    pub net: &'a NetVm,
    pub focus: Focus,
}
