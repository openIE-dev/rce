//! View models: the IDE's state and actions, with no rendering in them.
use crate::buffer::Buffer;
use crate::lsp::Lsp;
pub use crate::model::*;
use ide_receipt::{ActionKind, Draft, Verdict, chain::read_jsonl};
use ide_sandbox::net::{Approvals, NetManifest};
use rce_mcp::Server;
use rce_mcp::items::Edit;
use serde_json::json;
use std::path::{Path, PathBuf};

pub struct Ide {
    pub server: Server,
    pub lsp: Lsp,
    pub editor: EditorVm,
    pub buffer: Buffer,
    /// Where copy/cut/paste go (in-process by default; the desktop app
    /// installs the system clipboard).
    pub clip: Box<dyn crate::clip::Clipboard>,
    /// The file as on disk, for `dirty`.
    saved: String,
    pub focus: Focus,
    pub proposals: Vec<Proposal>,
    pub receipts: ReceiptsVm,
    pub memory: MemoryVm,
    pub net: NetVm,
    pub ledger_path: PathBuf,
    pub approvals_path: PathBuf,
    next_id: u64,
}

fn uri(root: &Path, rel: &str) -> String {
    format!("file://{}", root.join(rel).display())
}

impl Ide {
    pub fn new(server: Server, lsp: Lsp, ledger_path: PathBuf, file: &str) -> Self {
        let approvals_path = server.ws.root.join(".rce-ide/net-approvals.json");
        let mut ide = Ide {
            server,
            lsp,
            editor: EditorVm::default(),
            buffer: Buffer::default(),
            clip: Box::new(crate::clip::Memory::default()),
            saved: String::new(),
            focus: Focus::Editor,
            proposals: vec![],
            receipts: ReceiptsVm::default(),
            memory: MemoryVm::default(),
            net: NetVm::default(),
            ledger_path,
            approvals_path,
            next_id: 1,
        };
        ide.open(file);
        ide.refresh();
        ide
    }

    pub fn open(&mut self, rel: &str) {
        let text = self
            .server
            .ws
            .resolve(rel, false)
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();
        let u = uri(&self.server.ws.root, rel);
        let diagnostics = self.lsp.open(&u, &text);
        let server = format!(
            "{} ({})",
            self.lsp.server_name,
            if self.lsp.is_stdio() {
                "stdio"
            } else {
                "in-process"
            }
        );
        if self.editor.path == rel {
            // Reload after an accepted proposal: one undoable change.
            self.buffer.set_text(&text);
        } else {
            let rows = self.buffer.rows;
            self.buffer = Buffer::new(&text);
            self.buffer.rows = rows;
        }
        self.saved = text.clone();
        self.editor = EditorVm {
            path: rel.into(),
            text,
            diagnostics,
            server,
            ..Default::default()
        };
        self.sync_editor();
    }

    /// Visible editor lines (from the panel height).
    pub fn set_rows(&mut self, rows: usize) {
        if self.buffer.rows != rows {
            self.buffer.rows = rows;
            self.buffer.reveal();
            self.sync_editor();
        }
    }

    fn sync_editor(&mut self) {
        let e = &mut self.editor;
        e.view = self.buffer.visible();
        e.text = self.buffer.text();
        e.dirty = e.text != self.saved;
        e.can_undo = self.buffer.can_undo();
        e.can_redo = self.buffer.can_redo();
    }

    fn rediagnose(&mut self) {
        let u = uri(&self.server.ws.root, &self.editor.path);
        self.editor.diagnostics = self.lsp.change(&u, &self.buffer.text());
    }

    /// Queue an MEI proposal (not applied until accepted).
    pub fn propose(&mut self, path: &str, origin: &str, summary: &str, edits: Vec<Edit>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let edits = edits
            .iter()
            .map(|e| serde_json::to_value(e).expect("edit json"))
            .collect();
        self.proposals.push(Proposal {
            id,
            path: path.into(),
            origin: origin.into(),
            summary: summary.into(),
            edits,
            status: ProposalStatus::Pending,
        });
        id
    }

    pub fn panels(&self) -> Panels<'_> {
        Panels {
            editor: &self.editor,
            proposals: &self.proposals,
            receipts: &self.receipts,
            memory: &self.memory,
            net: &self.net,
            focus: self.focus,
        }
    }

    pub fn pending(&self) -> Vec<&Proposal> {
        self.proposals
            .iter()
            .filter(|p| p.status == ProposalStatus::Pending)
            .collect()
    }

    pub fn apply(&mut self, a: Action) -> Result<(), String> {
        match a {
            Action::Accept(id) => {
                let p = self
                    .proposals
                    .iter()
                    .find(|p| p.id == id && p.status == ProposalStatus::Pending)
                    .cloned()
                    .ok_or("no such pending proposal")?;
                let r = self
                    .server
                    .call("propose_edit", &json!({"path": p.path, "edits": p.edits}))
                    .map_err(|(_, m)| m)?;
                let sc = &r["structuredContent"];
                let receipt = sc["receipt"]["hash"].as_str().map(str::to_string);
                let st = if r["isError"] == json!(true) {
                    let mut diagnostics: Vec<String> = sc["diagnostics"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect();
                    if let Some(e) = sc["error"].as_str() {
                        diagnostics.push(e.into());
                    }
                    ProposalStatus::Refused {
                        diagnostics,
                        receipt,
                    }
                } else {
                    ProposalStatus::Accepted {
                        receipt: receipt.unwrap_or_default(),
                    }
                };
                self.set_status(id, st);
                if p.path == self.editor.path {
                    let path = p.path.clone();
                    self.open(&path);
                }
            }
            Action::Reject(id) => {
                let p = self
                    .proposals
                    .iter()
                    .find(|p| p.id == id && p.status == ProposalStatus::Pending)
                    .cloned()
                    .ok_or("no such pending proposal")?;
                let mut d = Draft::new(
                    ActionKind::Other("proposal.reject".into()),
                    "user",
                    self.server.policy_hash().to_string(),
                );
                d.t_start_ns = ide_energy_now();
                d.t_end_ns = d.t_start_ns;
                d.verdict = Verdict::Rejected {
                    reason: "rejected by the user".into(),
                };
                d.detail =
                    json!({"proposal": p.id, "path": p.path, "origin": p.origin, "edits": p.edits});
                let r = self.server.ledger.append(d).map_err(|e| e.to_string())?;
                self.set_status(id, ProposalStatus::Rejected { receipt: r.hash });
            }
            Action::Recall(q) => {
                self.memory = MemoryVm {
                    query: q.clone(),
                    ..Default::default()
                };
                match self
                    .server
                    .call("recall", &json!({"query": q, "limit": 10}))
                {
                    Ok(r) => {
                        let sc = &r["structuredContent"];
                        self.memory.gear = sc["found"].as_str().map(str::to_string);
                        self.memory.escalate = sc["escalate"] == json!(true);
                        self.memory.hits = sc["hits"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .map(|h| MemoryHit {
                                key: h["key"].as_str().unwrap_or("").into(),
                                score: h["score"].as_f64().unwrap_or(0.0),
                                detail: match (&h["def"], &h["fact"]) {
                                    (d, _) if d.is_object() => format!(
                                        "{} {} {}:{}",
                                        d["kind"].as_str().unwrap_or(""),
                                        d["name"].as_str().unwrap_or(""),
                                        d["file"].as_str().unwrap_or(""),
                                        d["line"]
                                    ),
                                    (_, f) if f.is_object() => f["text"]
                                        .as_str()
                                        .or(f["body"].as_str())
                                        .unwrap_or("")
                                        .chars()
                                        .take(120)
                                        .collect(),
                                    _ => String::new(),
                                },
                            })
                            .collect();
                    }
                    Err((_, m)) => self.memory.error = Some(m),
                }
            }
            Action::Net { key, allow } => {
                let mut a = Approvals::load(&self.approvals_path);
                a.decided.insert(key, allow);
                a.save(&self.approvals_path).map_err(|e| e.to_string())?;
            }
            // Typing is a local buffer; it reaches disk only via an accepted
            // propose_edit (set_file).
            Action::SetText(t) => {
                self.buffer.set_text(&t);
                self.rediagnose();
                self.sync_editor();
            }
            Action::Edit(op) => {
                let op = match op {
                    EditOp::Paste(None) => EditOp::Paste(self.clip.get()),
                    op => op,
                };
                // Copy/cut with no selection leave the clipboard alone.
                let copied = matches!(op, EditOp::Copy | EditOp::Cut)
                    .then(|| self.buffer.selected_text())
                    .flatten();
                let changed = self.buffer.apply(op);
                if let Some(t) = copied {
                    self.clip.set(&t);
                }
                if changed {
                    self.rediagnose();
                }
                self.sync_editor();
                return Ok(()); // no ledger/manifest re-read per keystroke
            }
            Action::Save => {
                if self.editor.dirty {
                    let path = self.editor.path.clone();
                    let n = self.editor.diagnostics.len();
                    self.propose(
                        &path,
                        "editor",
                        &format!("save buffer ({n} diagnostic(s) now)"),
                        vec![Edit::SetFile {
                            text: self.buffer.text(),
                        }],
                    );
                }
            }
            Action::Focus(f) => self.focus = f,
            Action::Refresh => {}
        }
        self.refresh();
        Ok(())
    }

    fn set_status(&mut self, id: u64, st: ProposalStatus) {
        if let Some(p) = self.proposals.iter_mut().find(|p| p.id == id) {
            p.status = st;
        }
    }

    /// Re-read the ledger and the network manifest and approvals.
    pub fn refresh(&mut self) {
        let rs = read_jsonl(&self.ledger_path).unwrap_or_default();
        let key = rs.first().and_then(|r| {
            let b: [u8; 32] = hex::decode(&r.signer).ok()?.try_into().ok()?;
            ide_receipt::VerifyingKey::from_bytes(&b).ok()
        });
        self.receipts = ReceiptsVm::from_receipts(&rs, key.as_ref());
        let m = NetManifest::load(&self.server.ws.root).unwrap_or_default();
        let a = Approvals::load(&self.approvals_path);
        self.net = NetVm {
            pending: m
                .net
                .iter()
                .filter(|c| !a.decided.contains_key(&c.key()))
                .map(|c| (c.key(), c.reason.clone()))
                .collect(),
            decided: a.decided.into_iter().collect(),
        };
    }
}

fn ide_energy_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
