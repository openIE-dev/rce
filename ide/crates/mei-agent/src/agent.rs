//! The driver: runs the turn chart against the workspace.
use crate::catalog::{Catalog, GearTier, Provider};
use crate::chart::{self, GEARS};
use crate::http::{self, SystemOne};
use ide_energy::sim::{Features, Simulator};
use ide_receipt::{ActionKind, Draft, Measured, MeterSource, Verdict};
use mei_machine::ChartActor;
use rce_mcp::Server;
use rce_mcp::items::Edit;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::time::Duration;

/// What the agent is asked to do, in one file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Task {
    pub path: String,
    pub goal: String,
    /// Deterministic definition of done. Without it, done is asked of the
    /// System-One model (when configured), and is otherwise "not done".
    pub acceptance: Option<Acceptance>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Acceptance {
    /// Top-level item names the file must contain.
    pub items: Vec<String>,
    /// Minimum number of `example` lines in the file.
    pub min_examples: usize,
}

/// A G1 rule: if the goal contains `when` (case-insensitive), propose `edits`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Rule {
    pub when: String,
    pub edits: Vec<Edit>,
}

/// What an external G4 turn reported back.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExternalTurn {
    /// Hash of the external backend's own signed turn receipt, if any.
    pub receipt: Option<String>,
    /// Files it changed through rce-mcp (already checker-gated).
    pub mediated: Vec<String>,
    /// Summary for this agent's receipt detail.
    pub detail: Value,
    /// Joules measured on this machine during the turn, with source.
    pub e_meas: Option<Measured>,
}

/// An optional G4: an outside agent that edits through rce-mcp itself
/// (e.g. an ACP agent, see the `acp-client` crate). After its turn the
/// workspace is checked here and done is decided as for any gear.
pub trait ExternalG4 {
    fn name(&self) -> String;
    fn turn(&mut self, prompt: &str) -> Result<ExternalTurn, String>;
}

pub struct AgentConfig {
    pub budget_j: f64,
    /// Joules a gear must have left to be entered (index = gear). A
    /// configured floor, used only by the budget guard.
    pub gear_floor_j: [f64; 5],
    pub systemone: Option<SystemOne>,
    pub catalog: Catalog,
    pub rules: Vec<Rule>,
    pub timeout: Duration,
    /// Used at G4 instead of a catalog provider when set.
    pub external_g4: Option<Box<dyn ExternalG4>>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        AgentConfig {
            budget_j: 0.0,
            gear_floor_j: [0.0; 5],
            systemone: None,
            catalog: Catalog::default(),
            rules: vec![],
            timeout: Duration::from_secs(60),
            external_g4: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Status {
    Accepted { gear: usize },
    Failed { gear: usize, reason: String },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Outcome {
    pub status: Status,
    pub gears_tried: Vec<usize>,
    /// Joules charged against the budget (measured where a meter existed,
    /// else predicted, else a remote prior; see each receipt).
    pub charged_j: f64,
    /// Steps where nothing could be charged (no meter, no calibration, no prior).
    pub uncharged_steps: usize,
    pub receipts: Vec<String>,
}

pub struct Agent {
    pub server: Server,
    pub cfg: AgentConfig,
    charged_j: f64,
    uncharged: usize,
    receipts: Vec<String>,
}

/// What one gear concluded.
enum Concl {
    Done,
    NotDone(String),
    Reject(Vec<String>),
}

const SYSTEM_PROMPT: &str = "You edit one RCE source file. Reply with JSON only: {\"edits\": [...]}. \
Each edit is {\"op\":\"replace_item\",\"name\":N,\"text\":T}, {\"op\":\"insert_item\",\"after\":N?,\"text\":T}, \
{\"op\":\"delete_item\",\"name\":N} or {\"op\":\"set_file\",\"text\":T}. Items are addressed by name \
(the header line up to '(' or '->' or '=' or ':'). The checker decides; refused edits are not applied.";

/// Pull the edits out of a model reply (bare JSON or a fenced block).
pub fn parse_edits(content: &str) -> Result<Vec<Edit>, String> {
    let t = content.trim();
    let t = t
        .strip_prefix("```json")
        .or_else(|| t.strip_prefix("```"))
        .unwrap_or(t);
    let t = t.strip_suffix("```").unwrap_or(t).trim();
    let v: Value = serde_json::from_str(t).map_err(|e| format!("reply is not JSON: {e}"))?;
    serde_json::from_value(v.get("edits").cloned().ok_or("reply has no `edits`")?)
        .map_err(|e| format!("edits: {e}"))
}

impl Agent {
    pub fn new(server: Server, cfg: AgentConfig) -> Self {
        Agent {
            server,
            cfg,
            charged_j: 0.0,
            uncharged: 0,
            receipts: vec![],
        }
    }

    fn remaining(&self) -> f64 {
        self.cfg.budget_j - self.charged_j
    }

    /// Charge a step: the measurement if any, else the prediction.
    fn charge(&mut self, e_sim: Option<f64>, e_meas: Option<Measured>) {
        match (e_meas, e_sim) {
            (Some(m), _) => self.charged_j += m.joules,
            (None, Some(s)) => self.charged_j += s,
            (None, None) => self.uncharged += 1,
        }
    }

    /// Write an agent receipt and charge it.
    fn receipt(&mut self, mut d: Draft) -> Result<(), String> {
        if d.e_sim_j.is_none() {
            let f: Features =
                serde_json::from_value(d.detail["features"].clone()).unwrap_or_default();
            d.e_sim_j = self.server.sim.predict(&f).map(|p| p.joules);
        }
        self.charge(d.e_sim_j, d.e_meas);
        let r = self.server.ledger.append(d).map_err(|e| e.to_string())?;
        self.receipts.push(r.hash);
        Ok(())
    }

    fn step(&mut self, kind: ActionKind, gear: usize, detail: Value) -> Draft {
        let mut d = Draft::new(
            kind,
            format!("g{gear}"),
            self.server.policy_hash().to_string(),
        );
        d.t_start_ns = ide_energy::meter::now_ns();
        d.t_end_ns = d.t_start_ns;
        d.detail = detail;
        d
    }

    /// Call an rce-mcp tool; its receipt is written by the server. Charge it here.
    fn tool(&mut self, name: &str, args: Value) -> Result<(bool, Value), String> {
        let r = self.server.call(name, &args).map_err(|(_, m)| m)?;
        let sc = r["structuredContent"].clone();
        let rc = &sc["receipt"];
        let e_meas: Option<Measured> = serde_json::from_value(rc["e_meas"].clone()).ok().flatten();
        self.charge(rc["e_sim_j"].as_f64(), e_meas);
        if let Some(h) = rc["hash"].as_str() {
            self.receipts.push(h.to_string());
        }
        Ok((r["isError"] == json!(true), sc))
    }

    fn send(a: &mut ChartActor<'_>, ev: &str) -> Result<(), String> {
        if a.send(ev) {
            Ok(())
        } else {
            Err(format!("chart refused `{ev}` in {:?}", a.configuration()))
        }
    }

    fn child(a: &mut ChartActor<'_>, ev: &str) -> Result<(), String> {
        if a.send_to_child(ev) {
            Ok(())
        } else {
            Err(format!("gear chart refused `{ev}`"))
        }
    }

    /// Deterministic done check, when the task defines one.
    fn accepted(&self, task: &Task) -> Option<bool> {
        let acc = task.acceptance.as_ref()?;
        let p = self.server.ws.resolve(&task.path, false).ok()?;
        let src = std::fs::read_to_string(p).ok()?;
        let names: Vec<String> = rce_mcp::items::items(&src)
            .into_iter()
            .map(|i| i.name)
            .collect();
        let items_ok = acc
            .items
            .iter()
            .all(|w| names.iter().any(|n| *n == rce_mcp::items::item_name(w)));
        let ex = rce_spec_check_examples(&src);
        Some(items_ok && ex >= acc.min_examples)
    }

    /// Done after a passing check: acceptance, else System-One, else not done.
    fn done(&mut self, task: &Task, gear: usize) -> Result<Concl, String> {
        if let Some(ok) = self.accepted(task) {
            return Ok(if ok {
                Concl::Done
            } else {
                Concl::NotDone("acceptance not met".into())
            });
        }
        match self.ask_done(task, gear)? {
            Some(true) => Ok(Concl::Done),
            Some(false) => Ok(Concl::NotDone("System-One: not done".into())),
            None => Ok(Concl::NotDone(
                "no acceptance and no System-One model to decide done".into(),
            )),
        }
    }

    /// One System-One `noul` call. `None` if none is configured or it failed.
    fn ask_done(&mut self, task: &Task, gear: usize) -> Result<Option<bool>, String> {
        let Some(s1) = self.cfg.systemone.clone() else {
            return Ok(None);
        };
        let src = self
            .server
            .ws
            .resolve(&task.path, false)
            .ok()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();
        let state = json!({"goal": task.goal, "file": task.path, "source": src});
        let mut d = self.step(
            ActionKind::Other("systemone.decide".into()),
            gear,
            json!({}),
        );
        let r = s1.noul(
            &state,
            "done",
            "Does the source already do what the goal asks?",
        );
        d.t_end_ns = ide_energy::meter::now_ns();
        d.model = s1.model.clone();
        let out = match &r {
            Ok(n) => {
                d.tokens_in = n.input_tokens;
                d.detail = json!({"base_url": s1.base_url, "p_true": n.p_true, "confidence": n.confidence,
                    "features": Features { gear: format!("g{gear}"), model: s1.model.clone(), tokens_in: n.input_tokens, ..Default::default() }});
                Some(n.p_true > 0.5)
            }
            Err(e) => {
                d.verdict = Verdict::Rejected {
                    reason: e.to_string(),
                };
                d.detail = json!({"base_url": s1.base_url, "error": e.to_string()});
                None
            }
        };
        self.receipt(d)?;
        Ok(out)
    }

    fn check_workspace(&mut self, path: &str) -> Result<Concl, String> {
        let (err, sc) = self.tool("rce_check", json!({"paths": [path]}))?;
        if err {
            let d: Vec<String> = sc["files"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|f| f["diagnostic"].as_str().map(str::to_string))
                .collect();
            return Ok(Concl::Reject(d));
        }
        Ok(Concl::Done)
    }

    fn apply(&mut self, task: &Task, edits: &[Edit]) -> Result<Option<Vec<String>>, String> {
        let edits_json: Vec<Value> = edits.iter().map(edit_json).collect();
        let (err, sc) = self.tool(
            "propose_edit",
            json!({"path": task.path, "edits": edits_json}),
        )?;
        if err {
            let mut d: Vec<String> = sc["diagnostics"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect();
            if let Some(e) = sc["error"].as_str() {
                d.push(e.to_string());
            }
            return Ok(Some(d));
        }
        Ok(None)
    }

    fn gear0(&mut self, a: &mut ChartActor<'_>, task: &Task) -> Result<Concl, String> {
        Self::child(a, "run_check")?;
        let c = self.check_workspace(&task.path)?;
        Self::child(a, "run_tests")?;
        self.tool("rce_test", json!({"paths": [task.path]}))?;
        match c {
            Concl::Reject(d) => Ok(Concl::Reject(d)),
            _ => match self.accepted(task) {
                Some(true) => Ok(Concl::Done),
                Some(false) => Ok(Concl::NotDone("acceptance not met".into())),
                None => Ok(Concl::NotDone(
                    "G0 cannot decide done without acceptance".into(),
                )),
            },
        }
    }

    fn gear1(&mut self, a: &mut ChartActor<'_>, task: &Task) -> Result<Concl, String> {
        let goal = task.goal.to_lowercase();
        let hit = self
            .cfg
            .rules
            .iter()
            .position(|r| goal.contains(&r.when.to_lowercase()));
        let mut d = self.step(
            ActionKind::Other("rules.lookup".into()),
            1,
            json!({"rule": hit, "rules": self.cfg.rules.len()}),
        );
        d.verdict = Verdict::NotRun;
        self.receipt(d)?;
        let Some(i) = hit else {
            Self::child(a, "lookup_miss")?;
            return Ok(Concl::NotDone("no rule matches".into()));
        };
        Self::child(a, "lookup_hit")?;
        let edits = self.cfg.rules[i].edits.clone();
        match self.apply(task, &edits)? {
            Some(diags) => Ok(Concl::Reject(diags)),
            None => self.done(task, 1),
        }
    }

    fn gear2(&mut self, a: &mut ChartActor<'_>, task: &Task) -> Result<Concl, String> {
        if let Concl::Reject(d) = self.check_workspace(&task.path)? {
            Self::child(a, "failed")?;
            return Ok(Concl::Reject(d));
        }
        if let Some(ok) = self.accepted(task) {
            Self::child(a, "failed")?;
            return Ok(if ok {
                Concl::Done
            } else {
                Concl::NotDone("acceptance not met; G2 does not edit".into())
            });
        }
        Self::child(a, "ask")?;
        match self.ask_done(task, 2)? {
            Some(true) => {
                Self::child(a, "answered")?;
                Ok(Concl::Done)
            }
            Some(false) => {
                Self::child(a, "answered")?;
                Ok(Concl::NotDone("System-One: not done".into()))
            }
            None => {
                Self::child(a, "failed")?;
                Ok(Concl::NotDone("System-One unavailable".into()))
            }
        }
    }

    fn model_gear(
        &mut self,
        a: &mut ChartActor<'_>,
        task: &Task,
        gear: usize,
        last: &[String],
    ) -> Result<Concl, String> {
        if gear == 4 && self.cfg.external_g4.is_some() {
            return self.external_gear(a, task, last);
        }
        let tier = if gear == 3 {
            GearTier::G3
        } else {
            GearTier::G4
        };
        let Some(p): Option<Provider> = self.cfg.catalog.pick(tier).cloned() else {
            Self::child(a, "failed")?;
            return Ok(Concl::NotDone(format!(
                "no {tier:?} provider in the catalog"
            )));
        };
        Self::child(a, "select")?;
        let key = match &p.key_env {
            Some(k) => match std::env::var(k) {
                Ok(v) if !v.is_empty() => Some(v),
                _ => {
                    Self::child(a, "failed")?;
                    let mut d = self.step(ActionKind::ModelCall, gear, json!({"provider": p.provider, "error": format!("key env `{k}` not set; no request sent")}));
                    d.model = Some(p.model_id.clone());
                    d.verdict = Verdict::Rejected {
                        reason: format!("missing key env {k}"),
                    };
                    self.receipt(d)?;
                    return Ok(Concl::NotDone(format!("{} key `{k}` not set", p.provider)));
                }
            },
            None => None,
        };
        let src = self
            .server
            .ws
            .resolve(&task.path, false)
            .ok()
            .and_then(|x| std::fs::read_to_string(x).ok())
            .unwrap_or_default();
        let user =
            json!({"goal": task.goal, "path": task.path, "source": src, "last_diagnostics": last})
                .to_string();
        Self::child(a, "request")?;
        let mut d = self.step(ActionKind::ModelCall, gear, json!({}));
        d.model = Some(p.model_id.clone());
        let span = self.server.meter.as_ref().and_then(|m| m.begin());
        let r = http::chat(
            &p.base_url,
            key.as_deref(),
            &p.model_id,
            SYSTEM_PROMPT,
            &user,
            self.cfg.timeout,
        );
        let jm = match (self.server.meter.as_ref(), span) {
            (Some(m), Some(sp)) => m.end(sp),
            _ => None,
        };
        d.t_end_ns = ide_energy::meter::now_ns();
        let chat = match r {
            Ok(c) => c,
            Err(e) => {
                Self::child(a, "failed")?;
                d.verdict = Verdict::Rejected {
                    reason: e.to_string(),
                };
                d.detail =
                    json!({"provider": p.provider, "base_url": p.base_url, "error": e.to_string()});
                self.receipt(d)?;
                return Ok(Concl::NotDone(format!("{}: {e}", p.provider)));
            }
        };
        Self::child(a, "response")?;
        d.tokens_in = chat.tokens_in;
        d.tokens_out = chat.tokens_out;
        let local = p.key_env.is_none();
        d.e_meas = match (
            local,
            jm,
            self.server.meter.as_ref().map(|m| m.source()),
        ) {
            // A local server on this machine: the hardware meter saw it.
            (true, Some(x), Some("rapl")) => Some(Measured {
                joules: x,
                source: MeterSource::Rapl,
            }),
            (true, Some(x), Some("nvml")) => Some(Measured {
                joules: x,
                source: MeterSource::Nvml,
            }),
            (true, Some(x), Some("powermetrics")) => Some(Measured {
                joules: x,
                source: MeterSource::Powermetrics,
            }),
            (true, Some(x), Some("ioreport")) => Some(Measured {
                joules: x,
                source: MeterSource::Ioreport,
            }),
            (true, Some(x), Some("smc")) => Some(Measured {
                joules: x,
                source: MeterSource::Smc,
            }),
            (true, Some(x), Some("hwmon")) => Some(Measured {
                joules: x,
                source: MeterSource::Hwmon,
            }),
            (true, Some(x), Some("power-supply")) => Some(Measured {
                joules: x,
                source: MeterSource::PowerSupply,
            }),
            (true, Some(x), Some("jetson")) => Some(Measured {
                joules: x,
                source: MeterSource::Jetson,
            }),
            // A remote provider: only the catalog prior, tagged as an estimate.
            (false, ..) => p
                .energy_prior_j_per_token
                .filter(|_| chat.usage_reported)
                .map(|j| Measured {
                    joules: j * (chat.tokens_in + chat.tokens_out) as f64,
                    source: MeterSource::EstimatedRemote,
                }),
            _ => None,
        };
        let cost_usd = match (
            p.price_in_per_mtok,
            p.price_out_per_mtok,
            chat.usage_reported,
        ) {
            (Some(i), Some(o), true) => {
                Some((i * chat.tokens_in as f64 + o * chat.tokens_out as f64) / 1e6)
            }
            _ => None,
        };
        d.detail = json!({"provider": p.provider, "base_url": p.base_url, "usage_reported": chat.usage_reported, "cost_usd": cost_usd,
            "features": Features { gear: format!("g{gear}"), model: Some(p.model_id.clone()), tokens_in: chat.tokens_in, tokens_out: chat.tokens_out,
                net_bytes: (user.len() + chat.content.len()) as u64, ..Default::default() }});
        let edits = parse_edits(&chat.content);
        if let Err(e) = &edits {
            d.verdict = Verdict::Rejected { reason: e.clone() };
        }
        self.receipt(d)?;
        let edits = match edits {
            Ok(e) => e,
            Err(e) => return Ok(Concl::Reject(vec![e])),
        };
        match self.apply(task, &edits)? {
            Some(diags) => Ok(Concl::Reject(diags)),
            None => self.done(task, gear),
        }
    }

    fn external_gear(
        &mut self,
        a: &mut ChartActor<'_>,
        task: &Task,
        last: &[String],
    ) -> Result<Concl, String> {
        Self::child(a, "select")?;
        let prompt = format!(
            "Goal: {}\nFile: {} (workspace-relative)\nUse only the rce MCP tools (read_form, propose_edit, rce_check, rce_test). \
Every change must go through propose_edit; the checker decides.\nLast diagnostics: {}",
            task.goal,
            task.path,
            if last.is_empty() {
                "none".to_string()
            } else {
                last.join("; ")
            }
        );
        Self::child(a, "request")?;
        let mut d = self.step(ActionKind::ModelCall, 4, json!({}));
        let ext = self.cfg.external_g4.as_mut().expect("checked");
        d.model = Some(ext.name());
        let r = ext.turn(&prompt);
        d.t_end_ns = ide_energy::meter::now_ns();
        match r {
            Err(e) => {
                Self::child(a, "failed")?;
                d.verdict = Verdict::Rejected { reason: e.clone() };
                d.detail = json!({"external": d.model, "error": e});
                self.receipt(d)?;
                Ok(Concl::NotDone(format!("external G4: {e}")))
            }
            Ok(t) => {
                Self::child(a, "response")?;
                d.e_meas = t.e_meas;
                d.detail = json!({"external": d.model, "turn_receipt": t.receipt, "mediated": t.mediated, "turn": t.detail,
                    "features": Features { gear: "g4".into(), model: d.model.clone(), ..Default::default() }});
                self.receipt(d)?;
                match self.check_workspace(&task.path)? {
                    Concl::Reject(diags) => Ok(Concl::Reject(diags)),
                    _ => self.done(task, 4),
                }
            }
        }
    }

    /// Run one turn to accept or fail.
    pub fn run(&mut self, task: &Task) -> Result<Outcome, String> {
        let chart = chart::turn_chart();
        let mut a = chart::actor(&chart);
        if self.remaining() < self.cfg.gear_floor_j[0] {
            Self::send(&mut a, "budget_spent")?;
        }
        Self::send(&mut a, "start")?;
        let mut tried = vec![];
        let mut last: Vec<String> = vec![];
        let reason;
        loop {
            if a.is_active("failed") {
                reason = "budget below the G0 floor".to_string();
                break;
            }
            let g = chart::gear_of(&a);
            tried.push(g);
            let c = match g {
                0 => self.gear0(&mut a, task)?,
                1 => self.gear1(&mut a, task)?,
                2 => self.gear2(&mut a, task)?,
                _ => self.model_gear(&mut a, task, g, &last)?,
            };
            Self::send(&mut a, "proposed")?;
            let why = match &c {
                Concl::Done => "done".to_string(),
                Concl::NotDone(w) => w.clone(),
                Concl::Reject(d) => format!("checker reject: {}", d.join("; ")),
            };
            Self::send(
                &mut a,
                match c {
                    Concl::Done => "set_done",
                    Concl::NotDone(_) => "set_not_done",
                    Concl::Reject(ref d) => {
                        last = d.clone();
                        "set_reject"
                    }
                },
            )?;
            let next_floor = self.cfg.gear_floor_j.get(g + 1).copied().unwrap_or(0.0);
            if a.is_active("budget_ok") && self.remaining() < next_floor.max(f64::MIN_POSITIVE) {
                Self::send(&mut a, "budget_spent")?;
            }
            Self::send(&mut a, "decide")?;
            let state = ["accept", "escalate", "failed"]
                .into_iter()
                .find(|s| a.is_active(s))
                .unwrap_or("?");
            let mut d = self.step(ActionKind::Other(format!("turn.{state}")), g, json!({
                "why": why, "remaining_j": self.remaining(), "configuration": a.configuration().states().collect::<Vec<_>>()}));
            d.verdict = match state {
                "accept" => Verdict::Pass,
                _ => Verdict::Rejected {
                    reason: why.clone(),
                },
            };
            self.receipt(d)?;
            match state {
                "accept" => {
                    return Ok(self.outcome(Status::Accepted { gear: g }, tried));
                }
                "escalate" => Self::send(&mut a, "next")?,
                _ => {
                    reason = if a.is_active("budget_out") {
                        format!("joule budget spent at {} ({why})", GEARS[g])
                    } else {
                        format!("no gear above {} ({why})", GEARS[g])
                    };
                    break;
                }
            }
        }
        let g = chart::gear_of(&a);
        Ok(self.outcome(Status::Failed { gear: g, reason }, tried))
    }

    fn outcome(&mut self, status: Status, gears_tried: Vec<usize>) -> Outcome {
        Outcome {
            status,
            gears_tried,
            charged_j: self.charged_j,
            uncharged_steps: self.uncharged,
            receipts: std::mem::take(&mut self.receipts),
        }
    }
}

fn rce_spec_check_examples(src: &str) -> usize {
    src.lines()
        .filter(|l| l.trim_start().starts_with("example "))
        .count()
}

fn edit_json(e: &Edit) -> Value {
    match e {
        Edit::ReplaceItem { name, text } => {
            json!({"op": "replace_item", "name": name, "text": text})
        }
        Edit::InsertItem { after, text } => {
            json!({"op": "insert_item", "after": after, "text": text})
        }
        Edit::DeleteItem { name } => json!({"op": "delete_item", "name": name}),
        Edit::SetFile { text } => json!({"op": "set_file", "text": text}),
    }
}
