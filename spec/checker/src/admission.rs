//! Admission as one piece of code that every tool shares (spec 6.9, 6.11,
//! 6.12, 7.1, 10.2, and W-UI-1 of Appendix E, HC-29): `rce check`, `rce test`, `rce index`, `rce-check` and
//! the language server all reach a verdict through [`module`], so a text
//! one of them admits is admitted by all (10.1, principle 3), and the
//! admission receipt of 10.15 (`receipt.rs`) is written from what this code
//! measured.
//!
//! * [`RuleSet`]: the rule set in force, the table of 7.1 (every rule with
//!   its method) and the modes of its altered rules, read from the compiled
//!   spec data. Its hash is in every receipt.
//! * [`Mode`]: the mode of an altered rule. `auto`: `rce fmt` writes the
//!   rewrite and, until it is written, the text requires change. `observe`:
//!   `rce fmt` records the rewrite it would write and changes nothing; the
//!   report does not block admission and `rce check` prints it as
//!   `observed`. `off`: the rule is not judged. A rule whose method is not
//!   altered has no mode, so no table can switch it off.
//! * [`Attention`]: what a run leaves for a person (7.1): the requires-change
//!   and recommended findings, the alterations applied and those recorded
//!   in observe mode. Only requires-change findings make the status non-zero.
use crate::ast::Module;
use crate::interp::Measure;
use crate::link::Linked;
use crate::mutants::MutRow;
use std::collections::{BTreeMap, HashSet};
use std::sync::LazyLock;

/// The mode of a rule whose method is altered (7.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Off,
    Observe,
    Auto,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Mode> {
        match s {
            "off" => Some(Mode::Off),
            "observe" => Some(Mode::Observe),
            "auto" => Some(Mode::Auto),
            _ => None,
        }
    }
    pub fn text(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::Observe => "observe",
            Mode::Auto => "auto",
        }
    }
}

/// One row of the rule set: the method, and the mode of an altered rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub method: String,
    pub mode: Option<Mode>,
}

impl Row {
    /// `requires change`, `recommended`, or `altered MODE`.
    pub fn text(&self) -> String {
        match self.mode {
            Some(m) => format!("{} {}", self.method, m.text()),
            None => self.method.clone(),
        }
    }
}

/// A rule set: every rule with its method and, when altered, its mode.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct RuleSet {
    pub rows: BTreeMap<String, Row>,
}

/// The format line of the rule-set hash.
pub const RULES_FORMAT: &str = "rce-rules/1";

static IN_FORCE: LazyLock<RuleSet> = LazyLock::new(|| {
    let mut rows = BTreeMap::new();
    for (rule, method, mode) in crate::specread::rule_table() {
        // an altered rule the modes table does not list enters observed
        let mode = if method == "altered" { Some(mode.as_deref().and_then(Mode::parse).unwrap_or(Mode::Observe)) } else { None };
        rows.insert(rule, Row { method, mode });
    }
    RuleSet { rows }
});

impl RuleSet {
    /// The rule set compiled into this checker: the table of 7.1 and its
    /// modes, from `spec/data/spec-tables.json`.
    pub fn in_force() -> &'static RuleSet {
        &IN_FORCE
    }

    /// The mode of rule `rule`: None when its method is not altered.
    pub fn mode(&self, rule: &str) -> Option<Mode> {
        self.rows.get(rule).and_then(|r| r.mode)
    }

    /// The rule set with the mode of one altered rule changed (a promotion
    /// or a demotion). A rule whose method is not altered keeps no mode.
    pub fn with_mode(&self, rule: &str, mode: Mode) -> RuleSet {
        let mut s = self.clone();
        if let Some(r) = s.rows.get_mut(rule)
            && r.method == "altered"
        {
            r.mode = Some(mode);
        }
        s
    }

    /// The rows as the receipt lists them: `[RULE, METHOD]`, or
    /// `[RULE, "altered", MODE]`, sorted by rule.
    pub fn table(&self) -> serde_json::Value {
        serde_json::Value::Array(
            self.rows
                .iter()
                .map(|(k, r)| {
                    let mut v = vec![serde_json::json!(k), serde_json::json!(r.method)];
                    if let Some(m) = r.mode {
                        v.push(serde_json::json!(m.text()));
                    }
                    serde_json::Value::Array(v)
                })
                .collect(),
        )
    }

    /// The rule set read back from a receipt's `table`; None when a row is
    /// not `[RULE, METHOD]` or `[RULE, "altered", MODE]`.
    pub fn from_table(v: &serde_json::Value) -> Option<RuleSet> {
        let mut rows = BTreeMap::new();
        for r in v.as_array()? {
            let a = r.as_array()?;
            let rule = a.first()?.as_str()?.to_string();
            let method = a.get(1)?.as_str()?.to_string();
            if !["altered", "recommended", "requires change"].contains(&method.as_str()) {
                return None;
            }
            let mode = match (method.as_str(), a.len()) {
                ("altered", 3) => Some(Mode::parse(a[2].as_str()?)?),
                (_, 2) if method != "altered" => None,
                _ => return None,
            };
            if rows.insert(rule, Row { method, mode }).is_some() {
                return None;
            }
        }
        Some(RuleSet { rows })
    }

    /// The SHA-256 of the canonical JSON of [`RuleSet::table`], under the
    /// domain `rce-rules/1`: a change of any rule's method or mode, and an
    /// added or removed rule, changes it.
    pub fn sha256(&self) -> String {
        crate::manifest::sha256_hex(format!("{RULES_FORMAT}\n{}", crate::interp::canon_json(&self.table())).as_bytes())
    }

    /// What changed from `old` to `self`: (added, removed, changed), each a
    /// line `RULE METHOD` or `RULE OLD -> NEW`.
    pub fn diff(&self, old: &RuleSet) -> (Vec<String>, Vec<String>, Vec<String>) {
        let (mut add, mut rem, mut chg) = (vec![], vec![], vec![]);
        for (k, r) in &self.rows {
            match old.rows.get(k) {
                None => add.push(format!("{k} {}", r.text())),
                Some(o) if o != r => chg.push(format!("{k} {} -> {}", o.text(), r.text())),
                _ => {}
            }
        }
        for (k, o) in &old.rows {
            if !self.rows.contains_key(k) {
                rem.push(format!("{k} {}", o.text()));
            }
        }
        (add, rem, chg)
    }
}

/// The rule a report names: `line N: W-RULE-1 ...` gives `W-RULE-1`.
pub fn rule_of(report: &str) -> Option<&str> {
    let r = report.split_once(": ")?.1;
    let code = r.split(' ').next()?;
    code.starts_with("W-").then_some(code)
}

/// The static reports of a text split by the modes of `rules`: the reports
/// that require change, and those of a rule in observe mode, which do not
/// block admission. A report of a rule that is off is dropped.
pub fn by_mode(rules: &RuleSet, errs: Vec<String>) -> (Vec<String>, Vec<String>) {
    let (mut req, mut obs) = (vec![], vec![]);
    for e in errs {
        match rule_of(&e).and_then(|r| rules.mode(r)) {
            Some(Mode::Off) => {}
            Some(Mode::Observe) => obs.push(e),
            _ => req.push(e),
        }
    }
    (req, obs)
}

/// What admission found and measured of one module.
pub struct Admitted {
    /// the reports that require change; empty means admitted
    pub errs: Vec<String>,
    /// the reports of rules in observe mode (7.1): recorded, not blocking
    pub observed: Vec<String>,
    /// what admission measured of each obligation (6.9, 6.11), when it ran
    pub measure: Option<Measure>,
    /// what the mutation gate measured of each obligation (6.12), when it ran
    pub mutation: Vec<MutRow>,
    /// the mutant set: those judged and the whole set
    pub mutants: (usize, usize),
    /// the module as checked, with its imports linked in
    pub linked: Module,
}

/// Type check and admit one module of a program under `rules`: the one
/// admission of every tool. `mods` are the modules given together (4.1),
/// `known` their paths, and `mutation` runs the mutation gate (W-MUT-1).
pub fn module(rules: &RuleSet, m: &Module, mods: &[&Module], known: &HashSet<String>, mutation: bool) -> Admitted {
    let mut lk = crate::link::link(m, mods);
    let mut c = crate::check::Checker::new();
    c.imported_fns = lk.imported_fns.clone();
    c.imported_types = lk.imported_types.clone();
    for e in std::mem::take(&mut lk.errs) {
        c.e(e);
    }
    c.check_module(&lk.module, known);
    linked(rules, lk, std::mem::take(&mut c.errs), m.items.len(), known, mutation)
}

/// Admission of a linked module whose static reports are `errs`, under
/// `rules`: the reports of rules in observe mode are set apart and those of
/// rules that are off dropped; when no report is left, admission (6.9,
/// 6.11), then the frames of a module that emits them held to `ui.rules`
/// (W-UI-1, HC-29), and then, when `mutation` is true and every obligation
/// holds, the mutation gate (6.12) on the first `own` items run.
pub fn linked(rules: &RuleSet, mut lk: Linked, errs: Vec<String>, own: usize, known: &HashSet<String>, mutation: bool) -> Admitted {
    let (mut errs, observed) = by_mode(rules, errs);
    let (mut measure, mut rows, mut mutants) = (None, vec![], (0, 0));
    if errs.is_empty() {
        let sites = if mutation { crate::mutants::site_addresses(&mut lk.module, own) } else { crate::mutants::Sites { addrs: vec![], emitted: Default::default() } };
        let (e, mut ms) = crate::interp::admit_measured(&lk.module);
        errs = e;
        // W-UI-1 (7, HC-29): the frames of a module that emits them are
        // held to ui.rules once its obligations hold
        if errs.is_empty() {
            errs = crate::frame::gate(&lk.module, own, &mut ms.frames);
        }
        if mutation && errs.is_empty() {
            let p = crate::mutants::Program { linked: &lk.module, own, imported_fns: &lk.imported_fns, imported_types: &lk.imported_types, known };
            let g = crate::mutants::gate_measured(&p, &sites, &ms);
            errs = g.errs;
            rows = g.rows;
            mutants = (g.judged, g.total);
        }
        measure = Some(ms);
    }
    Admitted { errs, observed, measure, mutation: rows, mutants, linked: lk.module }
}

/// Every module of a program (`files`, name and source) parsed, checked and
/// admitted under `rules`, in the order given: per file, its admission or
/// the parse report.
pub fn program(rules: &RuleSet, files: &[(String, String)], mutation: bool) -> Vec<(String, Result<Admitted, String>)> {
    let mut parsed = vec![];
    let mut out: Vec<(String, Result<Admitted, String>)> = vec![];
    for (p, src) in files {
        match crate::parse::parse(src) {
            Ok(m) => parsed.push((p.clone(), m)),
            Err(e) => out.push((p.clone(), Err(e))),
        }
    }
    let known: HashSet<String> = parsed.iter().filter_map(|(_, m)| m.path.as_ref().map(|q| q.join("."))).collect();
    let mods: Vec<&Module> = parsed.iter().map(|(_, m)| m).collect();
    for (p, m) in &parsed {
        out.push((p.clone(), Ok(module(rules, m, &mods, &known, mutation))));
    }
    let pos = |n: &String| files.iter().position(|(p, _)| p == n).unwrap_or(usize::MAX);
    out.sort_by_key(|(n, _)| pos(n));
    out
}

/// What a run of `rce check` or `rce fmt` leaves for a person (7.1): the
/// findings that require change, the recommended findings, the alterations
/// written and the alterations recorded in observe mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Attention {
    pub requires_change: usize,
    pub recommended: usize,
    pub altered: usize,
    pub observed: usize,
}

impl Attention {
    /// The summary line every run of `rce check` and `rce fmt` ends with,
    /// on standard error.
    pub fn line(&self) -> String {
        format!(
            "attention: {} requires change, {} recommended, {} altered, {} observed",
            self.requires_change, self.recommended, self.altered, self.observed
        )
    }
    /// The exit status: 1 when a finding requires change, else 0. A
    /// recommended finding, an alteration written and one recorded never
    /// change it.
    pub fn status(&self) -> i32 {
        if self.requires_change > 0 { crate::cli::REQUIRES_CHANGE } else { crate::cli::OK }
    }
}
