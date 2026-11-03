//! Admission receipts (spec 10.15): the record of why a program was
//! admitted, written by `rce check write RECEIPT FILE...` and revalidated by
//! `rce check receipt RECEIPT FILE...`.
//!
//! A receipt is evidence of a past admission, never an authority:
//! revalidation recomputes admission from the program text, and no tool
//! reads a receipt to decide whether a program is admitted. The receipt is
//! canonical JSON (6.1.1) followed by one line end, so the same program,
//! rule set and checker give the same bytes. As with the break-even receipt
//! of D.11, everything a receipt states that can be derived is derived again
//! when it is read, and a receipt whose stored hash disagrees with what it
//! lists is invalid (AR-5).
//!
//! The binding `PROGRAM;RULES;CHECKER` names the three things a receipt is
//! bound to, in the form of an attestation (`WHO;YYYY-MM-DD;COVER`, 6.11):
//! the name-free program hash (10.8), the rule-set hash
//! (`admission::RuleSet::sha256`) and the checker commit.
use crate::admission::{self, RuleSet};
use crate::interp::canon_json;
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

/// The format of a receipt.
pub const FORMAT: &str = "rce-admission-receipt/1";

/// The members of a receipt, sorted.
const KEYS: [&str; 9] = ["attestations", "binding", "checker", "format", "mutation", "obligations", "observed", "program", "rules"];

/// The checker that admits: its commit, the specification draft and the
/// release label of the language (10.1, `rce version`).
pub fn checker() -> Value {
    json!({"commit": crate::GIT_COMMIT, "draft": crate::cli::SPEC_DRAFT, "release": crate::parse::LANGUAGE_RELEASE})
}

/// The receipt of a program (`files`, name and source, given together)
/// admitted under `rules`, as its canonical bytes; or, when it is not
/// admitted, the reports of each file that requires change.
pub fn write(rules: &RuleSet, files: &[(String, String)]) -> Result<String, String> {
    write_of(rules, files, &admission::program(rules, files, true))
}

/// `write`, from the admission `adm` of `files` under `rules`
/// (`admission::program` with the mutation gate).
pub fn write_of(rules: &RuleSet, files: &[(String, String)], adm: &Program) -> Result<String, String> {
    let v = record(rules, files, adm, &checker())?;
    Ok(canon_json(&v) + "\n")
}

/// The admission of a program, per file (`admission::program`).
pub type Program = [(String, Result<admission::Admitted, String>)];

fn record(rules: &RuleSet, files: &[(String, String)], adm: &Program, checker: &Value) -> Result<Value, String> {
    let mut bad = vec![];
    let mut mods = vec![];
    let mut obligations: Vec<Value> = vec![];
    let mut attestations: Vec<String> = vec![];
    let mut observed: BTreeMap<String, usize> = BTreeMap::new();
    let (mut judged, mut total) = (0usize, 0usize);
    for ((p, a), (_, src)) in adm.iter().zip(files) {
        let a = match a {
            Err(e) => {
                bad.push(format!("{p}: {e}"));
                continue;
            }
            Ok(a) if !a.errs.is_empty() => {
                bad.push(format!("{p}: {}", a.errs.join("; ")));
                continue;
            }
            Ok(a) => a,
        };
        let m0 = crate::parse::parse(src).map_err(|e| format!("{p}: {e}"))?;
        for r in &a.observed {
            if let Some(rule) = admission::rule_of(r) {
                *observed.entry(rule.to_string()).or_default() += 1;
            }
        }
        judged += a.mutants.0;
        total += a.mutants.1;
        let ms = a.measure.clone().unwrap_or_default();
        for (line, kind, hash) in crate::apidiff::obligation_hashes(&m0) {
            let mut o = Map::new();
            o.insert("kind".into(), json!(kind));
            o.insert("sha256".into(), json!(hash));
            if kind == "law" {
                match ms.decided.iter().find(|d| d.line == line).map(|d| (&d.verdict, &d.cover)) {
                    Some((crate::law::Verdict::Holds { states, steps }, _)) => {
                        o.insert("result".into(), json!("holds"));
                        o.insert("states".into(), json!(states));
                        o.insert("steps".into(), json!(steps));
                    }
                    Some((crate::law::Verdict::Attested { who, date, .. }, cover)) => {
                        let at = format!("{who};{date};{cover}");
                        o.insert("result".into(), json!("attested"));
                        o.insert("attestation".into(), json!(at));
                        attestations.push(at);
                    }
                    _ => return Err(format!("{p}: line {line}: a law of an admitted program has no verdict that admits it")),
                }
            } else {
                o.insert("result".into(), json!("holds"));
                o.insert("fuel".into(), json!(ms.fuel.get(&line).copied().unwrap_or(0)));
                // W-UI-1: the frames the example's run derived and ui.rules judged
                if kind == "example"
                    && let Some(n) = ms.frames.get(&line)
                {
                    o.insert("frames".into(), json!(n));
                }
            }
            if let Some(r) = a.mutation.iter().find(|r| r.line == line) {
                o.insert("mutation".into(), json!({"measured": r.measured, "reach": r.reach, "killed": r.killed}));
            }
            obligations.push(Value::Object(o));
        }
        mods.push(m0);
    }
    if !bad.is_empty() {
        return Err(bad.join("\n"));
    }
    // order-free, as the program hash is: obligations by their canonical
    // text, and one entry for an obligation written twice
    obligations.sort_by_key(canon_json);
    obligations.dedup();
    attestations.sort();
    attestations.dedup();
    let program = crate::apidiff::program_hash(&mods);
    let rules_sha = rules.sha256();
    let binding = format!("{program};{rules_sha};{}", checker["commit"].as_str().unwrap_or(""));
    Ok(json!({
        "format": FORMAT,
        "binding": binding,
        "program": program,
        "checker": checker,
        "rules": {"sha256": rules_sha, "table": rules.table()},
        "obligations": obligations,
        "mutation": {"judged": judged, "total": total},
        "attestations": attestations,
        "observed": observed,
    }))
}

/// What revalidating a receipt found (10.15, AR-4). `admitted` is the
/// verdict of admission recomputed now, under the rules in force.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// the program, the rule set and the checker are those of the receipt,
    /// and the recomputed receipt has its bytes
    Holds,
    /// the program's name-free hash differs
    ProgramChanged { old: String, new: String, admitted: bool },
    /// the rule set differs: the rules added, removed and changed (method
    /// or mode), from the receipt's set to the set in force
    RulesChanged { old: String, new: String, added: Vec<String>, removed: Vec<String>, changed: Vec<String>, admitted: bool },
    /// the checker differs; `same` when the recomputed record, but for the
    /// checker, equals the receipt's
    CheckerChanged { old: String, new: String, admitted: bool, same: bool },
    /// program, rules and checker are the receipt's, and the record the
    /// receipt holds is not the one admission gives
    Differs { admitted: bool },
}

impl Outcome {
    pub fn holds(&self) -> bool {
        *self == Outcome::Holds
    }
}

/// A receipt read back and checked against itself (AR-5): the format, the
/// members, canonical bytes, a rule table whose hash is the one it states,
/// and a binding made of its program, rule-set hash and checker commit.
pub struct Read {
    pub value: Value,
    pub rules: RuleSet,
}

pub fn read(text: &str) -> Result<Read, String> {
    let v: Value = serde_json::from_str(text).map_err(|e| format!("AR-5 the receipt is not JSON: {e}"))?;
    let o = v.as_object().ok_or("AR-5 the receipt is not a JSON object")?;
    if o.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(format!("AR-5 the receipt's format is not {FORMAT}"));
    }
    let keys: Vec<&str> = o.keys().map(String::as_str).collect();
    if keys != KEYS {
        return Err(format!("AR-5 the receipt's members are {keys:?}, not {KEYS:?}"));
    }
    if canon_json(&v) + "\n" != text {
        return Err("AR-5 the receipt's bytes are not its canonical JSON followed by one line end".into());
    }
    let r = &v["rules"];
    let rules = RuleSet::from_table(&r["table"]).ok_or("AR-5 the receipt's rule table has a row that is not [RULE, METHOD] or [RULE, \"altered\", MODE]")?;
    if r["sha256"].as_str() != Some(rules.sha256().as_str()) {
        return Err("AR-5 the receipt's rule-set hash is not the hash of the rules it lists".into());
    }
    let binding = format!("{};{};{}", v["program"].as_str().unwrap_or(""), rules.sha256(), v["checker"]["commit"].as_str().unwrap_or(""));
    if v["binding"].as_str() != Some(binding.as_str()) {
        return Err("AR-5 the receipt's binding is not PROGRAM;RULES;CHECKER of its own members".into());
    }
    Ok(Read { value: v, rules })
}

/// Revalidate the receipt `text` against the program `files` under the
/// rules in force `rules` (AR-4). Admission is recomputed in every case;
/// the receipt decides nothing. Err when the receipt is invalid (AR-5).
pub fn revalidate(rules: &RuleSet, text: &str, files: &[(String, String)]) -> Result<Outcome, String> {
    revalidate_of(rules, text, files, &admission::program(rules, files, true))
}

/// `revalidate`, from the admission `adm` of `files` under `rules`.
pub fn revalidate_of(rules: &RuleSet, text: &str, files: &[(String, String)], adm: &Program) -> Result<Outcome, String> {
    let got = read(text)?;
    let stored = &got.value;
    let now = record(rules, files, adm, &checker());
    let admitted = now.is_ok();
    let parsed: Result<Vec<_>, String> = files.iter().map(|(_, s)| crate::parse::parse(s)).collect();
    let program = match &parsed {
        Ok(ms) => crate::apidiff::program_hash(ms),
        Err(_) => "(does not parse)".to_string(),
    };
    let old_program = stored["program"].as_str().unwrap_or("").to_string();
    if old_program != program {
        return Ok(Outcome::ProgramChanged { old: old_program, new: program, admitted });
    }
    if got.rules != *rules {
        let (added, removed, changed) = rules.diff(&got.rules);
        return Ok(Outcome::RulesChanged { old: got.rules.sha256(), new: rules.sha256(), added, removed, changed, admitted });
    }
    if stored["checker"] != checker() {
        let show = |c: &Value| format!("{} (draft {}, release {})", c["commit"].as_str().unwrap_or("?"), c["draft"].as_str().unwrap_or("?"), c["release"].as_str().unwrap_or("?"));
        // the record recomputed by this checker, but bound to the old one
        let same = record(rules, files, adm, &stored["checker"]).is_ok_and(|v| v == *stored);
        return Ok(Outcome::CheckerChanged { old: show(&stored["checker"]), new: show(&checker()), admitted, same });
    }
    match now {
        Ok(v) if v == *stored => Ok(Outcome::Holds),
        _ => Ok(Outcome::Differs { admitted }),
    }
}

/// The lines `rce check receipt` prints for an outcome.
pub fn show(receipt: &str, o: &Outcome, stored: &Value) -> Vec<String> {
    let verdict = |a: bool| if a { "the program is admitted" } else { "the program requires change" };
    match o {
        Outcome::Holds => vec![format!(
            "receipt holds {receipt}: program {} admitted under rules {} by checker {}",
            stored["program"].as_str().unwrap_or(""),
            stored["rules"]["sha256"].as_str().unwrap_or(""),
            stored["checker"]["commit"].as_str().unwrap_or("")
        )],
        Outcome::ProgramChanged { old, new, admitted } => {
            vec![format!("receipt does not hold {receipt}: the program changed (program {old} -> {new}); under the rules in force {}", verdict(*admitted))]
        }
        Outcome::RulesChanged { old, new, added, removed, changed, admitted } => {
            let mut v = vec![format!("receipt does not hold {receipt}: the rule set changed (rules {old} -> {new}); under the new set {}", verdict(*admitted))];
            v.extend(added.iter().map(|r| format!("    added {r}")));
            v.extend(removed.iter().map(|r| format!("    removed {r}")));
            v.extend(changed.iter().map(|r| format!("    changed {r}")));
            v
        }
        Outcome::CheckerChanged { old, new, admitted, same } => vec![format!(
            "receipt does not hold {receipt}: the checker changed (checker {old} -> {new}); {}, and the record it gives {} the receipt's",
            verdict(*admitted),
            if *same { "is" } else { "is not" }
        )],
        Outcome::Differs { admitted } => vec![format!(
            "receipt does not hold {receipt}: its record is not the one admission gives for the same program, rules and checker; {}",
            verdict(*admitted)
        )],
    }
}
