//! Rung f of `rce repair` (spec 10.13, RF-1 to RF-7): candidate programs
//! written outside RCE, by a model (hosted or local) or by a person, enter
//! the ladder as a file of proposals. The checker calls no model. A proposal
//! is data: it carries an origin label, it is recorded by its digest, and it
//! is judged by the gate that judges every exact candidate (`check_program`,
//! which runs every example and check and then the mutation gate W-MUT-1).
//!
//! * [`read`]: the proposals file (RF-1), canonical JSON of format
//!   `rce-proposals/1`; each proposal's digest is the SHA-256 under
//!   `rce-proposal/1` of its text.
//! * [`mode`]: the mode of rung f, `off`, `observe` or `auto`, read from the
//!   modes table of 10.13 as the altered rules read theirs (7.1, RM-1 to
//!   RM-3). A rung the table does not list is `observe`, and rung f entered
//!   as `observe` (RF-3).
//! * [`evaluate`]: the promotion criterion registered in
//!   `spec/proposals/repair-observe/criterion.json` (RF-5), evaluated over
//!   observation records, one `rce repair json` report per task of the
//!   registered corpus (RF-6).
//! * [`promotion_check`]: the mode row of rung f is `auto` only when the
//!   recorded evaluation shows the criterion met (RF-7).
use crate::admission::Mode;
use crate::interp::canon_json;
use crate::manifest::sha256_hex;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// The format of a proposals file.
pub const FORMAT: &str = "rce-proposals/1";
/// The domain of a proposal's digest.
pub const DIGEST: &str = "rce-proposal/1";
/// The format of the registration of the promotion criterion.
pub const REGISTRATION: &str = "rce-repair-registration/1";
/// The format of an evaluation of the promotion criterion.
pub const EVALUATION: &str = "rce-repair-criterion/1";
/// The registration, its pinned SHA-256 and the recorded evaluation, under
/// `spec/`.
pub const REGISTRATION_PATH: &str = "proposals/repair-observe/criterion.json";
pub const PINNED_PATH: &str = "proposals/repair-observe/criterion.json.sha256";
pub const EVALUATION_PATH: &str = "proposals/repair-observe/evaluation.json";

/// One proposal: its origin label, its text and its digest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    pub origin: String,
    pub text: String,
    pub sha256: String,
}

/// A proposals file: the SHA-256 of its bytes and its proposals, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposals {
    pub sha256: String,
    pub list: Vec<Proposal>,
}

/// The digest of a proposal's text: SHA-256 of `rce-proposal/1`, a line
/// end and the text, as the rule-set hash is formed (10.15, AR-1).
pub fn digest(text: &str) -> String {
    sha256_hex(format!("{DIGEST}\n{text}").as_bytes())
}

/// The bytes of a proposals file holding `list` (origin, text), in order:
/// the canonical JSON of 6.1.1 and one line end.
pub fn write(list: &[(&str, &str)]) -> String {
    let ps: Vec<Value> = list.iter().map(|(o, t)| json!({"origin": o, "text": t})).collect();
    canon_json(&json!({"format": FORMAT, "proposals": ps})) + "\n"
}

/// Read a proposals file (RF-1): the object `{"format": "rce-proposals/1",
/// "proposals": [...]}`, each proposal exactly `origin` (a text of one line,
/// not empty) and `text`, written as canonical JSON (6.1.1) followed by one
/// line end. Anything else is a usage error that names the rule.
pub fn read(name: &str, bytes: &str) -> Result<Proposals, String> {
    let bad = |why: String| format!("{name}: RF-1 {why}");
    let v: Value = serde_json::from_str(bytes).map_err(|e| bad(format!("the file is not JSON ({e})")))?;
    let o = v.as_object().ok_or_else(|| bad("the file is not a JSON object".into()))?;
    if o.keys().map(String::as_str).collect::<Vec<_>>() != ["format", "proposals"] {
        return Err(bad("the members are exactly `format` and `proposals`".into()));
    }
    if o["format"] != FORMAT {
        return Err(bad(format!("the format is `{FORMAT}`")));
    }
    let mut list = vec![];
    for (i, p) in o["proposals"].as_array().ok_or_else(|| bad("`proposals` is a list".into()))?.iter().enumerate() {
        let n = i + 1;
        let p = p.as_object().filter(|p| p.keys().map(String::as_str).collect::<Vec<_>>() == ["origin", "text"]);
        let p = p.ok_or_else(|| bad(format!("proposal {n} has exactly the members `origin` and `text`")))?;
        let origin = p["origin"].as_str().filter(|s| !s.is_empty() && !s.contains(['\n', '\r']));
        let origin = origin.ok_or_else(|| bad(format!("the origin of proposal {n} is a text of one line, not empty")))?;
        let text = p["text"].as_str().ok_or_else(|| bad(format!("the text of proposal {n} is a text")))?;
        list.push(Proposal { origin: origin.to_string(), text: text.to_string(), sha256: digest(text) });
    }
    if bytes != canon_json(&v) + "\n" {
        return Err(bad("the file is not canonical JSON (6.1.1) followed by one line end: keys sorted, no space outside texts".into()));
    }
    Ok(Proposals { sha256: sha256_hex(bytes.as_bytes()), list })
}

/// The mode of rung f by the modes table of 10.13 (RF-3); a rung the table
/// does not list is `observe`.
pub fn mode() -> Mode {
    crate::specread::rung_mode("f").and_then(Mode::parse).unwrap_or(Mode::Observe)
}

// ---------- the promotion criterion (RF-5, RF-6) ----------

/// What the evaluator reads from a registration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Registration {
    /// the SHA-256 of the registration's bytes
    pub sha256: String,
    pub id: String,
    /// the date it was registered, `YYYY-MM-DD`
    pub date: String,
    pub corpus_sha256: String,
    pub tasks: usize,
    /// whether each record is made with the task's original as GOOD
    pub known: bool,
    pub bound: usize,
    /// the largest share of the closes rung f may give, as (num, den)
    pub share: (u64, u64),
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// Read a registration (RF-5); an error names what is missing.
pub fn registration(bytes: &str) -> Result<Registration, String> {
    let bad = |why: &str| format!("RF-5 the registration {why}");
    let v: Value = serde_json::from_str(bytes).map_err(|e| bad(&format!("is not JSON ({e})")))?;
    if v["format"] != REGISTRATION {
        return Err(bad(&format!("has the format `{REGISTRATION}`")));
    }
    let s = |p: &str| v.pointer(p).and_then(Value::as_str).map(String::from);
    let n = |p: &str| v.pointer(p).and_then(Value::as_u64);
    let id = s("/id").ok_or_else(|| bad("has an `id`"))?;
    let date = s("/registered").map(|d| d.chars().take(10).collect::<String>()).filter(|d| {
        d.len() == 10 && d.bytes().enumerate().all(|(i, c)| if i == 4 || i == 7 { c == b'-' } else { c.is_ascii_digit() })
    });
    let date = date.ok_or_else(|| bad("starts `registered` with its date, YYYY-MM-DD"))?;
    let corpus_sha256 = s("/corpus/sha256").filter(|h| is_hex64(h)).ok_or_else(|| bad("fixes the corpus by `corpus.sha256`"))?;
    let tasks = n("/corpus/tasks").ok_or_else(|| bad("counts the corpus's tasks in `corpus.tasks`"))? as usize;
    let known = match s("/observation/known").as_deref() {
        Some("original") => true,
        Some("none") => false,
        _ => return Err(bad("says in `observation.known` whether GOOD is the task's `original` or `none`")),
    };
    if s("/observation/mode").as_deref() != Some("observe") {
        return Err(bad("makes its observation in `observe` mode"));
    }
    let bound = n("/observation/bound").ok_or_else(|| bad("gives the bound of the observation in `observation.bound`"))? as usize;
    let share = match v.pointer("/criterion/max_stochastic_share").and_then(Value::as_array).map(|a| a.iter().map(Value::as_u64).collect::<Vec<_>>()) {
        Some(a) if a.len() == 2 && a[0].is_some() && a[1].is_some_and(|d| d > 0) => (a[0].unwrap(), a[1].unwrap()),
        _ => return Err(bad("gives `criterion.max_stochastic_share` as [NUM, DEN]")),
    };
    Ok(Registration { sha256: sha256_hex(bytes.as_bytes()), id, date, corpus_sha256, tasks, known, bound, share })
}

/// One task of the corpus: its id and the SHA-256 of its mutant (FILE) and
/// of its original (GOOD), as `scripts/ladder-tasks.py` writes them.
struct Task {
    id: String,
    mutant: String,
    original: Option<String>,
}

fn tasks(corpus: &str) -> Result<Vec<Task>, String> {
    let mut out = vec![];
    for (i, l) in corpus.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let bad = |why: &str| format!("CORPUS: line {}: {why}", i + 1);
        let v: Value = serde_json::from_str(l).map_err(|e| bad(&format!("not JSON ({e})")))?;
        let id = v["id"].as_str().ok_or_else(|| bad("a task has an `id`"))?.to_string();
        let mutant = v["mutant_sha256"].as_str().filter(|h| is_hex64(h)).ok_or_else(|| bad("a task has a `mutant_sha256`"))?.to_string();
        let original = v["original_sha256"].as_str().filter(|h| is_hex64(h)).map(String::from);
        out.push(Task { id, mutant, original });
    }
    Ok(out)
}

/// The observation records: one `rce repair json` report per line.
fn records(text: &str) -> Result<Vec<Value>, String> {
    let mut out = vec![];
    for (i, l) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()) {
        let bad = |why: &str| format!("RECORDS: line {}: {why}", i + 1);
        let v: Value = serde_json::from_str(l).map_err(|e| bad(&format!("not JSON ({e})")))?;
        if v["format"] != "rce-repair/1" {
            return Err(bad("a record is the report of `rce repair json` (format `rce-repair/1`)"));
        }
        if !v["inputs"]["file"].as_str().is_some_and(is_hex64) || v.get("proposals").is_none() || v.get("bound").is_none() {
            return Err(bad("a record has the members `inputs`, `bound` and `proposals` of RF-4"));
        }
        out.push(v);
    }
    Ok(out)
}

/// The kinds of comparison of an observed proposal with GOOD (RF-4).
const AGAINST: [&str; 5] = ["behaviour-equal", "contradicts", "not compared", "restored", "undecided"];

/// Evaluate the registered criterion (RF-6) over the corpus and the records:
/// the evaluation as JSON (format `rce-repair-criterion/1`). An error is a
/// usage error: a registration, corpus or record that cannot be read, or a
/// corpus that is not the registered one.
pub fn evaluate(reg_bytes: &str, corpus: &str, recs: &str) -> Result<Value, String> {
    let reg = registration(reg_bytes)?;
    let csha = sha256_hex(corpus.as_bytes());
    if csha != reg.corpus_sha256 {
        return Err(format!("CORPUS: RF-6 the corpus has SHA-256 {csha}; the registration fixes {}", reg.corpus_sha256));
    }
    let ts = tasks(corpus)?;
    if ts.len() != reg.tasks {
        return Err(format!("CORPUS: RF-6 the corpus has {} tasks; the registration counts {}", ts.len(), reg.tasks));
    }
    let rs = records(recs)?;
    // each task takes one record of its mutant, in the order of the records
    let mut by_file: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, r) in rs.iter().enumerate() {
        by_file.entry(r["inputs"]["file"].as_str().unwrap().to_string()).or_default().push(i);
    }
    let mut used = vec![false; rs.len()];
    let (mut exact, mut stochastic, mut given, mut not_closed, mut no_f) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut observed, mut wmut, mut judged) = (0u64, 0u64, 0u64);
    let mut known: BTreeMap<&str, u64> = AGAINST.iter().map(|k| (*k, 0)).collect();
    let mut origins: BTreeMap<String, u64> = BTreeMap::new();
    let mut cov: BTreeMap<&str, Vec<String>> = ["missing", "no_proposal", "not_observe", "wrong_bound", "wrong_known"].iter().map(|k| (*k, vec![])).collect();
    for t in &ts {
        let Some(i) = by_file.get_mut(&t.mutant).and_then(|v| if v.is_empty() { None } else { Some(v.remove(0)) }) else {
            cov.get_mut("missing").unwrap().push(t.id.clone());
            continue;
        };
        used[i] = true;
        let r = &rs[i];
        let p = &r["proposals"];
        if p["mode"] != "observe" {
            cov.get_mut("not_observe").unwrap().push(t.id.clone());
        }
        if r["bound"].as_u64() != Some(reg.bound as u64) {
            cov.get_mut("wrong_bound").unwrap().push(t.id.clone());
        }
        let good = r["inputs"]["known"].as_str();
        if reg.known && (t.original.is_none() || good != t.original.as_deref()) || !reg.known && good.is_some() {
            cov.get_mut("wrong_known").unwrap().push(t.id.clone());
        }
        judged += p["judged"].as_u64().unwrap_or(0);
        for x in p["records"].as_array().into_iter().flatten() {
            if !x["verdict"].as_str().unwrap_or("").starts_with("not judged") {
                *origins.entry(x["origin"].as_str().unwrap_or("").to_string()).or_default() += 1;
            }
        }
        let rung = r["rung"].as_str();
        if r["closed"] == true && rung.is_some_and(|x| ["a", "b", "c", "d", "e"].contains(&x)) {
            exact += 1;
        } else if rung == Some("none") {
            given += 1;
        } else {
            not_closed += 1;
            if p["ran"] != true {
                no_f += 1;
            } else if p["judged"].as_u64().unwrap_or(0) == 0 {
                cov.get_mut("no_proposal").unwrap().push(t.id.clone());
            }
            let obs: Vec<&Value> = p["records"].as_array().into_iter().flatten().filter(|x| x["observed"] == true).collect();
            if !obs.is_empty() {
                stochastic += 1;
            }
            for x in obs {
                observed += 1;
                if x["mutation"]["vacuous"] == 0 {
                    wmut += 1;
                }
                if reg.known {
                    let k = x["known"]["result"].as_str().unwrap_or("not compared");
                    *known.entry(AGAINST.iter().find(|a| **a == k).copied().unwrap_or("not compared")).or_default() += 1;
                }
            }
        }
    }
    let outside = used.iter().filter(|u| !**u).count();
    let mut v = json!({
        "format": EVALUATION,
        "registration": {"id": reg.id, "sha256": reg.sha256, "registered": reg.date},
        "corpus": {"sha256": csha, "tasks": ts.len()},
        "records": {"sha256": sha256_hex(recs.as_bytes()), "count": rs.len(), "outside": outside},
        "closes": {"exact": exact, "stochastic": stochastic, "max_share": [reg.share.0, reg.share.1]},
        "tasks": {"admitted_as_given": given, "not_closed": not_closed, "rung_f_not_run": no_f},
        "observed": {"proposals": observed, "w_mut_1": wmut, "known": known},
        "proposals": {"judged": judged, "origins": origins},
        "coverage": cov,
    });
    let why = reasons(&v, &reg);
    v["holds"] = json!(why.is_empty());
    v["reasons"] = json!(why);
    Ok(v)
}

/// Why the criterion does not hold on an evaluation, re-derived from its
/// counts and the registration: empty when it holds. The evaluator and the
/// mode-row check (RF-7) both decide by this one function.
pub fn reasons(v: &Value, reg: &Registration) -> Vec<String> {
    let n = |p: &str| v.pointer(p).and_then(Value::as_u64).unwrap_or(0);
    let mut why = vec![];
    for (k, what) in [
        ("missing", "have no record"),
        ("no_proposal", "that rungs a to e do not close had no proposal judged"),
        ("not_observe", "were not observed in `observe` mode"),
        ("wrong_bound", "were not run with the registered bound"),
        ("wrong_known", "were not run with the registered GOOD"),
    ] {
        let ids = v.pointer(&format!("/coverage/{k}")).and_then(Value::as_array).map(Vec::len).unwrap_or(1);
        if ids > 0 {
            why.push(format!("{ids} tasks {what}"));
        }
    }
    if n("/records/outside") > 0 {
        why.push(format!("{} records are of programs outside the corpus", n("/records/outside")));
    }
    let (e, f) = (n("/closes/exact"), n("/closes/stochastic"));
    let (num, den) = reg.share;
    if e + f == 0 {
        why.push("no task closes, so no share can be measured".into());
    } else if f * den > num * (e + f) {
        why.push(format!("{f} of {} closes would come from rung f ({}), above {num} of {den}", e + f, tenths(f, e + f)));
    }
    let obs = n("/observed/proposals");
    if n("/observed/w_mut_1") != obs {
        why.push(format!("{} of {obs} observed proposals do not pass W-MUT-1", obs - n("/observed/w_mut_1").min(obs)));
    }
    let c = n("/observed/known/contradicts");
    if c > 0 {
        why.push(format!("{c} observed proposals contradict GOOD"));
    }
    if v["corpus"]["sha256"] != reg.corpus_sha256.as_str() {
        why.push("the evaluation is not of the registered corpus".into());
    }
    if v["registration"]["sha256"] != reg.sha256.as_str() {
        why.push("the evaluation is not of this registration".into());
    }
    why
}

/// `a` of `b` as a percentage rounded down to a tenth.
fn tenths(a: u64, b: u64) -> String {
    if b == 0 {
        return "n/a".into();
    }
    let t = a * 1000 / b;
    format!("{}.{}%", t / 10, t % 10)
}

/// The text report of `rce repair criterion` (RF-6).
pub fn evaluation_report(v: &Value) -> String {
    let n = |p: &str| v.pointer(p).and_then(Value::as_u64).unwrap_or(0);
    let s = |p: &str| v.pointer(p).and_then(Value::as_str).unwrap_or("").to_string();
    let mut o = String::new();
    o.push_str(&format!("registration  {} (registered {}), sha256 {}\n", s("/registration/id"), s("/registration/registered"), s("/registration/sha256")));
    o.push_str(&format!("corpus        {} tasks, sha256 {} (the registered corpus)\n", n("/corpus/tasks"), s("/corpus/sha256")));
    o.push_str(&format!("records       {}, sha256 {}; {} of programs outside the corpus\n", n("/records/count"), s("/records/sha256"), n("/records/outside")));
    let (e, f) = (n("/closes/exact"), n("/closes/stochastic"));
    o.push_str(&format!(
        "closes        {e} by rungs a to e, {f} that rung f would close: {f} of {} ({}), at most {} of {}\n",
        e + f,
        tenths(f, e + f),
        n("/closes/max_share/0"),
        n("/closes/max_share/1")
    ));
    o.push_str(&format!(
        "tasks         {} admitted as given, {} not closed by rungs a to e, of which {} where rung f did not run\n",
        n("/tasks/admitted_as_given"),
        n("/tasks/not_closed"),
        n("/tasks/rung_f_not_run")
    ));
    let k: Vec<String> = AGAINST.iter().map(|a| format!("{} {a}", n(&format!("/observed/known/{a}")))).collect();
    o.push_str(&format!("observed      {} proposals: {} pass W-MUT-1; against GOOD: {}\n", n("/observed/proposals"), n("/observed/w_mut_1"), k.join(", ")));
    let origins: Vec<String> = v["proposals"]["origins"].as_object().map(|m| m.iter().map(|(k, x)| format!("{k} {x}")).collect()).unwrap_or_default();
    o.push_str(&format!("proposals     {} judged; by origin: {}\n", n("/proposals/judged"), if origins.is_empty() { "none".into() } else { origins.join(", ") }));
    for (k, ids) in v["coverage"].as_object().into_iter().flatten() {
        let ids: Vec<&str> = ids.as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        if !ids.is_empty() {
            o.push_str(&format!("coverage      {k}: {}\n", ids.join(", ")));
        }
    }
    if v["holds"] == true {
        o.push_str("criterion     holds: the mode row of rung f may be changed to auto (RF-7)\n");
    } else {
        let why: Vec<&str> = v["reasons"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        o.push_str(&format!("criterion     does not hold: {}\n", why.join("; ")));
    }
    o
}

/// The JSON of an evaluation: canonical JSON (6.1.1) and one line end.
pub fn evaluation_json(v: &Value) -> String {
    canon_json(v) + "\n"
}

// ---------- the mode-row check (RF-7) ----------

/// RF-7: the registration is the one its pinned SHA-256 names, and the mode
/// row of rung f is `auto` only when the recorded evaluation is of this
/// registration and, re-derived from its counts, shows the criterion met.
/// `pinned` is the text of the `.sha256` file (`HEX  criterion.json`);
/// `evaluation` the recorded evaluation, when there is one. Ok gives the
/// line `rce-spec-check` prints.
pub fn promotion_check(mode: Mode, reg_bytes: &str, pinned: &str, evaluation: Option<&str>) -> Result<String, String> {
    let reg = registration(reg_bytes)?;
    let pin = pinned.split_whitespace().next().unwrap_or("");
    if pin != reg.sha256 {
        return Err(format!("RF-7 the registration has SHA-256 {}, not the pinned {pin}: a registration is not edited after it is pinned", reg.sha256));
    }
    let ev = match evaluation {
        None => None,
        Some(t) => {
            let v: Value = serde_json::from_str(t).map_err(|e| format!("RF-7 the recorded evaluation is not JSON ({e})"))?;
            if v["format"] != EVALUATION {
                return Err(format!("RF-7 the recorded evaluation has the format `{EVALUATION}`"));
            }
            Some(v)
        }
    };
    let met = ev.as_ref().map(|v| (reasons(v, &reg), v["holds"] == true));
    let said = match &met {
        None => "no evaluation recorded".to_string(),
        Some((why, _)) if why.is_empty() => "the recorded evaluation meets it".to_string(),
        Some((why, _)) => format!("the recorded evaluation does not meet it ({})", why.join("; ")),
    };
    if let Some((why, holds)) = &met
        && why.is_empty() != *holds
    {
        return Err(format!("RF-7 the recorded evaluation says `holds: {holds}`, and its counts say otherwise ({said})"));
    }
    let line = format!("rung f of rce repair is {}; promotion criterion {} (registered {}, sha256 {}): {said}", mode.text(), reg.id, reg.date, reg.sha256);
    match (mode, &met) {
        (Mode::Auto, Some((why, _))) if why.is_empty() => Ok(line),
        (Mode::Auto, _) => Err(format!("RF-7 the mode row of rung f is auto, but {said}")),
        _ => Ok(line),
    }
}
