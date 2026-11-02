//! The evidence authority of a lift report (Appendix D.8, L5 and L6;
//! `spec/proposals/evidence-authority.md`).
//!
//! The levels, verdicts and each probe's claims are data
//! (`spec/lift/authority.json`, format `rce-lift-authority/1`). [`close`]
//! decides one claim from its evidence records, by the same rule as
//! `spec/lift/authority.py`, which the lift probes call: integrity and digest
//! binding first (fail closed), then a record that differs, then what keeps a
//! claim open, then authority. A claim closes only on a record that holds, at
//! a level that closes the claim's required level, of that level's kind,
//! bound to every digest the claim states. Inferred evidence never closes a
//! claim: the verdict is `undecided`.
//!
//! The probes' reports (format `rce-lift-report/1`) are checked against this
//! rule by `tests/lift_authority.rs`, without Python.
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};

/// The verdict of one claim and its reason.
pub type Verdict = (String, String);

/// The SHA-256 of a byte string as 64 lowercase hex digits: the form of every
/// binding of a lift report (the digest of a file is that of its bytes).
pub fn sha256_hex(b: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
}

fn is_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn text<'a>(v: &'a Value, k: &str) -> &'a str {
    v.get(k).and_then(Value::as_str).unwrap_or("")
}

fn level<'a>(t: &'a Value, authority: &str) -> Option<&'a Value> {
    t["levels"].as_array()?.iter().find(|l| text(l, "authority") == authority)
}

fn closes(t: &Value, authority: &str, required: &str) -> bool {
    level(t, authority)
        .and_then(|l| l["closes"].as_array())
        .is_some_and(|c| c.iter().any(|x| x.as_str() == Some(required)))
}

fn level_kind<'a>(t: &'a Value, authority: &str) -> &'a str {
    level(t, authority).map(|l| text(l, "kind")).unwrap_or("")
}

/// Why a record cannot be read, or `None`.
pub fn malformed(r: &Value, t: &Value) -> Option<String> {
    if !r.is_object() {
        return Some("a record is not an object".into());
    }
    for k in ["claim", "verifier", "authority", "kind", "result"] {
        if text(r, k).is_empty() {
            return Some(format!("the record has no {k}"));
        }
    }
    if !r.get("revision").is_some_and(|v| v.is_i64() || v.is_u64()) {
        return Some("the record has no integer revision".into());
    }
    if level(t, text(r, "authority")).is_none() {
        return Some(format!("unknown authority `{}`", text(r, "authority")));
    }
    if t["kinds"].get(text(r, "kind")).is_none() {
        return Some(format!("unknown kind `{}`", text(r, "kind")));
    }
    if !t["results"].as_array().is_some_and(|a| a.iter().any(|x| x.as_str() == Some(text(r, "result")))) {
        return Some(format!("unknown result `{}`", text(r, "result")));
    }
    let Some(b) = r.get("bindings").and_then(Value::as_object) else {
        return Some("the record has no bindings".into());
    };
    for (k, v) in b {
        if !v.as_str().is_some_and(is_digest) {
            return Some(format!("binding `{k}` is not a SHA-256 digest"));
        }
    }
    None
}

fn detail(r: &Value) -> String {
    let d = text(r, "detail");
    if d.is_empty() { String::new() } else { format!(": {d}") }
}

/// The verdict of one claim (L6). `bindings` holds the digests the report
/// states now (a null value: one it cannot state); `deps` the verdicts of
/// the claims the claim depends on.
pub fn close(claim: &Value, bindings: &Map<String, Value>, records: &[Value], t: &Value, deps: &BTreeMap<String, String>) -> Verdict {
    let v = |a: &str, b: String| (a.to_string(), b);
    let id = text(claim, "id");
    let rev = claim["revision"].as_i64();
    let required = text(claim, "requires");
    let binds: Vec<&str> = claim["binds"].as_array().map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let mine: Vec<&Value> = records.iter().filter(|r| r.get("claim").and_then(Value::as_str) == Some(id)).collect();
    for r in &mine {
        if let Some(why) = malformed(r, t) {
            return v("failed", format!("malformed evidence: {why}"));
        }
    }
    let live: Vec<&Value> = mine.iter().copied().filter(|r| r["revision"].as_i64() == rev).collect();
    let bound = |r: &Value, k: &str| r["bindings"].get(k).and_then(Value::as_str).map(str::to_string);
    for r in &live {
        for k in &binds {
            let Some(have) = bound(r, k) else { continue };
            if bindings.get(*k).and_then(Value::as_str) != Some(have.as_str()) {
                return v("failed", format!("digest mismatch: {} is bound to another {k}", text(r, "verifier")));
            }
        }
    }
    for r in &live {
        if text(r, "result") == "differs" {
            return v("failed", format!("{}: differs{}", text(r, "verifier"), detail(r)));
        }
    }
    for d in claim["depends"].as_array().into_iter().flatten().filter_map(Value::as_str) {
        if deps.get(d).map(String::as_str) != Some("closed") {
            return v("open", format!("dependency {d} is not closed"));
        }
    }
    if mine.is_empty() {
        return v("open", "no evidence".into());
    }
    if live.is_empty() {
        return v("open", format!("stale revision: no record answers revision {}", rev.unwrap_or(0)));
    }
    let mut seen = BTreeSet::new();
    for r in &live {
        if !seen.insert(text(r, "verifier")) {
            return v("open", format!("duplicate evidence from {}", text(r, "verifier")));
        }
    }
    for r in &live {
        let a = text(r, "authority");
        if text(r, "result") == "holds" && closes(t, a, required) && text(r, "kind") == level_kind(t, a) && binds.iter().all(|k| bound(r, k).is_some()) {
            return v("closed", format!("{}: {a}", text(r, "verifier")));
        }
    }
    for r in &live {
        if text(r, "result") == "undecided" {
            return v("undecided", format!("{}: undecided{}", text(r, "verifier"), detail(r)));
        }
    }
    for r in &live {
        if text(r, "result") == "holds" && text(r, "kind") == "inferred" {
            return v("undecided", format!("{}: inferred evidence never closes a claim", text(r, "verifier")));
        }
    }
    for r in &live {
        let a = text(r, "authority");
        if text(r, "result") == "holds" && text(r, "kind") != level_kind(t, a) {
            return v("undecided", format!("{}: {} evidence is not promoted to {a}", text(r, "verifier"), text(r, "kind")));
        }
    }
    for r in &live {
        if text(r, "result") == "holds" && closes(t, text(r, "authority"), required) {
            let missing: Vec<&str> = binds.iter().copied().filter(|k| bound(r, k).is_none()).collect();
            return v("open", format!("{} is not bound to {}", text(r, "verifier"), missing.join(", ")));
        }
    }
    let have: BTreeSet<&str> = live.iter().map(|r| text(r, "authority")).collect();
    v("open", format!("no evidence of authority {required}: {}", have.into_iter().collect::<Vec<_>>().join(", ")))
}

/// Every claim of a lift report decided again from its own evidence and
/// bindings: (claim id, the verdict the report states, the verdict of
/// [`close`]). A claim's dependencies are the claims before it.
pub fn recheck(report: &Value, t: &Value) -> Vec<(String, Verdict, Verdict)> {
    let mut out = vec![];
    let mut deps = BTreeMap::new();
    for c in report["claims"].as_array().into_iter().flatten() {
        let empty = Map::new();
        let b = c["bindings"].as_object().unwrap_or(&empty);
        let ev: Vec<Value> = c["evidence"].as_array().cloned().unwrap_or_default();
        let lift = text(report, "lift");
        let decl = t["claims"][lift].as_array().and_then(|a| a.iter().find(|d| d["id"] == c["id"])).cloned().unwrap_or(Value::Null);
        let got = close(&decl, b, &ev, t, &deps);
        deps.insert(text(c, "id").to_string(), got.0.clone());
        out.push((text(c, "id").to_string(), (text(c, "verdict").to_string(), text(c, "reason").to_string()), got));
    }
    out
}
