//! The evidence authority of lift reports (Appendix D.8, L5 and L6;
//! spec/proposals/evidence-authority.md), checked without Python:
//!
//! * every vector of spec/lift/authority-vectors.json gets its verdict and
//!   reason from `lift_authority::close`, the rule that spec/lift/authority.py
//!   applies in the probes; the vectors cover every authority level, inferred
//!   evidence that never closes a claim, and digest mismatches that fail closed;
//! * the levels of spec/lift/authority.json are the rows of D.8's table;
//! * the committed report of the Go probe on door.go is bound to the digests
//!   of door.go, table.json, lift_go.py and door.lifted.rce, states every
//!   claim of the probe with its authority, and states the verdicts the rule
//!   gives; a copy with one digest changed fails closed, and a copy whose
//!   evidence is marked inferred is undecided;
//! * the evidence `rce equiv` gives for the lifted door and the hand-written
//!   one closes GO-EQUIV at the level `bisimilar`.
use rce_spec_check::{cli, lift_authority as la};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

fn read(p: &str) -> Vec<u8> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn json_file(p: &str) -> Value {
    serde_json::from_slice(&read(p)).unwrap_or_else(|e| panic!("{p}: {e}"))
}

fn table() -> Value {
    json_file("../lift/authority.json")
}

fn report() -> Value {
    json_file("../lift/go/door.report.json")
}

#[test]
fn every_vector_gets_its_verdict() {
    let t = table();
    let v = json_file("../lift/authority-vectors.json");
    let vs = v["vectors"].as_array().unwrap();
    assert!(vs.len() >= 20);
    for x in vs {
        let deps: BTreeMap<String, String> = x["deps"].as_object().unwrap().iter().map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string())).collect();
        let got = la::close(&x["claim"], x["bindings"].as_object().unwrap(), x["records"].as_array().unwrap(), &t, &deps);
        assert_eq!((got.0.as_str(), got.1.as_str()), (x["verdict"].as_str().unwrap(), x["reason"].as_str().unwrap()), "{}", x["name"]);
    }
}

#[test]
fn the_vectors_cover_every_level_and_every_verdict() {
    let t = table();
    let v = json_file("../lift/authority-vectors.json");
    let vs = v["vectors"].as_array().unwrap();
    for l in t["levels"].as_array().unwrap() {
        let a = l["authority"].as_str().unwrap();
        let mine: Vec<&Value> = vs.iter().filter(|x| x["records"].as_array().unwrap().iter().any(|r| r["authority"] == a)).collect();
        assert!(!mine.is_empty(), "no vector has evidence of authority {a}");
        if a == "inferred" {
            assert!(l["closes"].as_array().unwrap().is_empty(), "inferred evidence closes nothing");
            assert!(mine.iter().all(|x| x["verdict"] != "closed" || x["records"].as_array().unwrap().iter().any(|r| r["kind"] == "observed")), "inferred evidence alone never closes a claim");
        } else {
            assert!(mine.iter().any(|x| x["verdict"] == "closed"), "no vector closes a claim with {a}");
        }
    }
    // a holding record whose kind is inferred never closes, whatever authority it names
    for x in vs {
        let rs = x["records"].as_array().unwrap();
        if rs.iter().all(|r| r["kind"] == "inferred") && !rs.is_empty() {
            assert_ne!(x["verdict"], "closed", "{}", x["name"]);
            assert!(rs.iter().any(|r| r["result"] != "holds") || x["verdict"] == "undecided", "{}", x["name"]);
        }
    }
    for w in ["closed", "open", "undecided", "failed"] {
        assert!(vs.iter().any(|x| x["verdict"] == w), "no vector is {w}");
        assert!(t["verdicts"].get(w).is_some());
    }
    assert!(vs.iter().any(|x| x["verdict"] == "failed" && x["reason"].as_str().unwrap().starts_with("digest mismatch")));
    assert!(vs.iter().any(|x| x["verdict"] == "failed" && x["reason"].as_str().unwrap().starts_with("malformed evidence")));
}

/// D.8's table of levels names, for each level, its authority, its kind and
/// REA's name: the rows agree with spec/lift/authority.json.
#[test]
fn the_levels_are_the_rows_of_d8() {
    let t = table();
    let book = String::from_utf8(read("../book/15-handler-spec-profile.md")).unwrap();
    let d8 = &book[book.find("### D.8").unwrap()..book.find("### D.9").unwrap()];
    for l in t["levels"].as_array().unwrap() {
        let a = l["authority"].as_str().unwrap();
        let row = d8.lines().find(|x| x.starts_with(&format!("| `{a}` |"))).unwrap_or_else(|| panic!("D.8 has no row for {a}"));
        assert!(row.contains(&format!("| {} |", l["kind"].as_str().unwrap())), "{row}");
        match l["rea"].as_str() {
            Some(r) => assert!(row.contains(&format!("`{}`", r.split(' ').next().unwrap())), "{row}"),
            None => assert!(row.ends_with("| none |"), "{row}"),
        }
        for c in l["closes"].as_array().unwrap() {
            assert!(row.contains(&format!("`{}`", c.as_str().unwrap())), "{row}");
        }
    }
}

#[test]
fn the_door_report_is_bound_to_its_files() {
    let r = report();
    assert_eq!(r["format"], "rce-lift-report/1");
    assert_eq!(r["lift"], "go");
    let want: Map<String, Value> = [
        ("source", la::sha256_hex(&read("../lift/go/door.go"))),
        ("table", la::sha256_hex(&read("../lift/go/table.json"))),
        ("lifter", la::sha256_hex(&read("../lift/go/lift_go.py"))),
        ("form", la::sha256_hex(&read("../lift/go/door.lifted.rce"))),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), Value::String(v)))
    .collect();
    let t = table();
    let decl = t["claims"]["go"].as_array().unwrap();
    let claims = r["claims"].as_array().unwrap();
    assert_eq!(claims.len(), decl.len(), "the report states every claim of the probe");
    for (c, d) in claims.iter().zip(decl) {
        assert_eq!((&c["id"], &c["revision"], &c["requires"]), (&d["id"], &d["revision"], &d["requires"]));
        let b = c["bindings"].as_object().unwrap();
        let names: Vec<&str> = b.keys().map(String::as_str).collect();
        let mut binds: Vec<&str> = d["binds"].as_array().unwrap().iter().map(|x| x.as_str().unwrap()).collect();
        binds.sort();
        assert_eq!(names, binds, "{}: the bindings are those the claim states", c["id"]);
        for (k, v) in &want {
            assert_eq!(&b[k], v, "{}: {k} is not the digest of its file; run lift_go.py report door.go again", c["id"]);
        }
        for e in c["evidence"].as_array().unwrap() {
            assert!(la::malformed(e, &t).is_none());
            assert!(e["authority"].is_string() && e["kind"].is_string());
        }
    }
    let got: Vec<(&str, &str)> = claims.iter().map(|c| (c["id"].as_str().unwrap(), c["verdict"].as_str().unwrap())).collect();
    assert_eq!(got, [("GO-ROWS", "closed"), ("GO-AGREE", "open"), ("GO-EQUIV", "open")]);
    for (id, stated, again) in la::recheck(&r, &t) {
        assert_eq!(stated, again, "{id}: the report's verdict is the rule's");
    }
}

/// Every row the GO-ROWS evidence names is a row of table.json with a clause
/// and a derivation.
#[test]
fn the_rows_used_cite_their_clauses() {
    let tab = json_file("../lift/go/table.json");
    let rows: BTreeMap<&str, &Value> = tab["rows"].as_array().unwrap().iter().map(|x| (x["id"].as_str().unwrap(), x)).collect();
    let r = report();
    let d = r["claims"][0]["evidence"][0]["detail"].as_str().unwrap();
    let used = d.split_once(": ").unwrap().1;
    for id in used.split(", ") {
        let row = rows.get(id).unwrap_or_else(|| panic!("{id} is not a row of table.json"));
        assert!(!row["clause"].as_str().unwrap().is_empty() && !row["derivation"].as_str().unwrap().is_empty(), "{id}");
    }
}

#[test]
fn a_changed_digest_fails_closed_and_inferred_evidence_stays_undecided() {
    let t = table();
    let mut r = report();
    r["claims"][0]["evidence"][0]["bindings"]["source"] = json!("0".repeat(64));
    let (_, _, got) = &la::recheck(&r, &t)[0];
    assert_eq!(got.0, "failed");
    assert_eq!(got.1, "digest mismatch: lift_go.py rows is bound to another source");

    let mut r = report();
    r["claims"][0]["bindings"]["table"] = json!("1".repeat(64));
    assert_eq!(la::recheck(&r, &t)[0].2.0, "failed", "a report whose table digest is not the evidence's fails closed");

    let mut r = report();
    r["claims"][0]["evidence"][0]["kind"] = json!("inferred");
    let (_, _, got) = &la::recheck(&r, &t)[0];
    assert_eq!(got.0, "undecided", "a holding record of kind inferred is not promoted to derived");

    let mut r = report();
    r["claims"][0]["evidence"][0]["authority"] = json!("pinned");
    r["claims"][0]["evidence"][0]["kind"] = json!("observed");
    assert_eq!(la::recheck(&r, &t)[0].2.0, "open", "pinned evidence does not close a derived claim");
}

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap())
}

/// The record `lift_go.py report door.go RCE HAND` writes for GO-EQUIV, made
/// here from `rce equiv json`: equivalent and exact, so it closes the claim
/// at `bisimilar`. Bound to a counterpart the report does not name, it fails.
#[test]
fn rce_equiv_closes_go_equiv_at_bisimilar() {
    let hand = "../examples/61_dim_minimal.rce";
    let (code, out) = rce(&["equiv", "json", "../lift/go/door.lifted.rce", hand]);
    assert_eq!(code, 0, "{out}");
    let e: Value = serde_json::from_str(&out).unwrap();
    assert_eq!((e["equivalent"].as_bool(), e["exact"].as_bool()), (Some(true), Some(true)));
    let t = table();
    let mut r = report();
    let checker = la::sha256_hex(&read(env!("CARGO_BIN_EXE_rce")));
    let counterpart = la::sha256_hex(&read(hand));
    let b = &mut r["claims"][2]["bindings"];
    b["checker"] = json!(checker);
    b["counterpart"] = json!(counterpart);
    let rec = json!({"claim": "GO-EQUIV", "revision": 1, "verifier": "rce equiv", "authority": "bisimilar", "kind": "derived",
                     "result": "holds", "bindings": b.clone(), "detail": "equivalent, exact"});
    r["claims"][2]["evidence"] = json!([rec]);
    assert_eq!(la::recheck(&r, &t)[2].2, ("closed".to_string(), "rce equiv: bisimilar".to_string()));
    r["claims"][2]["bindings"]["counterpart"] = json!(la::sha256_hex(&read("../examples/01_counter.rce")));
    assert_eq!(la::recheck(&r, &t)[2].2.0, "failed");
}
