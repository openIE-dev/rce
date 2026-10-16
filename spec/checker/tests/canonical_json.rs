//! Canonical JSON (spec 6.1.1): the golden vectors of `spec/data/canonical-json-vectors.json`
//! must be reproduced byte for byte by the reference implementation and must
//! carry the SHA-256 of those bytes. `scripts/canonical-json-vectors.py --check`
//! checks the same file with an encoder written from the text of 6.1.1 alone.
use rce_spec_check::interp::{FV, V, canon_json, d_sha, float_json, form_value};
use rce_spec_check::parse::parse;
use serde_json::Value;

/// Vectors on which the Rust reference formatter disagrees with the rule of
/// 6.1.1 (shortest digits; on a tie between two shortest candidates, the even
/// last digit, as ECMAScript Number::toString and Python repr do). Rust's
/// `{:e}` picks the odd candidate. Pinned here so the list shrinks when the
/// formatter is fixed; see 6.1.1 "Open items".
const RUST_FLOAT_TIE_DIVERGENCES: [(&str, &str); 1] = [("float-1424953923781206.2", "1424953923781206.3")];

fn rust_tie(id: &str) -> Option<&'static str> {
    RUST_FLOAT_TIE_DIVERGENCES.iter().find(|(i, _)| *i == id).map(|(_, r)| *r)
}

fn vectors() -> Vec<Value> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/canonical-json-vectors.json");
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    assert_eq!(v["format"], "rce-canonical-json-vectors/1");
    v["vectors"].as_array().unwrap().clone()
}

/// A vector value as an RCE runtime value, so that `form_value` (the embedding
/// of 6.1, including the sorting of records and maps) is part of what is tested.
fn to_v(t: &Value) -> V {
    let a = t.as_array().unwrap();
    match a[0].as_str().unwrap() {
        "null" => V::Unit,
        "bool" => V::Bool(a[1].as_bool().unwrap()),
        "int" => V::Int(a[1].as_str().unwrap().parse::<i128>().unwrap(), "i64"),
        "float" => V::F(f64::from_bits(u64::from_str_radix(a[1].as_str().unwrap().trim_start_matches("0x"), 16).unwrap())),
        "text" => V::Text(a[1].as_str().unwrap().to_string()),
        "list" => V::List(a[1].as_array().unwrap().iter().map(to_v).collect()),
        "rec" => V::Rec(
            "R".into(),
            a[1].as_array().unwrap().iter().map(|kv| (kv[0].as_str().unwrap().to_string(), to_v(&kv[1]))).collect(),
        ),
        "map" => V::Map(a[1].as_array().unwrap().iter().map(|kv| (to_v(&kv[0]), to_v(&kv[1]))).collect()),
        t => panic!("unknown tag {t}"),
    }
}

#[test]
fn golden_vectors_are_reproduced_byte_for_byte() {
    // a module with no kind named R, so records are plain records (no "$v")
    let m = parse(&std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples/01_counter.rce")).unwrap()).unwrap();
    let vs = vectors();
    assert!(vs.len() >= 60, "only {} vectors", vs.len());
    for v in &vs {
        let id = v["id"].as_str().unwrap();
        let got = form_value(&to_v(&v["value"]), &m).json();
        if let Some(rust) = rust_tie(id) {
            assert_eq!(got, rust, "pinned divergence of vector {id} changed: remove it from the list if fixed");
            assert_ne!(d_sha(&got), v["sha256"].as_str().unwrap(), "{id}");
            continue;
        }
        assert_eq!(got, v["canonical"].as_str().unwrap(), "vector {id}");
        assert_eq!(d_sha(&got), v["sha256"].as_str().unwrap(), "digest of vector {id}");
    }
}

/// The rule of 6.1.1 on every vector, with no pinned exceptions. Fails today
/// on the tie vector; enable by removing `#[ignore]` when the formatter is fixed.
#[test]
#[ignore = "known divergence: Rust shortest-digit tie rule, spec 6.1.1 'Open items'"]
fn the_reference_follows_the_tie_rule() {
    for v in vectors() {
        let a = v["value"].as_array().unwrap();
        if a[0] == "float" {
            let x = f64::from_bits(u64::from_str_radix(a[1].as_str().unwrap().trim_start_matches("0x"), 16).unwrap());
            assert_eq!(float_json(x), v["canonical"].as_str().unwrap(), "{}", v["id"]);
        }
    }
}

#[test]
fn float_forms_agree_with_the_float_vectors() {
    for v in vectors() {
        let a = v["value"].as_array().unwrap();
        if a[0] == "float" {
            let x = f64::from_bits(u64::from_str_radix(a[1].as_str().unwrap().trim_start_matches("0x"), 16).unwrap());
            if let Some(rust) = rust_tie(v["id"].as_str().unwrap()) {
                assert_eq!(float_json(x), rust);
                continue;
            }
            assert_eq!(float_json(x), v["canonical"].as_str().unwrap(), "{}", v["id"]);
            assert_eq!(FV::Float(x).json(), v["canonical"].as_str().unwrap());
        }
    }
}

/// Section 6.1.1 says one writer defines the bytes. `canon_json` (serde based;
/// writes `rce trace --json`) must give the same bytes as `FV::json` (which
/// is hashed) on every vector. It does not for U+0008 and U+000C.
#[test]
#[ignore = "known divergence, spec 6.1.1 'Open items': canon_json writes \\b and \\f, FV::json writes \\u0008 and \\u000c"]
fn the_trace_writer_agrees_with_the_hashed_writer() {
    for v in vectors() {
        let canonical = v["canonical"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(canonical).unwrap();
        assert_eq!(canon_json(&parsed), canonical, "vector {}", v["id"]);
    }
}

/// Documents the divergence the ignored test above exposes, so it cannot
/// silently change: exactly the vectors with U+0008 or U+000C differ.
#[test]
fn the_known_divergence_is_exactly_u0008_and_u000c() {
    let mut differ = vec![];
    for v in vectors() {
        let canonical = v["canonical"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(canonical).unwrap();
        if rust_tie(v["id"].as_str().unwrap()).is_some() {
            continue; // the parsed float is re-printed by the Rust formatter, a different divergence
        }
        if canon_json(&parsed) != canonical {
            differ.push(v["id"].as_str().unwrap().to_string());
        }
    }
    assert_eq!(differ, vec!["text-u0008", "text-u000c"]);
}
