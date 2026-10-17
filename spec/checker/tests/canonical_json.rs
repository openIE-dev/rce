//! Canonical JSON (spec 6.1.1): the golden vectors of `spec/data/canonical-json-vectors.json`
//! must be reproduced byte for byte by the reference implementation and must
//! carry the SHA-256 of those bytes. `scripts/canonical-json-vectors.py --check`
//! checks the same file with an encoder written from the text of 6.1.1 alone.
use rce_spec_check::interp::{FV, V, canon_json, d_sha, float_json, form_value};
use rce_spec_check::parse::parse;
use serde_json::Value;

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
        assert_eq!(got, v["canonical"].as_str().unwrap(), "vector {id}");
        assert_eq!(d_sha(&got), v["sha256"].as_str().unwrap(), "digest of vector {id}");
    }
}

/// The rule of 6.1.1 on every vector, with no pinned exceptions (including
/// the exact ties: the even last digit wins, as in ECMAScript and Python).
#[test]
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
            assert_eq!(float_json(x), v["canonical"].as_str().unwrap(), "{}", v["id"]);
            assert_eq!(FV::Float(x).json(), v["canonical"].as_str().unwrap());
        }
    }
}

/// Section 6.1.1 says one writer defines the bytes. `canon_json` (serde based;
/// writes `rce trace --json`) must give the same bytes as `FV::json` (which
/// is hashed) on every vector, U+0008 and U+000C included.
#[test]
fn the_trace_writer_agrees_with_the_hashed_writer() {
    for v in vectors() {
        let canonical = v["canonical"].as_str().unwrap();
        let parsed: Value = serde_json::from_str(canonical).unwrap();
        assert_eq!(canon_json(&parsed), canonical, "vector {}", v["id"]);
    }
}

/// `shortest_e` on exact ties, both directions, and on values where the
/// closest digits are not a tie.
#[test]
fn shortest_e_breaks_exact_ties_to_even() {
    use rce_spec_check::interp::shortest_e;
    let t = |x: f64| shortest_e(x);
    assert_eq!(t(1424953923781206.25), ("14249539237812062".to_string(), 15)); // tie, even 2: down
    assert_eq!(t(1424953923781206.75), ("14249539237812068".to_string(), 15)); // tie, odd 7: up
    assert_eq!(t(-1424953923781207.25), ("14249539237812072".to_string(), 15));
    assert_eq!(t(0.1), ("1".to_string(), -1));
    assert_eq!(t(5e-324), ("5".to_string(), -324));
    assert_eq!(t(1e23), ("1".to_string(), 23)); // 1e23 is not exact in binary64; the shortest reading is still "1"
}
