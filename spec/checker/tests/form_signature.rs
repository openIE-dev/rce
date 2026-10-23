//! The Form signature of 6.1 on the language side: every entry of every
//! example log D carries the canonical fields, the chain links, and the
//! state hash is the hash of the recorded state's positional form (T7;
//! the same rule as the engine profile, D.6).
use rce_spec_check::interp::{D_GENESIS, d_sha, form_json};
use rce_spec_check::parse::parse;
use serde_json::Value;

const CANON: [&str; 5] = ["seq", "event", "prev_hash", "state_hash", "hash"];

#[test]
fn every_example_log_is_a_canonical_chain() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../examples");
    let mut entries = 0;
    for f in std::fs::read_dir(&dir).unwrap() {
        let p = f.unwrap().path();
        if p.extension().and_then(|e| e.to_str()) != Some("rce") {
            continue;
        }
        let Ok(m) = parse(&std::fs::read_to_string(&p).unwrap()) else { continue };
        let j: Value = serde_json::from_str(&form_json(&m)).unwrap();
        for log in j["logs"].as_array().unwrap() {
            let mut prev = D_GENESIS.to_string();
            for (i, e) in log["D"].as_array().unwrap().iter().enumerate() {
                for k in CANON {
                    assert!(e.get(k).is_some(), "{}: entry {i} lacks {k}", p.display());
                }
                assert_eq!(e["seq"], i);
                assert_eq!(e["prev_hash"], prev.as_str());
                // T7: the hashes are over the positional forms, repeated under
                // `hashed`; `event` and `state` are the readable forms.
                assert_eq!(e["state_hash"], d_sha(&rce_spec_check::interp::canon_json(&e["hashed"]["state"])).as_str(), "{}", p.display());
                assert!(e["hashed"]["state"].is_array(), "the hashed state names no region");
                let body = serde_json::json!({
                    "event": e["hashed"]["event"], "prev_hash": e["prev_hash"],
                    "seq": e["seq"], "state_hash": e["state_hash"],
                });
                assert_eq!(e["hash"], d_sha(&rce_spec_check::interp::canon_json(&body)).as_str());
                assert!(e["event"]["type"].is_string());
                assert!(e["hashed"]["event"]["type"].is_number() || e["hashed"]["event"]["type"] == "Tick", "{}", p.display());
                prev = e["hash"].as_str().unwrap().to_string();
                entries += 1;
            }
        }
    }
    assert!(entries >= 10, "only {entries} entries checked");
}
