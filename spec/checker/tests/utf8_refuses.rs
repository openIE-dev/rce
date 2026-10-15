//! 5.15 `utf8`: invalid bytes refuse the event (T8) on the host and in
//! `rce trace`; the state is unchanged and nothing is substituted.
use rce_spec_check::cli::trace;
use serde_json::Value;

const SRC: &str = "state S\n    t: text = \"\"\n\nevent Dec(b: List[i64])\n\nstep S on Dec(b: b) -> b.utf8()\n";

fn steps(evs: &str) -> Vec<Value> {
    let t: Value = serde_json::from_str(&trace("t.rce", SRC, Some(evs)).unwrap()).unwrap();
    t["traces"][0]["steps"].as_array().unwrap().clone()
}

#[test]
fn invalid_bytes_refuse_and_leave_state() {
    for bad in ["[255]", "[192, 175]", "[237, 160, 128]", "[244, 144, 128, 128]", "[226, 130]", "[300]", "[-1]"] {
        let s = steps(&format!("Dec(b: [104, 105]), Dec(b: {bad})"));
        assert_eq!(s.len(), 2, "{bad}");
        assert_eq!(s[0]["outcome"], "applied");
        assert!(s[1]["outcome"]["refused"].as_str().unwrap().contains("utf8"), "{bad}: {}", s[1]["outcome"]);
        assert_eq!(s[1]["state_hash"], s[0]["state_hash"], "{bad} changed state");
    }
    assert_eq!(steps("Dec(b: [239, 187, 191, 97])")[0]["outcome"], "applied");
}
