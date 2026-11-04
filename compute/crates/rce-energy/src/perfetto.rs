//! Export energy spans and counters as Chrome/Perfetto trace JSON
//! (the Trace Event Format that ui.perfetto.dev opens). Lesson from
//! CodeGreen, which exports per-function energy for Perfetto.
//!
//! `synthetic` is written into the trace metadata so simulated or
//! box-generated joules are never mistaken for measurements.

use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub name: String,
    pub track: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub joules: f64,
}

pub fn to_trace(spans: &[Span], counters: &[(String, u64, f64)], synthetic: bool) -> Value {
    let mut ev: Vec<Value> = Vec::new();
    let mut tracks: Vec<&str> = Vec::new();
    for s in spans {
        let tid = match tracks.iter().position(|t| *t == s.track) {
            Some(i) => i,
            None => { tracks.push(&s.track); tracks.len() - 1 }
        };
        ev.push(json!({"ph":"X","name":s.name,"pid":1,"tid":tid,
            "ts": s.start_ns as f64 / 1e3, "dur": (s.end_ns.saturating_sub(s.start_ns)) as f64 / 1e3,
            "args": {"joules": s.joules}}));
    }
    for (i, t) in tracks.iter().enumerate() {
        ev.push(json!({"ph":"M","name":"thread_name","pid":1,"tid":i,"args":{"name":t}}));
    }
    for (name, t, w) in counters {
        ev.push(json!({"ph":"C","name":name,"pid":1,"ts": *t as f64 / 1e3,"args":{"watts": w}}));
    }
    json!({"traceEvents": ev, "displayTimeUnit":"ns",
           "metadata": {"source":"rce-ide","synthetic_energy": synthetic}})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spans_and_counters() {
        let v = to_trace(&[Span{name:"f".into(),track:"cpu".into(),start_ns:1000,end_ns:3000,joules:0.5}],
            &[("pkg".into(), 2000, 4.0)], true);
        let e = v["traceEvents"].as_array().unwrap();
        assert_eq!(e[0]["ph"], "X");
        assert_eq!(e[0]["dur"], 2.0);
        assert_eq!(e[0]["args"]["joules"], 0.5);
        assert!(e.iter().any(|x| x["ph"] == "C"));
        assert_eq!(v["metadata"]["synthetic_energy"], true);
    }
}
