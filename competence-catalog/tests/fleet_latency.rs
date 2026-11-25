use competence_catalog::fleet::{aggregate_latency, measure_local, NodeReading};

fn rd(node: &str, acts: u64, lat: Option<f64>) -> NodeReading {
    NodeReading { node: node.into(), source: "test".into(), uj: 10 * acts, acts, latency_us_per_act: lat }
}

#[test]
fn local_reading_carries_measured_latency() {
    let r = measure_local("lat-test", 50).unwrap();
    assert_eq!(r.acts, 50);
    let l = r.latency_us_per_act.expect("latency measured");
    assert!(l > 0.0);
}

#[test]
fn latency_aggregate_is_act_weighted() {
    let rs = vec![rd("a", 100, Some(10.0)), rd("b", 300, Some(20.0))];
    assert_eq!(aggregate_latency(&rs), Some(17.5));
}

#[test]
fn unmeasured_nodes_are_skipped_not_zeroed() {
    let rs = vec![rd("a", 100, Some(10.0)), rd("old", 300, None)];
    assert_eq!(aggregate_latency(&rs), Some(10.0));
    assert_eq!(aggregate_latency(&[rd("old", 300, None)]), None);
    assert_eq!(aggregate_latency(&[]), None);
}

#[test]
fn older_node_json_without_latency_deserializes() {
    let r: NodeReading = serde_json::from_str(r#"{"node":"n","source":"s","uj":1,"acts":1}"#).unwrap();
    assert_eq!(r.latency_us_per_act, None);
}
