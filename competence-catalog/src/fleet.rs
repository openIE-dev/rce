//! Wall-metered energy across the fleet — bare-metal activity tracked per node, aggregated.
//!
//! The fleet's computers can all track bare-metal activity (macOS IOReport package rails, Linux
//! RAPL). This turns `Modelled` cost into `Measured` receipts: each node measures the energy delta
//! of an execution window from its hardware counter, tags the source, and the fleet aggregates
//! readings into a total / mean joules-per-act. A node that cannot measure errors — never a
//! fabricated 0. This is the wall-metered provenance the Law of Coding's joule ranking requires.

use crate::energy::EnergyMeter;
use crate::lift::lift_expr;
use crate::repair::eval_expr;

/// One node's bare-metal energy reading for a batch of acts.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeReading {
    pub node: String,
    pub source: String,
    pub uj: u64,
    pub acts: u64,
}

/// Measure `acts` competent acts on this node via the hardware meter (wall-metered, `Measured`).
pub fn measure_local(node: &str, acts: usize) -> Result<NodeReading, String> {
    let mut m = EnergyMeter::open().map_err(|e| format!("{e:?}"))?;
    let source = m.source().to_string();
    let uj = m.measure_mean(acts, || {
        if let Ok(e) = lift_expr("2 + 3") {
            let _ = eval_expr(&e);
        }
    }).map_err(|e| format!("{e:?}"))?;
    Ok(NodeReading { node: node.to_string(), source, uj, acts: acts as u64 })
}

/// Aggregate fleet readings: (total µJ, total acts, mean µJ/act).
pub fn aggregate(readings: &[NodeReading]) -> (u64, u64, f64) {
    let total_uj: u64 = readings.iter().map(|r| r.uj).sum();
    let total_acts: u64 = readings.iter().map(|r| r.acts).sum();
    let mean = if total_acts == 0 { 0.0 } else { total_uj as f64 / total_acts as f64 };
    (total_uj, total_acts, mean)
}

/// Collect wall-metered readings from fleet nodes over HTTP (`GET {node}/energy?acts=N`).
/// A node that cannot measure is skipped (never a fabricated 0). This is the fleet-wide
/// aggregation the bare-metal fleet enables.
pub fn collect(nodes: &[String], acts: usize) -> Vec<NodeReading> {
    nodes.iter().filter_map(|n| {
        let url = format!("{n}/energy?acts={acts}");
        let resp = ureq::get(&url).call().ok()?;
        resp.into_json::<NodeReading>().ok()
    }).collect()
}