//! Energy on the graph. Each action (one ide-receipt) is attributed to the
//! code nodes its detail names, and to the hardware its measurement came
//! from (`device` in the detail, else the meter source tag). Joules are
//! E_meas when the receipt has a measurement, else E_sim, and the node
//! keeps both sums so the view can say which it shows. Receipts with
//! neither contribute nothing (no number is invented).
use crate::graph::{EdgeKind, Graph, Kind};
use serde::{Deserialize, Serialize};

/// One action, flattened from an ide-receipt (or from demo data, marked).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Action {
    pub seq: u64,
    /// Unix milliseconds (action end).
    pub t_ms: i64,
    pub action: String,
    pub e_sim_j: Option<f64>,
    pub e_meas_j: Option<f64>,
    /// Meter source tag (`rapl`, `nvml`, `ioreport`, `smc`, `hwmon`, ...).
    pub source: Option<String>,
    pub delta_j: Option<f64>,
    /// Hardware node id or label the action used, when the receipt names one.
    pub device: Option<String>,
    /// Names the action touched (item names, paths), from the receipt detail.
    pub names: Vec<String>,
    /// Demo/synthetic numbers, not from a receipt.
    #[serde(default)]
    pub synthetic: bool,
}

/// Mean and standard deviation of the deltas.
pub fn sigma(actions: &[Action]) -> Option<(f64, f64)> {
    let d: Vec<f64> = actions.iter().filter_map(|a| a.delta_j).collect();
    if d.len() < 2 {
        return None;
    }
    let m = d.iter().sum::<f64>() / d.len() as f64;
    let v = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (d.len() - 1) as f64;
    Some((m, v.sqrt()))
}

fn words(s: &str) -> impl Iterator<Item = &str> {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .filter(|w| !w.is_empty())
}

/// Which code nodes an action names: exact item names (or the bare last
/// segment of `Owner.method`) as whole words in its names.
pub fn targets(g: &Graph, a: &Action) -> Vec<usize> {
    let mut out = vec![];
    for (i, n) in g.nodes.iter().enumerate() {
        if n.kind.is_hw() || n.id.starts_with("check:") {
            continue;
        }
        let name = n.id.rsplit_once(':').map(|x| x.1).unwrap_or(&n.label);
        let hit = a.names.iter().any(|s| words(s).any(|w| w == name));
        if hit {
            out.push(i);
        }
    }
    out
}

/// The hardware node an action's energy was measured on.
pub fn hw_target(g: &Graph, a: &Action) -> Option<usize> {
    if let Some(d) = &a.device
        && let Some(i) = g
            .nodes
            .iter()
            .position(|n| n.kind.is_hw() && (&n.id == d || &n.label == d))
    {
        return Some(i);
    }
    let first = |k: Kind| g.nodes.iter().position(|n| n.kind == k);
    let src = a.source.as_deref()?;
    match src {
        "rapl" | "ioreport" | "powermetrics" => first(Kind::Package),
        "nvml" => first(Kind::Gpu),
        "hwmon" => first(Kind::Gpu).or_else(|| first(Kind::System)),
        "jetson" | "smc" | "power-supply" => first(Kind::System),
        _ => None, // estimated-remote: not this machine's hardware
    }
}

/// Apply actions to `g`: node energy, anomaly flags, edge traffic, and
/// code -> hardware `Ran` edges. `k` is the sigma multiple for flags.
pub fn apply(g: &mut Graph, actions: &[Action], k: f64) {
    let sig = sigma(actions);
    for a in actions {
        let j = a.e_meas_j.or(a.e_sim_j);
        let anomalous = match (a.delta_j, sig) {
            (Some(d), Some((m, s))) if s > 0.0 => ((d - m) / s).abs() > k,
            _ => false,
        };
        let ts = targets(g, a);
        let hwn = hw_target(g, a);
        if ts.is_empty()
            && let Some(j) = j
        {
            g.unattributed_j += j;
        }
        let share = j.map(|j| j / ts.len().max(1) as f64);
        let pair = match (a.e_meas_j, a.e_sim_j) {
            (Some(m), Some(s)) => Some((m / ts.len().max(1) as f64, s / ts.len().max(1) as f64)),
            _ => None,
        };
        let mut credit = |i: usize| {
            let e = &mut g.nodes[i].energy;
            e.actions += 1;
            e.anomaly |= anomalous;
            e.synthetic |= a.synthetic;
            if let Some(s) = share {
                if a.e_meas_j.is_some() {
                    e.measured_j += s;
                } else {
                    e.simulated_j += s;
                }
                e.joules = Some(e.joules.unwrap_or(0.0) + s);
            }
            if let Some((m, s)) = pair {
                e.pair_meas_j += m;
                e.pair_sim_j += s;
            }
        };
        for &i in &ts {
            credit(i);
        }
        if let Some(h) = hwn
            && a.e_meas_j.is_some()
        {
            credit(h);
        }
        // Traffic: edges between nodes touched by the same action.
        for e in g.edges.iter_mut() {
            if ts.contains(&e.a) && ts.contains(&e.b) {
                e.weight += 1.0;
            }
        }
        if let (Some(h), Some(j)) = (hwn, a.e_meas_j) {
            for &i in &ts {
                g.link(
                    i,
                    h,
                    EdgeKind::Ran,
                    j / ts.len() as f64,
                    a.source.as_deref().unwrap_or(""),
                );
            }
        }
    }
}

/// Heat colour for joules relative to the max on screen: blue -> amber -> red.
pub fn heat(j: f64, max: f64) -> [f32; 4] {
    let t = if max > 0.0 {
        (j / max).clamp(0.0, 1.0) as f32
    } else {
        0.0
    };
    let t = t.sqrt();
    if t < 0.5 {
        let u = t * 2.0;
        [0.2 + 0.8 * u, 0.5 + 0.3 * u, 1.0 - 0.8 * u, 1.0]
    } else {
        let u = (t - 0.5) * 2.0;
        [1.0, 0.8 - 0.6 * u, 0.2 - 0.1 * u, 1.0]
    }
}

/// Diverging colour for log2(E_meas/E_sim): blue when the model
/// over-predicts, grey when it matches, red when the meter reads more.
/// Saturates at a factor of 4 either way.
pub fn diverge_color(d: f64) -> [f32; 4] {
    let t = (d / 2.0).clamp(-1.0, 1.0) as f32;
    if t < 0.0 {
        let u = -t;
        [0.6 - 0.45 * u, 0.62 - 0.2 * u, 0.66 + 0.34 * u, 1.0]
    } else {
        [0.6 + 0.4 * t, 0.62 - 0.45 * t, 0.66 - 0.5 * t, 1.0]
    }
}

#[cfg(test)]
mod diverge_tests {
    use super::*;
    use crate::graph::{Energy, Graph, Kind, Node};
    #[test]
    fn paired_actions_give_divergence() {
        let mut g = Graph::default();
        g.add(Node { id: "code:a:f".into(), label: "f".into(), kind: Kind::of_def("fn"), file: None, line: None,
            fit: false, rank: 0.0, energy: Default::default(), live: None, detail: String::new() });
        let a = Action { e_sim_j: Some(1.0), e_meas_j: Some(2.0), names: vec!["f".into()], ..Default::default() };
        apply(&mut g, &[a], 3.0);
        let d = g.nodes[0].energy.divergence().unwrap();
        assert!((d - 1.0).abs() < 1e-9);
        assert!(diverge_color(2.0)[0] > diverge_color(-2.0)[0]);
        assert!(Energy::default().divergence().is_none());
    }
}
