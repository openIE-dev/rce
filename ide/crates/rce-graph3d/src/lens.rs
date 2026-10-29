//! Readability for large graphs, after CodeCharta and ExplorViz:
//! a user-chosen metric for each visual channel, semantic zoom that folds
//! definitions into their file when the camera is far, and a mini-map.
use crate::camera::Orbit;
use crate::graph::{Edge, Graph, Kind, Node};
use crate::layout::Positions;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric { Rank, Joules, Measured, Simulated, Divergence, Actions, Watts, Degree }

/// Which metric drives which channel. `None` keeps the default look.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Mapping {
    pub size: Option<Metric>,
    pub color: Option<Metric>,
    pub glow: Option<Metric>,
}

pub fn value(g: &Graph, i: usize, m: Metric) -> Option<f64> {
    let n = &g.nodes[i];
    match m {
        Metric::Rank => Some(n.rank),
        Metric::Joules => n.energy.joules,
        Metric::Measured => (n.energy.measured_j > 0.0).then_some(n.energy.measured_j),
        Metric::Simulated => (n.energy.simulated_j > 0.0).then_some(n.energy.simulated_j),
        Metric::Divergence => n.energy.divergence(),
        Metric::Actions => Some(n.energy.actions as f64),
        Metric::Watts => n.live.as_ref().and_then(|l| l.watts),
        Metric::Degree => Some(g.edges.iter().filter(|e| e.a == i || e.b == i).count() as f64),
    }
}

/// Values normalised to 0..1 over the nodes that have one (min-max).
pub fn normalised(g: &Graph, m: Metric) -> Vec<Option<f32>> {
    let v: Vec<Option<f64>> = (0..g.nodes.len()).map(|i| value(g, i, m)).collect();
    let (lo, hi) = v.iter().flatten().fold((f64::MAX, f64::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    v.into_iter().map(|x| x.map(|x| if hi > lo { ((x - lo) / (hi - lo)) as f32 } else { 0.5 })).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Detail { Files, Defs }

/// Far camera: files; near: definitions. `extent` is the layout radius.
pub fn detail(cam: &Orbit, extent: f32) -> Detail {
    if cam.dist > extent * 2.5 { Detail::Files } else { Detail::Defs }
}

/// Fold every code node into one node per file at the centroid of its
/// members, summing energy and merging edges. Hardware is kept as is.
pub fn fold_files(g: &Graph, pos: &Positions) -> (Graph, Positions) {
    fold_files_except(g, pos, &Default::default())
}

/// The id of the folded node standing for `file`.
pub fn file_node_id(file: &str) -> String {
    format!("file:{file}")
}

/// The file a folded node stands for, if `n` is one.
pub fn folded_file(n: &Node) -> Option<&str> {
    (n.detail == "folded").then(|| n.id.strip_prefix("file:")).flatten()
}

/// As [`fold_files`], but files in `expanded` keep their definitions
/// (click a folded file to open it in place while the rest stays folded).
pub fn fold_files_except(g: &Graph, pos: &Positions, expanded: &std::collections::BTreeSet<String>) -> (Graph, Positions) {
    let mut out = Graph { form_hash: g.form_hash.clone(), unattributed_j: g.unattributed_j, ..Default::default() };
    let mut opos = Positions::new();
    let mut map = vec![0usize; g.nodes.len()];
    let mut byfile: BTreeMap<String, usize> = BTreeMap::new();
    let mut acc: BTreeMap<usize, ([f32; 3], f32)> = BTreeMap::new();
    for (i, n) in g.nodes.iter().enumerate() {
        let key = match (&n.file, n.kind.is_hw()) { (Some(f), false) if !expanded.contains(f) => Some(f.clone()), _ => None };
        let j = match key {
            None => out.add(n.clone()),
            Some(f) => *byfile.entry(f.clone()).or_insert_with(|| out.add(Node {
                id: format!("file:{f}"), label: f.clone(), kind: Kind::Other, file: Some(f), line: Some(1),
                requires_change: false, rank: 0.0, energy: Default::default(), live: None, detail: "folded".into(),
            })),
        };
        map[i] = j;
        if let Some(p) = pos.get(&n.id) {
            let a = acc.entry(j).or_insert(([0.0; 3], 0.0));
            for (s, x) in a.0.iter_mut().zip(p) { *s += x }
            a.1 += 1.0;
        }
        if out.nodes[j].detail == "folded" {
            let (d, e) = (&mut out.nodes[j], &n.energy);
            d.rank += n.rank;
            d.requires_change |= n.requires_change;
            d.energy.measured_j += e.measured_j;
            d.energy.simulated_j += e.simulated_j;
            d.energy.pair_meas_j += e.pair_meas_j;
            d.energy.pair_sim_j += e.pair_sim_j;
            d.energy.actions += e.actions;
            d.energy.anomaly |= e.anomaly;
            d.energy.synthetic |= e.synthetic;
            if let Some(x) = e.joules { d.energy.joules = Some(d.energy.joules.unwrap_or(0.0) + x) }
        }
    }
    for (j, (s, c)) in acc {
        if out.nodes[j].detail == "folded" {
            opos.insert(out.nodes[j].id.clone(), [s[0] / c, s[1] / c, s[2] / c]);
        }
    }
    for n in &out.nodes {
        if n.detail != "folded" && let Some(p) = pos.get(&n.id) { opos.insert(n.id.clone(), *p); }
    }
    let mut seen: BTreeMap<(usize, usize, String), usize> = BTreeMap::new();
    for e in &g.edges {
        let (a, b) = (map[e.a], map[e.b]);
        if a == b { continue }
        let k = (a, b, format!("{:?}", e.kind));
        match seen.get(&k) {
            Some(&x) => out.edges[x].weight += e.weight,
            None => { seen.insert(k, out.edges.len()); out.edges.push(Edge { a, b, ..e.clone() }) }
        }
    }
    (out, opos)
}

/// Mini-map: every node projected top-down (x, z) into `[0,1]^2`, plus the
/// camera target, so the host can draw an overview with a "you are here".
pub fn minimap(pos: &Positions, cam: &Orbit) -> (Vec<[f32; 2]>, [f32; 2]) {
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for p in pos.values() {
        lo = [lo[0].min(p[0]), lo[1].min(p[2])];
        hi = [hi[0].max(p[0]), hi[1].max(p[2])];
    }
    let f = |x: f32, z: f32| {
        let w = (hi[0] - lo[0]).max(1e-6);
        let h = (hi[1] - lo[1]).max(1e-6);
        [((x - lo[0]) / w).clamp(0.0, 1.0), ((z - lo[1]) / h).clamp(0.0, 1.0)]
    };
    (pos.values().map(|p| f(p[0], p[2])).collect(), f(cam.target[0], cam.target[2]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::EdgeKind;
    fn n(id: &str, file: &str, j: f64) -> Node {
        Node { id: id.into(), label: id.into(), kind: Kind::Fn, file: Some(file.into()), line: Some(1),
            requires_change: false, rank: 1.0, energy: crate::graph::Energy { joules: Some(j), ..Default::default() },
            live: None, detail: String::new() }
    }
    fn g() -> (Graph, Positions) {
        let mut g = Graph::default();
        let a = g.add(n("a", "x.rce", 1.0));
        let b = g.add(n("b", "x.rce", 2.0));
        let c = g.add(n("c", "y.rce", 4.0));
        g.link(a, c, EdgeKind::Call, 1.0, "");
        g.link(b, c, EdgeKind::Call, 1.0, "");
        g.link(a, b, EdgeKind::Call, 1.0, "");
        let p = [("a", [0.0, 0.0, 0.0]), ("b", [2.0, 0.0, 2.0]), ("c", [4.0, 0.0, 0.0])]
            .into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        (g, p)
    }
    #[test]
    fn fold_sums_energy_and_merges_edges() {
        let (g, p) = g();
        let (f, fp) = fold_files(&g, &p);
        assert_eq!(f.nodes.len(), 2);
        assert_eq!(f.nodes[0].energy.joules, Some(3.0));
        assert_eq!(f.edges.len(), 1);
        assert_eq!(f.edges[0].weight, 2.0);
        assert_eq!(fp["file:x.rce"], [1.0, 0.0, 1.0]);
    }
    #[test]
    fn expanded_file_keeps_its_definitions() {
        let (g, p) = g();
        let ex = ["x.rce".to_string()].into_iter().collect();
        let (f, fp) = fold_files_except(&g, &p, &ex);
        let ids: Vec<&str> = f.nodes.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b", "file:y.rce"]);
        assert_eq!(fp["a"], [0.0, 0.0, 0.0]);
        assert_eq!(folded_file(&f.nodes[2]), Some("y.rce"));
        assert_eq!(folded_file(&f.nodes[0]), None);
        assert_eq!(f.edges.len(), 3);
    }
    #[test]
    fn mapping_normalises_and_minimap_fits() {
        let (g, p) = g();
        let v = normalised(&g, Metric::Joules);
        assert_eq!(v, vec![Some(0.0), Some(1.0 / 3.0), Some(1.0)]);
        assert_eq!(normalised(&g, Metric::Watts), vec![None, None, None]);
        let (pts, _) = minimap(&p, &Orbit::default());
        assert!(pts.iter().all(|q| (0.0..=1.0).contains(&q[0]) && (0.0..=1.0).contains(&q[1])));
        let cam = Orbit { dist: 100.0, ..Default::default() };
        assert_eq!(detail(&cam, 5.0), Detail::Files);
    }
}
