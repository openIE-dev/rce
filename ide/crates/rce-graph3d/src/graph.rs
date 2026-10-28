//! The graph: code nodes from the Form, hardware nodes from discovery, and
//! the edges between them. Plain data (serde) so the browser build receives
//! exactly what the native side built.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    // Code.
    State,
    Event,
    Step,
    Fn,
    Type,
    Error,
    /// Other Form items (intent, const, impl, actor, extern).
    Other,
    // Hardware.
    System,
    Package,
    Cluster,
    Core,
    Cache,
    Memory,
    Gpu,
    Npu,
    Accel,
    Storage,
    Net,
    Bus,
}

impl Kind {
    pub fn is_hw(self) -> bool {
        self >= Kind::System
    }
    /// Map an rce-memory def kind (`FormIndex::defs[i].kind`).
    pub fn of_def(k: &str) -> Kind {
        match k {
            "state" => Kind::State,
            "event" => Kind::Event,
            "step" => Kind::Step,
            "fn" => Kind::Fn,
            "type" | "struct" | "kind" | "trait" | "protocol" => Kind::Type,
            "error" => Kind::Error,
            _ => Kind::Other,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::State => "state",
            Kind::Event => "event",
            Kind::Step => "step",
            Kind::Fn => "fn",
            Kind::Type => "type",
            Kind::Error => "error",
            Kind::Other => "item",
            Kind::System => "system",
            Kind::Package => "cpu package",
            Kind::Cluster => "cluster",
            Kind::Core => "core",
            Kind::Cache => "cache",
            Kind::Memory => "memory",
            Kind::Gpu => "gpu",
            Kind::Npu => "npu",
            Kind::Accel => "accelerator",
            Kind::Storage => "storage",
            Kind::Net => "network",
            Kind::Bus => "bus",
        }
    }
    /// Layer index for the layered-by-kind layout.
    pub fn layer(self) -> u32 {
        self as u32
    }
    /// Base colour (linear-ish sRGB values, alpha 1).
    pub fn color(self) -> [f32; 4] {
        match self {
            Kind::State => [0.35, 0.65, 1.0, 1.0],
            Kind::Event => [1.0, 0.78, 0.3, 1.0],
            Kind::Step => [0.45, 0.9, 0.55, 1.0],
            Kind::Fn => [0.75, 0.55, 1.0, 1.0],
            Kind::Type => [0.4, 0.85, 0.85, 1.0],
            Kind::Error => [1.0, 0.4, 0.4, 1.0],
            Kind::Other => [0.65, 0.65, 0.7, 1.0],
            _ => [0.62, 0.66, 0.74, 1.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeKind {
    /// A name reference (type use, field, pattern).
    Ref,
    /// A call.
    Call,
    /// Data flow: an event triggers a step (event -> step).
    Triggers,
    /// Data flow / ownership: a step owns the update of its state region (step -> state).
    Writes,
    /// A step can raise an error (step -> error).
    Raises,
    /// The checker reported the file this item lives in (error node -> item).
    Reports,
    /// Hardware containment (package -> core, core -> cache).
    Contains,
    /// Hardware link (memory bus, PCIe, network), labelled with bandwidth when probed.
    Link,
    /// Code ran on / spent energy on this hardware (code -> hardware), from receipts.
    Ran,
    /// Learned similarity, not a reference: the two Form nodes are nearest
    /// neighbours in an embedding space (weight = cosine).
    Near,
}

/// Energy on a node, with where the number came from.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Energy {
    /// Joules: sum of E_meas where present, else E_sim.
    pub joules: Option<f64>,
    /// Joules that are hardware measurements (E_meas).
    pub measured_j: f64,
    /// Joules that are predictions (E_sim) because nothing was measured.
    pub simulated_j: f64,
    /// Actions attributed.
    pub actions: u32,
    /// Some attributed |delta| exceeded k sigma.
    pub anomaly: bool,
    /// The numbers are synthetic demo data, not receipts.
    pub synthetic: bool,
    /// For actions that carried both numbers: summed E_meas and E_sim, so
    /// the view can show how far the model is from the meter per node.
    #[serde(default)]
    pub pair_meas_j: f64,
    #[serde(default)]
    pub pair_sim_j: f64,
}

impl Energy {
    /// log2(E_meas / E_sim) over paired actions: 0 means the model matched,
    /// +1 means the meter read twice the prediction. None without pairs.
    pub fn divergence(&self) -> Option<f64> {
        (self.pair_meas_j > 0.0 && self.pair_sim_j > 0.0)
            .then(|| (self.pair_meas_j / self.pair_sim_j).log2())
    }
}

/// Live telemetry for a hardware node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Live {
    pub watts: Option<f64>,
    pub joules: Option<f64>,
    pub util: Option<f64>,
    /// Source tag (`rapl`, `hwmon`, `smc`, ...) and its accuracy note.
    pub source: Option<String>,
    pub accuracy: Option<String>,
    /// `available`, `needs setup: ...`, `unsupported: ...`, `stale: ...`, `no source`.
    pub status: String,
    pub synthetic: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Stable identity across versions (`code:<file>:<name>`, `hw:<path>`).
    pub id: String,
    pub label: String,
    pub kind: Kind,
    pub file: Option<String>,
    /// 1-based line of the definition.
    pub line: Option<usize>,
    /// The checker reported this node's file (`fit`).
    #[serde(default)]
    pub fit: bool,
    /// Rank (PageRank for code), drives the node radius.
    #[serde(default)]
    pub rank: f64,
    #[serde(default)]
    pub energy: Energy,
    #[serde(default)]
    pub live: Option<Live>,
    /// Free detail (cache size, link speed, device id). `declared` marks a
    /// part known to exist from the platform but not probed.
    #[serde(default)]
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub kind: EdgeKind,
    /// Count (refs, calls, actions) or joules for `Ran`.
    pub weight: f64,
    #[serde(default)]
    pub label: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    /// Form version hash this graph was built from (empty for hardware-only).
    #[serde(default)]
    pub form_hash: String,
    /// Joules in receipts that named no node.
    #[serde(default)]
    pub unattributed_j: f64,
}

impl Graph {
    pub fn index(&self, id: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n.id == id)
    }
    pub fn ids(&self) -> BTreeMap<&str, usize> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect()
    }
    pub fn add(&mut self, n: Node) -> usize {
        if let Some(i) = self.index(&n.id) {
            return i;
        }
        self.nodes.push(n);
        self.nodes.len() - 1
    }
    pub fn link(&mut self, a: usize, b: usize, kind: EdgeKind, weight: f64, label: &str) {
        if let Some(e) = self
            .edges
            .iter_mut()
            .find(|e| e.a == a && e.b == b && e.kind == kind)
        {
            e.weight += weight;
            return;
        }
        self.edges.push(Edge {
            a,
            b,
            kind,
            weight,
            label: label.into(),
        });
    }
    /// The code node whose definition is the closest at or above `line` in `file`.
    pub fn node_at(&self, file: &str, line: usize) -> Option<usize> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.file.as_deref() == Some(file) && n.line.is_some_and(|l| l <= line))
            .max_by_key(|(i, n)| (n.line, usize::MAX - i))
            .map(|(i, _)| i)
    }
    /// Merge another graph (hardware into code); returns the index offset.
    pub fn merge(&mut self, o: &Graph) -> usize {
        let mut map = vec![];
        let base = self.nodes.len();
        for n in &o.nodes {
            map.push(self.add(n.clone()));
        }
        for e in &o.edges {
            self.link(map[e.a], map[e.b], e.kind, e.weight, &e.label);
        }
        base
    }
}

/// Classify a reference edge by the kinds at its ends.
pub fn edge_kind(from: Kind, to: Kind, call: bool) -> (EdgeKind, bool) {
    // (kind, reversed)
    match (from, to) {
        (Kind::Step, Kind::State) => (EdgeKind::Writes, false),
        (Kind::Step, Kind::Event) => (EdgeKind::Triggers, true),
        (Kind::Step, Kind::Error) => (EdgeKind::Raises, false),
        _ if call => (EdgeKind::Call, false),
        _ => (EdgeKind::Ref, false),
    }
}

/// A def as rce-memory's `FormIndex` reports it (kept as plain data so the
/// extraction is testable and usable without the native services).
#[derive(Debug, Clone)]
pub struct DefIn<'a> {
    pub name: &'a str,
    pub kind: &'a str,
    pub file: &'a str,
    pub line: usize,
    pub rank: f64,
    /// Concurrency annotation (RCE 0.6/0.6b), empty when none.
    pub note: &'a str,
}

/// Build the code graph from defs, refs `(from, to, call)` and per-file
/// checker verdicts (`Some(report)` for a file reported `fit`).
pub fn from_defs(
    defs: &[DefIn],
    refs: &[(usize, usize, bool)],
    verdicts: &BTreeMap<String, Option<String>>,
    form_hash: &str,
) -> Graph {
    let mut g = Graph {
        form_hash: form_hash.into(),
        ..Default::default()
    };
    for d in defs {
        let fit = matches!(verdicts.get(d.file), Some(Some(_)));
        let label = match d.name.strip_prefix("step ") {
            Some(s) => format!("step {}", s.split('@').next().unwrap_or(s)),
            None => d.name.to_string(),
        };
        g.nodes.push(Node {
            id: format!("code:{}:{}", d.file, d.name),
            label,
            kind: Kind::of_def(d.kind),
            file: Some(d.file.into()),
            line: Some(d.line),
            fit,
            rank: d.rank,
            energy: Energy::default(),
            live: None,
            detail: if d.note.is_empty() { d.kind.into() } else { format!("{} · {}", d.kind, d.note) },
        });
    }
    for &(f, t, call) in refs {
        if f >= g.nodes.len() || t >= g.nodes.len() {
            continue;
        }
        let (k, rev) = edge_kind(g.nodes[f].kind, g.nodes[t].kind, call);
        let (a, b) = if rev { (t, f) } else { (f, t) };
        g.link(a, b, k, 1.0, "");
    }
    // A file reported `fit` gets an error node naming the report, linked to its items.
    for (file, v) in verdicts {
        if let Some(msg) = v {
            let members: Vec<usize> = (0..g.nodes.len())
                .filter(|&i| g.nodes[i].file.as_deref() == Some(file.as_str()))
                .collect();
            let first = msg.lines().next().unwrap_or(msg);
            let e = g.add(Node {
                id: format!("check:{file}"),
                label: format!("fit: {}", trunc(first, 48)),
                kind: Kind::Error,
                file: Some(file.clone()),
                line: parse_line(msg),
                fit: true,
                rank: 0.0,
                energy: Energy::default(),
                live: None,
                detail: msg.clone(),
            });
            for m in members {
                g.link(e, m, EdgeKind::Reports, 1.0, "");
            }
        }
    }
    g
}

fn parse_line(msg: &str) -> Option<usize> {
    // Checker messages carry `line N` or `:N:`.
    let i = msg.find("line ")?;
    msg[i + 5..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .ok()
}

pub fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.into()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}
