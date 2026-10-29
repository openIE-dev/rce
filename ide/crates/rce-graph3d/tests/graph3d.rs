use rce_graph3d::camera::{self, Orbit, Viewport};
use rce_graph3d::graph::{EdgeKind, Kind};
use rce_graph3d::layout::{self, Mode, Options, Positions};
use rce_graph3d::overlay::{self, Action};
use rce_graph3d::time::{FactIn, Timeline, Version};
use rce_graph3d::{hw, native};
use std::collections::BTreeMap;

const COUNTER: &str = "state Counter\n    n: u64 = 0\n\nevent Inc\nevent Dec\n\nerror Underflow\n\nstep Counter on Inc -> n + 1\nstep Counter on Dec -> try n.checked_sub(1) else Underflow\n";

fn prog(s: &str) -> Vec<(String, String)> {
    vec![("counter.rce".into(), s.into())]
}

#[test]
fn extraction_kinds_and_dataflow() {
    let g = native::code_graph_of(&prog(COUNTER));
    let k = |l: &str| g.nodes.iter().find(|n| n.label == l).map(|n| n.kind);
    assert_eq!(k("Counter"), Some(Kind::State));
    assert_eq!(k("Inc"), Some(Kind::Event));
    assert_eq!(k("Underflow"), Some(Kind::Error));
    assert_eq!(g.nodes.iter().filter(|n| n.kind == Kind::Step).count(), 2);
    let has = |a: &str, b: &str, kind: EdgeKind| {
        g.edges.iter().any(|e| {
            g.nodes[e.a].label.starts_with(a) && g.nodes[e.b].label.starts_with(b) && e.kind == kind
        })
    };
    assert!(has("step", "Counter", EdgeKind::Writes));
    assert!(has("Inc", "step", EdgeKind::Triggers));
    assert!(has("step", "Underflow", EdgeKind::Raises));
    // Lines are 1-based def lines; node_at maps an editor line to its node.
    let inc = g.nodes.iter().position(|n| n.label == "Inc").unwrap();
    assert_eq!(g.nodes[inc].line, Some(4));
    assert_eq!(g.node_at("counter.rce", 4), Some(inc));
    assert!(!g.nodes.iter().any(|n| n.requires_change));
}

#[test]
fn file_requiring_change_gets_error_node() {
    let g = native::code_graph_of(&prog("state S\n    n: u64 = 0\nstep S on Nope -> n + 1\n"));
    let e = g
        .nodes
        .iter()
        .find(|n| n.id == "check:counter.rce")
        .expect("refusal node");
    assert_eq!(e.kind, Kind::Error);
    assert!(
        g.nodes
            .iter()
            .filter(|n| n.id.starts_with("code:"))
            .all(|n| n.requires_change)
    );
    assert!(g.edges.iter().any(|x| x.kind == EdgeKind::RequiresChange));
}

fn lay(g: &rce_graph3d::Graph, warm: &Positions) -> Positions {
    layout::layout(g, &Options::default(), warm, &BTreeMap::new())
}

#[test]
fn layout_is_deterministic_and_warm_start_is_stable() {
    let g = native::code_graph_of(&prog(COUNTER));
    let a = lay(&g, &Positions::new());
    let b = lay(&g, &Positions::new());
    assert_eq!(a, b, "bit-identical");
    assert!(a.values().all(|p| p.iter().all(|x| x.is_finite())));
    // Add an item: surviving nodes move little.
    let g2 = native::code_graph_of(&prog(&format!(
        "{COUNTER}event Reset\nstep Counter on Reset -> 0\n"
    )));
    let c = lay(&g2, &a);
    let moved: f32 = a
        .iter()
        .map(|(id, p)| {
            let q = c[id];
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
        })
        .fold(0.0, f32::max);
    let cold = lay(&g2, &Positions::new());
    let moved_cold: f32 = a
        .iter()
        .map(|(id, p)| {
            let q = cold[id];
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
        })
        .fold(0.0, f32::max);
    assert!(moved < 2.0, "warm start moved a node by {moved}");
    assert!(moved < moved_cold, "warm {moved} vs cold {moved_cold}");
    // Index order does not matter: ids seed the layout.
    let mut g3 = g.clone();
    g3.nodes.reverse();
    let n = g3.nodes.len();
    for e in g3.edges.iter_mut() {
        e.a = n - 1 - e.a;
        e.b = n - 1 - e.b;
    }
    assert_eq!(lay(&g3, &Positions::new()).len(), a.len());
}

#[test]
fn layered_by_kind_pins_z() {
    let g = native::code_graph_of(&prog(COUNTER));
    let o = Options {
        mode: Mode::ByKind,
        ..Default::default()
    };
    let p = layout::layout(&g, &o, &Positions::new(), &BTreeMap::new());
    for n in &g.nodes {
        assert_eq!(
            p[&n.id][2],
            ((n.kind.layer() as f64 - 3.0) * o.layer_gap) as f32
        );
    }
}

#[test]
fn barnes_hut_scales() {
    // 600 nodes in a ring of chains: finishes and stays finite.
    let mut g = rce_graph3d::Graph::default();
    for i in 0..600 {
        g.add(rce_graph3d::Node {
            id: format!("n{i}"),
            label: String::new(),
            kind: Kind::Fn,
            file: None,
            line: None,
            requires_change: false,
            rank: 0.0,
            energy: Default::default(),
            live: None,
            detail: String::new(),
        });
    }
    for i in 0..600 {
        g.link(i, (i + 1) % 600, EdgeKind::Call, 1.0, "");
    }
    let o = Options {
        iterations: 60,
        ..Default::default()
    };
    let p = layout::layout(&g, &o, &Positions::new(), &BTreeMap::new());
    assert!(p.values().all(|v| v.iter().all(|x| x.is_finite())));
}

#[test]
fn picking_hits_nearest_and_misses_empty_space() {
    let g = native::code_graph_of(&prog(COUNTER));
    let pos = lay(&g, &Positions::new());
    let mut cam = Orbit::default();
    cam.fit(&pos);
    let vp = Viewport {
        x: 100.0,
        y: 50.0,
        w: 800.0,
        h: 600.0,
    };
    for (i, n) in g.nodes.iter().enumerate() {
        let (x, y, _) = cam.project(vp, pos[&n.id]).unwrap();
        let hit = camera::pick(&g, &pos, &cam, vp, x, y).unwrap();
        // Either this node or one in front of it at that pixel.
        let zi = cam.project(vp, pos[&n.id]).unwrap().2;
        let zh = cam.project(vp, pos[&g.nodes[hit].id]).unwrap().2;
        assert!(hit == i || zh <= zi);
    }
    assert_eq!(camera::pick(&g, &pos, &cam, vp, 101.0, 51.0), None);
}

fn act(
    seq: u64,
    t: i64,
    names: &[&str],
    sim: Option<f64>,
    meas: Option<f64>,
    delta: Option<f64>,
) -> Action {
    Action {
        seq,
        t_ms: t,
        action: "check".into(),
        e_sim_j: sim,
        e_meas_j: meas,
        source: meas.map(|_| "rapl".into()),
        delta_j: delta,
        device: None,
        names: names.iter().map(|s| s.to_string()).collect(),
        synthetic: false,
    }
}

#[test]
fn time_scrubbing_versions_facts_receipts() {
    let v1 = native::code_graph_of(&prog(COUNTER));
    let v2 = native::code_graph_of(&prog(&format!(
        "{COUNTER}event Reset\nstep Counter on Reset -> 0\n"
    )));
    let mut cpu = hw::linux(std::path::Path::new("/nonexistent"));
    let pkg = cpu.add(rce_graph3d::Node {
        id: "hw:pkg0".into(),
        label: "CPU package 0".into(),
        kind: Kind::Package,
        file: None,
        line: None,
        requires_change: false,
        rank: 0.0,
        energy: Default::default(),
        live: None,
        detail: String::new(),
    });
    let _ = pkg;
    let mut tl = Timeline {
        versions: vec![
            Version {
                t_ms: 100,
                label: "v1".into(),
                graph: v1,
            },
            Version {
                t_ms: 200,
                label: "v2".into(),
                graph: v2,
            },
        ],
        facts: vec![FactIn {
            subject: "Counter".into(),
            predicate: "owner".into(),
            object: "david".into(),
            valid_from: 150,
            valid_to: Some(250),
            recorded_at: 160,
            expired_at: None,
        }],
        actions: vec![
            act(1, 120, &["Inc"], Some(2.0), None, None),
            act(
                2,
                210,
                &["Reset", "Counter"],
                Some(1.0),
                Some(3.0),
                Some(2.0),
            ),
            act(3, 220, &["Inc"], Some(1.0), Some(1.1), Some(0.1)),
            act(4, 230, &["Dec"], Some(1.0), Some(1.0), Some(0.0)),
            act(5, 240, &["Dec"], Some(1.0), Some(1.05), Some(0.05)),
        ],
        hardware: cpu,
    };
    tl.sort();
    assert_eq!(tl.at(50, 1.5).version, None);
    let s1 = tl.at(150, 1.5);
    assert_eq!(s1.version, Some(0));
    assert!(
        s1.facts.is_empty(),
        "fact recorded at 160 is not known at 150"
    );
    let inc = s1.graph.nodes.iter().find(|n| n.label == "Inc").unwrap();
    assert_eq!(inc.energy.joules, Some(2.0));
    assert_eq!(inc.energy.measured_j, 0.0, "E_sim only");
    let s2 = tl.at(245, 1.4);
    assert_eq!(s2.version, Some(1));
    assert!(s2.added.iter().any(|i| i.ends_with(":Reset")));
    assert!(s2.removed.is_empty());
    assert_eq!(s2.facts["Counter"], vec!["owner david".to_string()]);
    let reset = s2.graph.nodes.iter().find(|n| n.label == "Reset").unwrap();
    assert!(reset.energy.anomaly, "delta 2.0 is > 1.4 sigma");
    assert_eq!(reset.energy.measured_j, 1.5);
    // Measured actions reach the hardware they ran on.
    let p = s2.graph.index("hw:pkg0").unwrap();
    assert!(
        s2.graph
            .edges
            .iter()
            .any(|e| e.kind == EdgeKind::Ran && e.b == p)
    );
    assert!(s2.graph.nodes[p].energy.measured_j > 0.0);
    assert_eq!(tl.at(260, 1.5).facts.len(), 0, "valid_to passed");
    assert_eq!(tl.scrub(0.0), Some(100));
    assert_eq!(
        tl.scrub(1.0),
        Some(250),
        "the fact's valid_to is the last tick"
    );
    let fs = tl.first_seen();
    assert_eq!(fs.values().filter(|v| **v == 1).count(), 2);
}

#[test]
fn no_energy_no_number() {
    let mut g = native::code_graph_of(&prog(COUNTER));
    overlay::apply(&mut g, &[act(1, 1, &["Inc"], None, None, None)], 3.0);
    let inc = g.nodes.iter().find(|n| n.label == "Inc").unwrap();
    assert_eq!(inc.energy.joules, None);
    assert_eq!(inc.energy.actions, 1);
}

fn w(p: &std::path::Path, rel: &str, s: &str) {
    let f = p.join(rel);
    std::fs::create_dir_all(f.parent().unwrap()).unwrap();
    std::fs::write(f, s).unwrap();
}

#[test]
fn linux_topology_from_sysfs() {
    let d = tempfile::tempdir().unwrap();
    let r = d.path();
    w(r, "proc/sys/kernel/hostname", "box\n");
    w(r, "proc/cpuinfo", "model name\t: Test CPU\n");
    w(r, "sys/devices/system/cpu/online", "0-3\n");
    for c in 0..4 {
        let t = format!("sys/devices/system/cpu/cpu{c}");
        w(r, &format!("{t}/topology/physical_package_id"), "0");
        w(r, &format!("{t}/topology/core_id"), &(c / 2).to_string());
        w(r, &format!("{t}/cache/index0/level"), "1");
        w(r, &format!("{t}/cache/index0/type"), "Data");
        w(r, &format!("{t}/cache/index0/size"), "48K");
        w(
            r,
            &format!("{t}/cache/index0/shared_cpu_list"),
            if c < 2 { "0-1" } else { "2-3" },
        );
        w(r, &format!("{t}/cache/index1/level"), "3");
        w(r, &format!("{t}/cache/index1/type"), "Unified");
        w(r, &format!("{t}/cache/index1/size"), "32M");
        w(r, &format!("{t}/cache/index1/shared_cpu_list"), "0-3");
    }
    w(
        r,
        "sys/devices/system/node/node0/meminfo",
        "Node 0 MemTotal:       16384000 kB\n",
    );
    w(r, "sys/devices/system/node/node0/cpulist", "0-3");
    w(r, "sys/bus/pci/devices/0000:01:00.0/class", "0x030000");
    w(r, "sys/bus/pci/devices/0000:01:00.0/vendor", "0x10de");
    w(r, "sys/bus/pci/devices/0000:01:00.0/device", "0x2684");
    w(
        r,
        "sys/bus/pci/devices/0000:01:00.0/current_link_speed",
        "16.0 GT/s PCIe",
    );
    w(
        r,
        "sys/bus/pci/devices/0000:01:00.0/current_link_width",
        "16",
    );
    w(r, "sys/bus/pci/devices/0000:00:1f.0/class", "0x060100"); // bridge: skipped
    let g = hw::linux(r);
    let count = |k: Kind| g.nodes.iter().filter(|n| n.kind == k).count();
    assert_eq!(count(Kind::Package), 1);
    assert_eq!(count(Kind::Core), 2, "SMT siblings share a core");
    assert_eq!(count(Kind::Cache), 3, "two L1d groups, one shared L3");
    assert_eq!(count(Kind::Memory), 1);
    assert_eq!(count(Kind::Gpu), 1);
    assert_eq!(count(Kind::Bus), 1);
    assert!(g.nodes.iter().any(|n| n.label == "NUMA node0 15.6 GiB"));
    assert!(g.edges.iter().any(|e| e.label == "PCIe 16.0 GT/s PCIe x16"));
    // Nothing is invented on an empty root.
    let e = hw::linux(&r.join("empty"));
    assert_eq!(e.nodes.len(), 1, "only the system node");
}

#[test]
fn macos_topology_from_sysctl() {
    let m: BTreeMap<&str, &str> = [
        ("hw.model", "Mac17,1"),
        ("machdep.cpu.brand_string", "Apple M5 Max"),
        ("hw.nperflevels", "2"),
        ("hw.perflevel0.name", "Performance"),
        ("hw.perflevel0.physicalcpu", "12"),
        ("hw.perflevel0.l2cachesize", "16777216"),
        ("hw.perflevel1.name", "Efficiency"),
        ("hw.perflevel1.physicalcpu", "6"),
        ("hw.memsize", "137438953472"),
    ]
    .into();
    let g = hw::macos(&|k| m.get(k).map(|s| s.to_string()));
    assert_eq!(g.nodes.iter().filter(|n| n.kind == Kind::Core).count(), 18);
    assert!(
        g.nodes
            .iter()
            .any(|n| n.label == "unified memory 128.0 GiB")
    );
    let ane = g.nodes.iter().find(|n| n.kind == Kind::Npu).unwrap();
    assert!(ane.detail.starts_with("declared"));
}

#[test]
fn live_never_zero() {
    let mut g = hw::macos(&|k| (k == "machdep.cpu.brand_string").then(|| "Apple M5".into()));
    let rows = vec![rce_graph3d::dash::SourceRow {
        tag: "rapl".into(),
        domain: "cpu".into(),
        status: "unsupported: no powercap".into(),
        ..Default::default()
    }];
    native::attach_live(&mut g, &rows, None);
    let p = g.nodes.iter().find(|n| n.kind == Kind::Package).unwrap();
    let l = p.live.as_ref().unwrap();
    assert_eq!(l.watts, None);
    assert!(l.status.starts_with("unsupported"));
    let prims = rce_graph3d::dash::dashboard(&rows, 0.0, 0.0, 800.0, 400.0, "");
    let texts: Vec<String> = prims
        .iter()
        .filter_map(|p| match p {
            rce_graph3d::dash::Prim::Text { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert!(texts.iter().any(|t| t.starts_with("unsupported")));
    assert!(!texts.iter().any(|t| t.contains(" W") && t.starts_with('0')));
}

#[test]
fn embedding_neighbors_become_near_edges() {
    use rce_graph3d::graph::{from_defs, DefIn};
    let defs = [
        DefIn { name: "Door", kind: "state", file: "d.rce", line: 1, rank: 1.0, note: "" },
        DefIn { name: "Open", kind: "event", file: "d.rce", line: 4, rank: 1.0, note: "" },
    ];
    let mut g = from_defs(&defs, &[], &Default::default(), "h");
    let n = rce_graph3d::neighbors::link(
        &mut g,
        &[("code:d.rce:Door".into(), "code:d.rce:Open".into(), 0.8), ("code:d.rce:Nope".into(), "code:d.rce:Open".into(), 0.5)],
    );
    assert_eq!(n, 1);
    assert!(g.edges.iter().any(|e| e.kind == rce_graph3d::EdgeKind::Near && (e.weight - 0.8).abs() < 1e-6));
}

/// Needs the draft 0.6b syntax: `cargo test -p rce-graph3d --features conc06`.
#[cfg(feature = "conc06")]
#[test]
fn concurrency_06b_constructs_show_in_the_form_graph() {
    let dist = "replicated state Ledger\n    total: i64 = 0\n\ninterrupt event Alarm(code: u8) priority 7\nevent Deposit(amount: i64)\n\nstep Ledger on Deposit(amount: a) deadline 100\n    total += a\n\nstep Ledger on Alarm(code: c)\n    pass\n\nactor Counter\n    n: i64 = 0\n    on Deposit(amount: a)\n        n += a\n\nnode east { Counter }\n";
    let sup = "event Tick(n: i64)\n\nactor Worker\n    n: i64 = 0\n    on Tick(n: k)\n        n += k\n\nsupervisor Root strategy one_for_one\n    child w: Worker restart bound 3\n\nprotocol Ping\n    send text\n    recv i64\n";
    let g = rce_graph3d::native::code_graph_of(&[("d.rce".into(), dist.into()), ("s.rce".into(), sup.into())]);
    let det = |l: &str| g.nodes.iter().find(|n| n.label == l).map(|n| n.detail.clone()).unwrap_or_else(|| panic!("no node {l}: {:?}", g.nodes.iter().map(|n| &n.label).collect::<Vec<_>>()));
    assert_eq!(det("Ledger"), "state · replicated");
    assert_eq!(det("Alarm"), "event · interrupt priority 7");
    assert_eq!(det("Deposit"), "event");
    assert!(g.nodes.iter().any(|n| n.detail == "step · deadline 100 ticks"));
    assert_eq!(det("east"), "node · partition: Counter");
    assert_eq!(det("Root"), "supervisor · supervisor one_for_one, 1 children");
    assert_eq!(det("Ping"), "protocol · session type, 2 steps");
    let ix = |l: &str| g.nodes.iter().position(|n| n.label == l).unwrap();
    let linked = |a: usize, b: usize| g.edges.iter().any(|e| (e.a == a && e.b == b) || (e.a == b && e.b == a));
    assert!(linked(ix("east"), ix("Counter")));
    assert!(linked(ix("Root"), ix("Worker")));
}
