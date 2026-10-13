//! The 3D graph mode linked to the editor: a click on a node moves the
//! editor cursor to its definition; the editor cursor highlights its node.
#![cfg(feature = "gpu")]
use rce_ide::graph_panel::{self, GraphView};
use rce_ide::scene::R;

const SRC: &str = "state Counter\n    n: u64 = 0\n\nevent Inc\nevent Dec\n\nerror Underflow\n\nstep Counter on Inc -> n + 1\nstep Counter on Dec -> try n.checked_sub(1) else Underflow\n";

#[test]
fn click_jumps_editor_and_cursor_highlights() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("counter.rce"), SRC).unwrap();
    let o = rce_ide::app::Options {
        root: d.path().into(),
        file: "counter.rce".into(),
        lsp: None,
        memory: false,
    };
    let mut ide = rce_ide::app::open(&o).unwrap();
    let tl = graph_panel::timeline_from_ide(&ide, &[], &[], None);
    assert_eq!(tl.versions.len(), 1);
    let mut v = GraphView::new(tl, false);
    let area = R {
        x: 0.0,
        y: 0.0,
        w: 1200.0,
        h: 800.0,
    };
    let n = v
        .snap
        .graph
        .nodes
        .iter()
        .position(|n| n.label == "Dec")
        .unwrap();
    let id = v.snap.graph.nodes[n].id.clone();
    let (x, y, _) = v
        .cam
        .project(GraphView::viewport(area), v.pos[&id])
        .unwrap();
    let (file, line) = v.click(area, x, y).expect("a code node under the click");
    assert_eq!(
        v.style.selected.map(|i| v.snap.graph.nodes[i].kind),
        Some(rce_graph3d::Kind::Event)
    );
    assert_eq!(file, "counter.rce");
    graph_panel::jump(&mut ide, &file, line).unwrap();
    let cur = ide.buffer.visible().cursor.0 + 1;
    assert_eq!(cur, line);
    // The editor cursor on line 9 (the Inc step) highlights that step.
    v.set_cursor("counter.rce", 9);
    let c = v.style.cursor.unwrap();
    assert_eq!(v.snap.graph.nodes[c].label, "step Counter");
    assert_eq!(v.snap.graph.nodes[c].line, Some(9));
    // The view never writes the Form.
    assert_eq!(
        std::fs::read_to_string(d.path().join("counter.rce")).unwrap(),
        SRC
    );
}

#[test]
fn scrubber_click_replays_time() {
    use rce_graph3d::time::{Timeline, Version};
    let p = |s: &str| vec![("a.rce".to_string(), s.to_string())];
    let mut tl = Timeline::default();
    tl.versions.push(Version {
        t_ms: 0,
        label: "v1".into(),
        graph: rce_graph3d::native::code_graph_of(&p("event A\n")),
    });
    tl.versions.push(Version {
        t_ms: 1000,
        label: "v2".into(),
        graph: rce_graph3d::native::code_graph_of(&p("event A\nevent B\n")),
    });
    let mut v = GraphView::new(tl, false);
    assert_eq!(v.snap.version, Some(1));
    let area = R {
        x: 0.0,
        y: 0.0,
        w: 1000.0,
        h: 600.0,
    };
    assert_eq!(v.click(area, 20.0, 590.0), None);
    assert_eq!(v.snap.version, Some(0));
    assert_eq!(
        v.snap
            .graph
            .nodes
            .iter()
            .filter(|n| n.id.starts_with("code:"))
            .count(),
        1
    );
    v.click(area, 990.0, 590.0);
    assert_eq!(v.snap.version, Some(1));
    assert!(v.snap.added.iter().any(|i| i.ends_with(":B")));
}
