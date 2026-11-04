//! Integration test: round-trip a real `rce_graph::graph::FunctionGraph`
//! through `.fg` encode/decode and verify content-hash stability.

#![cfg(feature = "with-rce-graph")]

use rce_format::{graph_section, FgFile, SectionKind};
use rce_graph::graph::{FunctionGraph, GraphNode, OpKind, Param, Wire, WireKind};
use std::io::Cursor;

fn sample_graph() -> FunctionGraph {
    // Tiny graph: two f32 params x, y; add; returns f32.
    FunctionGraph {
        name: "add".into(),
        nodes: vec![
            GraphNode::Param(Param {
                name: "x".into(),
                data_type: "f32".into(),
            }),
            GraphNode::Param(Param {
                name: "y".into(),
                data_type: "f32".into(),
            }),
            GraphNode::Apply {
                op: OpKind::Add,
                output_type: "f32".into(),
            },
        ],
        wires: vec![
            Wire {
                from: 0,
                to: 2,
                port_in_idx: 0,
                wire_kind: WireKind::Move,
            },
            Wire {
                from: 1,
                to: 2,
                port_in_idx: 1,
                wire_kind: WireKind::Move,
            },
        ],
        output_node: 2,
        return_type: "f32".into(),
    }
}

#[test]
fn function_graph_roundtrips_through_fg_file() {
    let graph = sample_graph();
    let section = graph_section::graph_section_from(&graph).unwrap();
    let mut file = FgFile::new(vec![section]);

    let mut buf = Cursor::new(Vec::<u8>::new());
    file.write(&mut buf).unwrap();
    let original_hash = file.header.content_hash;

    buf.set_position(0);
    let read = FgFile::read(buf).unwrap();
    assert_eq!(read.header.content_hash, original_hash);

    let got = read.section(SectionKind::Graph).unwrap();
    let graph2: FunctionGraph = graph_section::graph_from_section(got).unwrap();
    assert_eq!(graph2.name, graph.name);
    assert_eq!(graph2.nodes.len(), graph.nodes.len());
    assert_eq!(graph2.wires.len(), graph.wires.len());
    assert_eq!(graph2.output_node, graph.output_node);
    assert_eq!(graph2.return_type, graph.return_type);
}

#[test]
fn deterministic_content_hash() {
    // Encoding the same graph twice must produce the same content hash.
    let g = sample_graph();
    let s1 = graph_section::graph_section_from(&g).unwrap();
    let s2 = graph_section::graph_section_from(&g).unwrap();
    let mut f1 = FgFile::new(vec![s1]);
    let mut f2 = FgFile::new(vec![s2]);
    let mut b1 = Cursor::new(Vec::<u8>::new());
    let mut b2 = Cursor::new(Vec::<u8>::new());
    f1.write(&mut b1).unwrap();
    f2.write(&mut b2).unwrap();
    assert_eq!(f1.header.content_hash, f2.header.content_hash);
}
