//! SCIP export (Sourcegraph's code intelligence index, scip.proto) of the
//! Form graph's definitions and call references, hand-encoded protobuf so
//! the tree stays pure Rust with no codegen. Ranges use the `range` field
//! (1, packed int32 [line, start, end], zero-based), which scip.proto marks
//! deprecated in favour of typed ranges but still documents for readers.
use crate::graph::{EdgeKind, Graph, Kind};
use std::collections::BTreeMap;

fn varint(mut v: u64, o: &mut Vec<u8>) {
    while v >= 0x80 { o.push((v as u8) | 0x80); v >>= 7 }
    o.push(v as u8);
}
fn tag(f: u32, wt: u8, o: &mut Vec<u8>) { varint(((f as u64) << 3) | wt as u64, o) }
fn bytes(f: u32, b: &[u8], o: &mut Vec<u8>) { tag(f, 2, o); varint(b.len() as u64, o); o.extend_from_slice(b) }
fn int(f: u32, v: u64, o: &mut Vec<u8>) { if v != 0 { tag(f, 0, o); varint(v, o) } }

fn esc(s: &str) -> String {
    if s.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '+' || c == '-' || c == '$') { s.into() }
    else { format!("`{}`", s.replace('`', "``")) }
}

/// SCIP symbol for a node: `rce . . . <file>/<name><suffix>`.
pub fn symbol(file: &str, name: &str, kind: Kind) -> String {
    let suf = match kind { Kind::Type | Kind::State | Kind::Event | Kind::Error => "#", Kind::Fn | Kind::Step => "().", _ => "." };
    format!("rce . . . {}/{}{suf}", esc(file), esc(name))
}

fn occurrence(line: usize, name_len: usize, sym: &str, def: bool) -> Vec<u8> {
    let mut o = Vec::new();
    let mut r = Vec::new();
    for v in [line.saturating_sub(1) as u64, 0, name_len as u64] { varint(v, &mut r) }
    bytes(1, &r, &mut o);
    bytes(2, sym.as_bytes(), &mut o);
    int(3, def as u64, &mut o);
    o
}

type Docs<'a> = BTreeMap<&'a str, (Vec<Vec<u8>>, Vec<Vec<u8>>)>;

pub fn export(g: &Graph, project_root: &str) -> Vec<u8> {
    let mut docs: Docs = BTreeMap::new();
    let sym = |i: usize| {
        let n = &g.nodes[i];
        n.file.as_deref().map(|f| symbol(f, &n.label, n.kind))
    };
    for (i, n) in g.nodes.iter().enumerate() {
        let (Some(f), Some(l), Some(s)) = (n.file.as_deref(), n.line, sym(i)) else { continue };
        if n.kind.is_hw() { continue }
        let d = docs.entry(f).or_default();
        d.0.push(occurrence(l, n.label.len(), &s, true));
        let mut si = Vec::new();
        bytes(1, s.as_bytes(), &mut si);
        if !n.detail.is_empty() { bytes(3, n.detail.as_bytes(), &mut si) }
        bytes(6, n.label.as_bytes(), &mut si);
        d.1.push(si);
    }
    // A call edge becomes a reference occurrence at the caller's line.
    for e in g.edges.iter().filter(|e| e.kind == EdgeKind::Call) {
        let (a, b) = (&g.nodes[e.a], &g.nodes[e.b]);
        let (Some(f), Some(l), Some(s)) = (a.file.as_deref(), a.line, sym(e.b)) else { continue };
        docs.entry(f).or_default().0.push(occurrence(l, b.label.len(), &s, false));
    }
    let mut idx = Vec::new();
    let mut meta = Vec::new();
    let mut tool = Vec::new();
    bytes(1, b"rce-graph3d", &mut tool);
    bytes(2, env!("CARGO_PKG_VERSION").as_bytes(), &mut tool);
    bytes(2, &tool, &mut meta);
    bytes(3, project_root.as_bytes(), &mut meta);
    int(4, 1, &mut meta); // UTF8
    bytes(1, &meta, &mut idx);
    for (f, (occ, syms)) in docs {
        let mut d = Vec::new();
        bytes(1, f.as_bytes(), &mut d);
        for o in occ { bytes(2, &o, &mut d) }
        for s in syms { bytes(3, &s, &mut d) }
        bytes(4, b"rce", &mut d);
        bytes(2, &d, &mut idx);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Node;
    #[test]
    fn encodes_documents_and_symbols() {
        let mut g = Graph::default();
        let mk = |l: &str, line| Node { id: l.into(), label: l.into(), kind: Kind::Fn, file: Some("src/a.rce".into()),
            line: Some(line), fit: false, rank: 0.0, energy: Default::default(), live: None, detail: String::new() };
        let a = g.add(mk("main", 3));
        let b = g.add(mk("leaf", 9));
        g.link(a, b, EdgeKind::Call, 1.0, "");
        let v = export(&g, "file:///w");
        assert_eq!(v[0], 0x0a); // field 1 (metadata), length-delimited
        let s = String::from_utf8_lossy(&v);
        assert!(s.contains("rce . . . `src/a.rce`/leaf()."));
        assert!(s.contains("src/a.rce"));
        assert_eq!(symbol("a", "T", Kind::Type), "rce . . . a/T#");
        let mut o = Vec::new(); varint(300, &mut o); assert_eq!(o, [0xac, 0x02]);
    }
}
