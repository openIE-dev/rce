//! Code property graph queries over the Form graph (Joern/CodeQL-style,
//! much smaller). One query per line:
//!   callers NAME | callees NAME | reach NAME | path A B
//!   kind KIND [joules > J] [divergence > D] | change | uses-hw NAME
//! Names match node labels. Results are node indices, in graph order
//! except `path`, which is in walk order.
use crate::graph::{EdgeKind, Graph};
use std::collections::{BTreeSet, VecDeque};

fn by_label(g: &Graph, l: &str) -> Result<usize, String> {
    g.nodes.iter().position(|n| n.label == l || n.id == l).ok_or(format!("no node {l}"))
}

fn flows(k: EdgeKind) -> bool {
    matches!(k, EdgeKind::Call | EdgeKind::Triggers | EdgeKind::Writes | EdgeKind::Ref | EdgeKind::Raises)
}

fn out(g: &Graph, i: usize) -> impl Iterator<Item = usize> + '_ {
    g.edges.iter().filter(move |e| e.a == i && flows(e.kind)).map(|e| e.b)
}

pub fn run(g: &Graph, q: &str) -> Result<Vec<usize>, String> {
    let w: Vec<&str> = q.split_whitespace().collect();
    let set = |s: BTreeSet<usize>| s.into_iter().collect::<Vec<_>>();
    match w.as_slice() {
        ["callers", n] => {
            let t = by_label(g, n)?;
            Ok(set(g.edges.iter().filter(|e| e.b == t && flows(e.kind)).map(|e| e.a).collect()))
        }
        ["callees", n] => Ok(set(out(g, by_label(g, n)?).collect())),
        ["reach", n] => {
            let s = by_label(g, n)?;
            let mut seen = BTreeSet::new();
            let mut q = VecDeque::from([s]);
            while let Some(i) = q.pop_front() {
                for j in out(g, i) { if j != s && seen.insert(j) { q.push_back(j) } }
            }
            Ok(set(seen))
        }
        ["path", a, b] => {
            let (a, b) = (by_label(g, a)?, by_label(g, b)?);
            let mut prev = vec![usize::MAX; g.nodes.len()];
            let mut q = VecDeque::from([a]);
            prev[a] = a;
            while let Some(i) = q.pop_front() {
                if i == b { break }
                for j in out(g, i) { if prev[j] == usize::MAX { prev[j] = i; q.push_back(j) } }
            }
            if prev[b] == usize::MAX { return Ok(vec![]) }
            let mut p = vec![b];
            while *p.last().unwrap() != a { p.push(prev[*p.last().unwrap()]) }
            p.reverse();
            Ok(p)
        }
        ["change"] => Ok((0..g.nodes.len()).filter(|&i| g.nodes[i].requires_change).collect()),
        ["uses-hw", n] => {
            let s = by_label(g, n)?;
            Ok(set(g.edges.iter().filter(|e| e.a == s && e.kind == EdgeKind::Ran).map(|e| e.b).collect()))
        }
        ["kind", k, rest @ ..] => {
            let mut v: Vec<usize> = (0..g.nodes.len()).filter(|&i| g.nodes[i].kind.as_str() == *k).collect();
            let mut r = rest;
            while let [m, ">", x, tail @ ..] = r {
                let x: f64 = x.parse().map_err(|_| format!("bad number {x}"))?;
                v.retain(|&i| {
                    let e = &g.nodes[i].energy;
                    match *m {
                        "joules" => e.joules.is_some_and(|j| j > x),
                        "divergence" => e.divergence().is_some_and(|d| d.abs() > x),
                        _ => false,
                    }
                });
                r = tail;
            }
            if !r.is_empty() { return Err(format!("cannot read filter {:?}", r)) }
            Ok(v)
        }
        _ => Err(format!("unknown query: {q}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Kind, Node};
    fn n(l: &str) -> Node {
        Node { id: format!("code:a:{l}"), label: l.into(), kind: Kind::Fn, file: Some("a.rce".into()), line: Some(1),
            requires_change: l == "bad", rank: 0.0, energy: Default::default(), live: None, detail: String::new() }
    }
    #[test]
    fn queries() {
        let mut g = Graph::default();
        let (a, b, c, d) = (g.add(n("a")), g.add(n("b")), g.add(n("c")), g.add(n("bad")));
        g.link(a, b, EdgeKind::Call, 1.0, "");
        g.link(b, c, EdgeKind::Call, 1.0, "");
        g.nodes[c].energy.joules = Some(5.0);
        assert_eq!(run(&g, "callers b").unwrap(), vec![a]);
        assert_eq!(run(&g, "reach a").unwrap(), vec![b, c]);
        assert_eq!(run(&g, "path a c").unwrap(), vec![a, b, c]);
        assert_eq!(run(&g, "path c a").unwrap(), Vec::<usize>::new());
        assert_eq!(run(&g, "change").unwrap(), vec![d]);
        assert_eq!(run(&g, "kind fn joules > 1").unwrap(), vec![c]);
        assert!(run(&g, "frob").is_err());
    }
}
