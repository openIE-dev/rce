//! Import an hwloc topology XML (`lstopo topo.xml`, hwloc 2.x format) as a
//! discovery input: Machine, Package, Group, NUMANode, Core, caches and I/O
//! objects become hardware nodes with containment edges. PUs (hardware
//! threads) are counted on their core rather than drawn. Misc objects are
//! skipped. A small hand-written tag scanner keeps the tree pure Rust.
use crate::graph::{EdgeKind, Graph, Kind, Node};
use std::collections::BTreeMap;

fn attrs(tag: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    let mut rest = tag;
    while let Some(eq) = rest.find("=\"") {
        let key = rest[..eq].rsplit(|c: char| c.is_whitespace()).next().unwrap_or("").to_string();
        let v0 = eq + 2;
        let Some(len) = rest[v0..].find('"') else { break };
        let v = rest[v0..v0 + len].replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&");
        m.insert(key, v);
        rest = &rest[v0 + len + 1..];
    }
    m
}

fn kind(a: &BTreeMap<String, String>) -> Option<(Kind, String)> {
    let t = a.get("type")?.as_str();
    let sub = a.get("subtype").cloned().unwrap_or_default();
    Some(match t {
        "Machine" => (Kind::System, "machine".into()),
        "Package" => (Kind::Package, "package".into()),
        "Die" | "Group" => (Kind::Cluster, if sub.is_empty() { t.to_lowercase() } else { sub }),
        "NUMANode" => (Kind::Memory, "numa".into()),
        "MemCache" => (Kind::Cache, "memcache".into()),
        "Core" => (Kind::Core, "core".into()),
        x if x.ends_with("Cache") => (Kind::Cache, x.trim_end_matches("Cache").to_string()),
        "Bridge" | "PCIDev" => (Kind::Bus, t.to_lowercase()),
        "OSDev" => {
            let ot = a.get("osdev_type").map(String::as_str).unwrap_or("");
            match (ot, sub.as_str()) {
                ("0", _) => (Kind::Storage, "block".into()),
                ("1", _) | (_, "GPU") | (_, "CUDA") | (_, "OpenCL") => (Kind::Gpu, "gpu".into()),
                ("2", _) | ("3", _) => (Kind::Net, "net".into()),
                ("5", _) => (Kind::Accel, "coproc".into()),
                _ => (Kind::Accel, "osdev".into()),
            }
        }
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

pub fn import(xml: &str) -> Result<Graph, Error> {
    if !xml.contains("<topology") {
        return Err(Error("not an hwloc topology XML".into()));
    }
    let mut g = Graph::default();
    // Stack of (graph index or None for skipped objects).
    let mut stack: Vec<Option<usize>> = vec![];
    let mut pus: BTreeMap<usize, u32> = BTreeMap::new();
    let mut i = 0;
    while let Some(o) = xml[i..].find('<') {
        let s = i + o;
        let Some(e) = xml[s..].find('>') else { return Err(Error("unterminated tag".into())) };
        let tag = &xml[s + 1..s + e];
        i = s + e + 1;
        let self_close = tag.ends_with('/');
        if let Some(rest) = tag.strip_prefix("object") {
            let a = attrs(rest);
            let parent = stack.iter().rev().flatten().next().copied();
            let idx = if a.get("type").map(String::as_str) == Some("PU") {
                if let Some(p) = parent { *pus.entry(p).or_default() += 1 }
                None
            } else if let Some((k, what)) = kind(&a) {
                let gp = a.get("gp_index").or(a.get("id")).cloned().unwrap_or_else(|| g.nodes.len().to_string());
                let os = a.get("os_index").map(|x| format!(" #{x}")).unwrap_or_default();
                let name = a.get("name").cloned();
                let mut detail = vec![format!("hwloc {}", a["type"])];
                if let Some(sz) = a.get("cache_size").or(a.get("local_memory")) { detail.push(format!("{sz} bytes")) }
                if let Some(c) = a.get("cpuset") { detail.push(format!("cpuset {c}")) }
                let n = g.add(Node {
                    id: format!("hw:hwloc:{gp}"), label: name.unwrap_or(format!("{what}{os}")), kind: k,
                    file: None, line: None, refused: false, rank: 0.0, energy: Default::default(), live: None,
                    detail: detail.join(", "),
                });
                if let Some(p) = parent { g.link(p, n, EdgeKind::Contains, 1.0, "") }
                Some(n)
            } else {
                None
            };
            if !self_close { stack.push(idx) }
        } else if tag.starts_with("/object") {
            stack.pop();
        } else if let Some(rest) = tag.strip_prefix("info") {
            let a = attrs(rest);
            if let (Some(Some(n)), Some(k), Some(v)) = (stack.last(), a.get("name"), a.get("value"))
                && (k == "CPUModel" || k == "GPUModel" || k == "PCIVendor" || k == "PCIDevice") {
                    g.nodes[*n].detail.push_str(&format!(", {k} {v}"));
                }
        }
    }
    for (n, c) in pus { g.nodes[n].detail.push_str(&format!(", {c} PU")) }
    if g.nodes.is_empty() { return Err(Error("no objects".into())) }
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn imports_hwloc_test_topology() {
        let g = import(include_str!("../tests/data/hwloc-16-2gr2gr2n2c+misc.xml")).unwrap();
        let count = |k: Kind| g.nodes.iter().filter(|n| n.kind == k).count();
        assert_eq!(count(Kind::System), 1);
        assert_eq!(count(Kind::Core), 16);
        assert_eq!(count(Kind::Memory), 8);
        assert_eq!(count(Kind::Cluster), 14);
        assert!(g.nodes.iter().filter(|n| n.kind == Kind::Core).all(|n| n.detail.ends_with("1 PU")));
        // Every non-root node has exactly one container.
        for i in 1..g.nodes.len() {
            assert_eq!(g.edges.iter().filter(|e| e.b == i && e.kind == EdgeKind::Contains).count(), 1);
        }
        assert!(import("<html/>").is_err());
    }
}
