//! Embedding-space neighbours (from any embedding model) as `Near` edges between code
//! nodes. The pairs come from outside (ids + cosine); unknown ids are skipped.
use crate::graph::{EdgeKind, Graph};

/// Add a `Near` edge for each `(id_a, id_b, cosine)` whose ids are in the
/// graph. Returns how many edges were added.
pub fn link(g: &mut Graph, pairs: &[(String, String, f32)]) -> usize {
    let ids: std::collections::BTreeMap<String, usize> = g.ids().into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let mut n = 0;
    for (a, b, c) in pairs {
        if let (Some(&i), Some(&j)) = (ids.get(a), ids.get(b)) {
            g.link(i, j, EdgeKind::Near, *c as f64, &format!("embedding cos {c:.2}"));
            n += 1;
        }
    }
    n
}
