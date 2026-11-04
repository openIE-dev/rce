//! Network/graph metrics: network analysis, community detection,
//! influence measures, network models, link prediction.

use crate::graph_algorithms::Graph;

// ═══════════════════════════════════════════════════════════════════
// NETWORK STATISTICS
// ═══════════════════════════════════════════════════════════════════

/// Average path length (mean of all shortest paths).
pub fn average_path_length(graph: &Graph) -> f64 {
    let n = graph.n_nodes;
    if n <= 1 { return 0.0; }
    let apsp = crate::graph_algorithms::floyd_warshall(graph);
    let mut sum = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in 0..n {
            if i != j && apsp[i][j].is_finite() { sum += apsp[i][j]; count += 1; }
        }
    }
    if count == 0 { return f64::INFINITY; }
    sum / count as f64
}

/// Global clustering coefficient (transitivity).
pub fn global_clustering_coefficient(graph: &Graph) -> f64 {
    let n = graph.n_nodes;
    let mut triangles = 0usize;
    let mut triples = 0usize;
    for u in 0..n {
        let neighbors: Vec<usize> = graph.adj[u].iter().map(|(v, _)| *v).collect();
        let k = neighbors.len();
        triples += k * (k.saturating_sub(1)) / 2;
        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                let vi = neighbors[i];
                let vj = neighbors[j];
                if graph.adj[vi].iter().any(|(w, _)| *w == vj) { triangles += 1; }
            }
        }
    }
    if triples == 0 { return 0.0; }
    triangles as f64 / triples as f64
}

/// Local clustering coefficient for a node.
pub fn local_clustering_coefficient(graph: &Graph, node: usize) -> f64 {
    let neighbors: Vec<usize> = graph.adj[node].iter().map(|(v, _)| *v).collect();
    let k = neighbors.len();
    if k < 2 { return 0.0; }
    let mut links = 0;
    for i in 0..k {
        for j in i + 1..k {
            if graph.adj[neighbors[i]].iter().any(|(w, _)| *w == neighbors[j]) { links += 1; }
        }
    }
    2.0 * links as f64 / (k * (k - 1)) as f64
}

/// Average clustering coefficient.
pub fn average_clustering_coefficient(graph: &Graph) -> f64 {
    let n = graph.n_nodes;
    if n == 0 { return 0.0; }
    (0..n).map(|i| local_clustering_coefficient(graph, i)).sum::<f64>() / n as f64
}

/// Degree distribution. Returns (degree, count) pairs.
pub fn degree_distribution(graph: &Graph) -> Vec<(usize, usize)> {
    let max_deg = graph.adj.iter().map(|a| a.len()).max().unwrap_or(0);
    let mut counts = vec![0usize; max_deg + 1];
    for adj in &graph.adj { counts[adj.len()] += 1; }
    counts.iter().enumerate().filter(|(_, c)| **c > 0).map(|(d, c)| (d, *c)).collect()
}

/// Average degree.
pub fn average_degree(graph: &Graph) -> f64 {
    let n = graph.n_nodes;
    if n == 0 { return 0.0; }
    graph.adj.iter().map(|a| a.len()).sum::<usize>() as f64 / n as f64
}

/// Degree assortativity (Pearson correlation of degrees at edge endpoints).
pub fn degree_assortativity(graph: &Graph) -> f64 {
    let mut sum_xy = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut m = 0.0;

    for u in 0..graph.n_nodes {
        let du = graph.adj[u].len() as f64;
        for (v, _) in &graph.adj[u] {
            if !graph.directed && *v < u { continue; }
            let dv = graph.adj[*v].len() as f64;
            sum_xy += du * dv;
            sum_x += du;
            sum_y += dv;
            sum_x2 += du * du;
            sum_y2 += dv * dv;
            m += 1.0;
        }
    }
    if m < 1.0 { return 0.0; }
    let num = sum_xy / m - (sum_x / m) * (sum_y / m);
    let den = ((sum_x2 / m - (sum_x / m).powi(2)) * (sum_y2 / m - (sum_y / m).powi(2))).sqrt();
    if den < 1e-15 { return 0.0; }
    num / den
}

/// Network efficiency (inverse of average shortest path).
pub fn network_efficiency(graph: &Graph) -> f64 {
    let n = graph.n_nodes;
    if n <= 1 { return 0.0; }
    let apsp = crate::graph_algorithms::floyd_warshall(graph);
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j && apsp[i][j].is_finite() && apsp[i][j] > 0.0 {
                sum += 1.0 / apsp[i][j];
            }
        }
    }
    sum / (n * (n - 1)) as f64
}

// ═══════════════════════════════════════════════════════════════════
// COMMUNITY DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Label propagation community detection. Returns community labels.
pub fn label_propagation(graph: &Graph, max_iter: usize, seed: u64) -> Vec<usize> {
    let n = graph.n_nodes;
    let mut labels: Vec<usize> = (0..n).collect();
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    fn prng(s: &mut [u64; 4]) -> u64 {
        let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45);
        r
    }
    for _ in 0..20 { prng(&mut state); }

    for _ in 0..max_iter {
        let mut changed = false;
        // Random order
        let mut order: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() { let j = prng(&mut state) as usize % (i + 1); order.swap(i, j); }

        for &u in &order {
            if graph.adj[u].is_empty() { continue; }
            // Count neighbor labels
            let mut counts = std::collections::HashMap::new();
            for (v, _) in &graph.adj[u] { *counts.entry(labels[*v]).or_insert(0usize) += 1; }
            let max_count = *counts.values().max().unwrap();
            let candidates: Vec<usize> = counts.iter().filter(|(_, c)| **c == max_count).map(|(l, _)| *l).collect();
            let new_label = candidates[prng(&mut state) as usize % candidates.len()];
            if labels[u] != new_label { labels[u] = new_label; changed = true; }
        }
        if !changed { break; }
    }
    labels
}

/// Modularity of a partition.
pub fn modularity(graph: &Graph, labels: &[usize]) -> f64 {
    let m = graph.n_edges() as f64;
    if m < 1.0 { return 0.0; }
    let mut q = 0.0;
    for u in 0..graph.n_nodes {
        let ku = graph.adj[u].len() as f64;
        for (v, _) in &graph.adj[u] {
            if labels[u] == labels[*v] {
                let kv = graph.adj[*v].len() as f64;
                q += 1.0 - ku * kv / (2.0 * m);
            }
        }
    }
    q / (2.0 * m)
}

/// Number of communities in a labeling.
pub fn n_communities(labels: &[usize]) -> usize {
    let mut unique = labels.to_vec();
    unique.sort(); unique.dedup();
    unique.len()
}

// ═══════════════════════════════════════════════════════════════════
// LINK PREDICTION
// ═══════════════════════════════════════════════════════════════════

/// Common neighbors score for link prediction.
pub fn common_neighbors(graph: &Graph, u: usize, v: usize) -> usize {
    let nu: std::collections::HashSet<usize> = graph.adj[u].iter().map(|(w, _)| *w).collect();
    graph.adj[v].iter().filter(|(w, _)| nu.contains(w)).count()
}

/// Jaccard coefficient for link prediction.
pub fn jaccard_link_score(graph: &Graph, u: usize, v: usize) -> f64 {
    let nu: std::collections::HashSet<usize> = graph.adj[u].iter().map(|(w, _)| *w).collect();
    let nv: std::collections::HashSet<usize> = graph.adj[v].iter().map(|(w, _)| *w).collect();
    let inter = nu.intersection(&nv).count() as f64;
    let union = nu.union(&nv).count() as f64;
    if union < 1e-15 { 0.0 } else { inter / union }
}

/// Adamic-Adar index for link prediction.
pub fn adamic_adar(graph: &Graph, u: usize, v: usize) -> f64 {
    let nu: std::collections::HashSet<usize> = graph.adj[u].iter().map(|(w, _)| *w).collect();
    let nv: std::collections::HashSet<usize> = graph.adj[v].iter().map(|(w, _)| *w).collect();
    nu.intersection(&nv).map(|&w| {
        let deg = graph.adj[w].len() as f64;
        if deg > 1.0 { 1.0 / deg.ln() } else { 0.0 }
    }).sum()
}

/// Preferential attachment score.
pub fn preferential_attachment(graph: &Graph, u: usize, v: usize) -> f64 {
    (graph.adj[u].len() * graph.adj[v].len()) as f64
}

// ═══════════════════════════════════════════════════════════════════
// NETWORK MODELS
// ═══════════════════════════════════════════════════════════════════

/// Erdos-Renyi random graph G(n, p).
pub fn erdos_renyi(n: usize, p: f64, seed: u64) -> Graph {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    fn prng(s: &mut [u64; 4]) -> u64 {
        let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45);
        r
    }
    for _ in 0..20 { prng(&mut state); }
    let mut graph = Graph::new(n, false);
    for i in 0..n {
        for j in i + 1..n {
            let u = (prng(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
            if u < p { graph.add_edge(i, j, 1.0); }
        }
    }
    graph
}

/// Watts-Strogatz small-world graph.
pub fn watts_strogatz(n: usize, k: usize, beta: f64, seed: u64) -> Graph {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    fn prng(s: &mut [u64; 4]) -> u64 {
        let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45);
        r
    }
    for _ in 0..20 { prng(&mut state); }
    let mut graph = Graph::new(n, false);
    // Ring lattice
    for i in 0..n {
        for j in 1..=k / 2 {
            graph.add_edge(i, (i + j) % n, 1.0);
        }
    }
    // Rewire with probability beta
    for i in 0..n {
        for _j in 1..=k / 2 {
            let u = (prng(&mut state) >> 11) as f64 / (1u64 << 53) as f64;
            if u < beta {
                let new_target = prng(&mut state) as usize % n;
                if new_target != i {
                    // Add new edge (simplified — doesn't remove old)
                    graph.add_edge(i, new_target, 1.0);
                }
            }
        }
    }
    graph
}

/// Barabasi-Albert preferential attachment graph.
pub fn barabasi_albert(n: usize, m: usize, seed: u64) -> Graph {
    let mut state = [seed, seed.wrapping_mul(6364136223846793005).wrapping_add(1),
                     seed.wrapping_mul(1442695040888963407).wrapping_add(3),
                     seed.wrapping_mul(2862933555777941757).wrapping_add(7)];
    fn prng(s: &mut [u64; 4]) -> u64 {
        let r = (s[0].wrapping_add(s[3])).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = s[3].rotate_left(45);
        r
    }
    for _ in 0..20 { prng(&mut state); }

    let mut graph = Graph::new(n, false);
    // Start with complete graph on m+1 nodes
    for i in 0..=m {
        for j in i + 1..=m { graph.add_edge(i, j, 1.0); }
    }
    // Add remaining nodes with preferential attachment
    let mut degrees = vec![0usize; n];
    for i in 0..=m { degrees[i] = m; }

    for new_node in m + 1..n {
        let total_degree: usize = degrees.iter().take(new_node).sum();
        let mut added = 0;
        let mut targets = std::collections::HashSet::new();
        while added < m && added < new_node {
            let r = prng(&mut state) as usize % total_degree.max(1);
            let mut cum = 0;
            for j in 0..new_node {
                cum += degrees[j];
                if cum > r && !targets.contains(&j) {
                    targets.insert(j);
                    graph.add_edge(new_node, j, 1.0);
                    degrees[new_node] += 1;
                    degrees[j] += 1;
                    added += 1;
                    break;
                }
            }
        }
    }
    graph
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Graph {
        // Triangle + pendant: 0-1-2-0, 2-3
        Graph::from_edges(4, &[
            (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 3, 1.0),
        ], false)
    }

    #[test]
    fn test_clustering() {
        let g = sample_graph();
        let cc = global_clustering_coefficient(&g);
        assert!(cc > 0.0); // has triangles
    }

    #[test]
    fn test_local_clustering() {
        let g = sample_graph();
        let lcc = local_clustering_coefficient(&g, 0);
        assert!((lcc - 1.0).abs() < 1e-10); // 0's neighbors (1,2) are connected
    }

    #[test]
    fn test_degree_distribution() {
        let g = sample_graph();
        let dd = degree_distribution(&g);
        assert!(!dd.is_empty());
    }

    #[test]
    fn test_average_degree() {
        let g = sample_graph();
        let avg = average_degree(&g);
        assert!(avg > 1.0);
    }

    #[test]
    fn test_modularity() {
        let g = sample_graph();
        let labels = vec![0, 0, 0, 1];
        let q = modularity(&g, &labels);
        assert!(q > -0.5); // should be reasonable
    }

    #[test]
    fn test_label_propagation() {
        let g = sample_graph();
        let labels = label_propagation(&g, 10, 42);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_common_neighbors() {
        let g = sample_graph();
        let cn = common_neighbors(&g, 0, 2);
        assert!(cn >= 1); // node 1 is common neighbor
    }

    #[test]
    fn test_erdos_renyi() {
        let g = erdos_renyi(20, 0.3, 42);
        assert_eq!(g.n_nodes, 20);
        assert!(g.n_edges() > 0);
    }

    #[test]
    fn test_barabasi_albert() {
        let g = barabasi_albert(20, 2, 42);
        assert_eq!(g.n_nodes, 20);
        assert!(g.n_edges() > 0);
    }

    #[test]
    fn test_network_efficiency() {
        let g = sample_graph();
        let eff = network_efficiency(&g);
        assert!(eff > 0.0);
    }
}
