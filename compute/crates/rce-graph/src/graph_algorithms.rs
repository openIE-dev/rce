//! Graph and network algorithms: BFS, DFS, Dijkstra, Bellman-Ford,
//! Floyd-Warshall, Prim, Kruskal, topological sort, connected components,
//! strongly connected components (Tarjan), PageRank, betweenness centrality.

use std::collections::{BinaryHeap, HashSet, VecDeque};
use std::cmp::Ordering;

// ═══════════════════════════════════════════════════════════════════
// GRAPH REPRESENTATION
// ═══════════════════════════════════════════════════════════════════

/// Weighted edge: (source, target, weight).
pub type Edge = (usize, usize, f64);

/// Adjacency list graph. Nodes are 0..n_nodes.
#[derive(Clone, Debug)]
pub struct Graph {
    pub n_nodes: usize,
    pub adj: Vec<Vec<(usize, f64)>>, // adj[u] = [(v, weight), ...]
    pub directed: bool,
}

impl Graph {
    /// Create empty graph with n nodes.
    pub fn new(n_nodes: usize, directed: bool) -> Self {
        Self { n_nodes, adj: vec![vec![]; n_nodes], directed }
    }

    /// Create from edge list.
    pub fn from_edges(n_nodes: usize, edges: &[Edge], directed: bool) -> Self {
        let mut g = Self::new(n_nodes, directed);
        for &(u, v, w) in edges {
            g.add_edge(u, v, w);
        }
        g
    }

    /// Add edge.
    pub fn add_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adj[u].push((v, weight));
        if !self.directed {
            self.adj[v].push((u, weight));
        }
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        let total: usize = self.adj.iter().map(|a| a.len()).sum();
        if self.directed { total } else { total / 2 }
    }

    /// Degree of node.
    pub fn degree(&self, node: usize) -> usize {
        self.adj[node].len()
    }
}

// ═══════════════════════════════════════════════════════════════════
// BFS & DFS
// ═══════════════════════════════════════════════════════════════════

/// Breadth-first search from source. Returns distances (f64::INFINITY if unreachable).
pub fn bfs(graph: &Graph, source: usize) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; graph.n_nodes];
    dist[source] = 0.0;
    let mut queue = VecDeque::new();
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        for &(v, _) in &graph.adj[u] {
            if dist[v].is_infinite() {
                dist[v] = dist[u] + 1.0;
                queue.push_back(v);
            }
        }
    }
    dist
}

/// BFS shortest path from source to target. Returns path or empty vec if no path.
pub fn bfs_path(graph: &Graph, source: usize, target: usize) -> Vec<usize> {
    let mut prev = vec![usize::MAX; graph.n_nodes];
    let mut visited = vec![false; graph.n_nodes];
    visited[source] = true;
    let mut queue = VecDeque::new();
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        if u == target { break; }
        for &(v, _) in &graph.adj[u] {
            if !visited[v] {
                visited[v] = true;
                prev[v] = u;
                queue.push_back(v);
            }
        }
    }
    if !visited[target] { return vec![]; }
    let mut path = vec![target];
    let mut cur = target;
    while cur != source {
        cur = prev[cur];
        path.push(cur);
    }
    path.reverse();
    path
}

/// Depth-first search from source. Returns visit order.
pub fn dfs(graph: &Graph, source: usize) -> Vec<usize> {
    let mut visited = vec![false; graph.n_nodes];
    let mut order = Vec::new();
    let mut stack = vec![source];
    while let Some(u) = stack.pop() {
        if visited[u] { continue; }
        visited[u] = true;
        order.push(u);
        for &(v, _) in graph.adj[u].iter().rev() {
            if !visited[v] {
                stack.push(v);
            }
        }
    }
    order
}

// ═══════════════════════════════════════════════════════════════════
// SHORTEST PATHS
// ═══════════════════════════════════════════════════════════════════

#[derive(PartialEq)]
struct DijkState { cost: f64, node: usize }
impl Eq for DijkState {}
impl PartialOrd for DijkState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for DijkState {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

/// Dijkstra's algorithm from source. Returns (distances, predecessors).
/// Predecessors: prev[v] = previous node on shortest path, usize::MAX if source or unreachable.
pub fn dijkstra(graph: &Graph, source: usize) -> (Vec<f64>, Vec<usize>) {
    let mut dist = vec![f64::INFINITY; graph.n_nodes];
    let mut prev = vec![usize::MAX; graph.n_nodes];
    dist[source] = 0.0;
    let mut heap = BinaryHeap::new();
    heap.push(DijkState { cost: 0.0, node: source });
    while let Some(DijkState { cost, node: u }) = heap.pop() {
        if cost > dist[u] { continue; }
        for &(v, w) in &graph.adj[u] {
            let new_dist = dist[u] + w;
            if new_dist < dist[v] {
                dist[v] = new_dist;
                prev[v] = u;
                heap.push(DijkState { cost: new_dist, node: v });
            }
        }
    }
    (dist, prev)
}

/// Reconstruct path from Dijkstra predecessors.
pub fn reconstruct_path(prev: &[usize], source: usize, target: usize) -> Vec<usize> {
    if prev[target] == usize::MAX && target != source { return vec![]; }
    let mut path = vec![target];
    let mut cur = target;
    while cur != source {
        cur = prev[cur];
        if cur == usize::MAX { return vec![]; }
        path.push(cur);
    }
    path.reverse();
    path
}

/// Bellman-Ford from source. Returns (distances, has_negative_cycle).
pub fn bellman_ford(graph: &Graph, source: usize) -> (Vec<f64>, bool) {
    let mut dist = vec![f64::INFINITY; graph.n_nodes];
    dist[source] = 0.0;
    for _ in 0..graph.n_nodes - 1 {
        for u in 0..graph.n_nodes {
            if dist[u].is_infinite() { continue; }
            for &(v, w) in &graph.adj[u] {
                if dist[u] + w < dist[v] {
                    dist[v] = dist[u] + w;
                }
            }
        }
    }
    // Check for negative cycles
    let mut neg_cycle = false;
    for u in 0..graph.n_nodes {
        if dist[u].is_infinite() { continue; }
        for &(v, w) in &graph.adj[u] {
            if dist[u] + w < dist[v] {
                neg_cycle = true;
                break;
            }
        }
        if neg_cycle { break; }
    }
    (dist, neg_cycle)
}

/// Floyd-Warshall all-pairs shortest paths. Returns n×n distance matrix.
pub fn floyd_warshall(graph: &Graph) -> Vec<Vec<f64>> {
    let n = graph.n_nodes;
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n { dist[i][i] = 0.0; }
    for u in 0..n {
        for &(v, w) in &graph.adj[u] {
            dist[u][v] = dist[u][v].min(w);
        }
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] + dist[k][j] < dist[i][j] {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    dist
}

// ═══════════════════════════════════════════════════════════════════
// MINIMUM SPANNING TREE
// ═══════════════════════════════════════════════════════════════════

/// Prim's MST. Returns edges of MST and total weight.
pub fn prim_mst(graph: &Graph) -> (Vec<(usize, usize, f64)>, f64) {
    let n = graph.n_nodes;
    if n == 0 { return (vec![], 0.0); }
    let mut in_mst = vec![false; n];
    let mut key = vec![f64::INFINITY; n];
    let mut parent = vec![usize::MAX; n];
    let mut mst_edges = Vec::new();
    let mut total = 0.0;
    key[0] = 0.0;
    // Simple O(V²) Prim — correct and clear
    for _ in 0..n {
        // Find minimum key vertex not in MST
        let mut u = usize::MAX;
        let mut min_key = f64::INFINITY;
        for v in 0..n {
            if !in_mst[v] && key[v] < min_key {
                min_key = key[v];
                u = v;
            }
        }
        if u == usize::MAX { break; }
        in_mst[u] = true;
        if parent[u] != usize::MAX {
            mst_edges.push((parent[u], u, key[u]));
            total += key[u];
        }
        for &(v, w) in &graph.adj[u] {
            if !in_mst[v] && w < key[v] {
                key[v] = w;
                parent[v] = u;
            }
        }
    }
    (mst_edges, total)
}

/// Union-Find for Kruskal's.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}
impl UnionFind {
    fn new(n: usize) -> Self { Self { parent: (0..n).collect(), rank: vec![0; n] } }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x { self.parent[x] = self.find(self.parent[x]); }
        self.parent[x]
    }
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb { return false; }
        match self.rank[ra].cmp(&self.rank[rb]) {
            Ordering::Less => self.parent[ra] = rb,
            Ordering::Greater => self.parent[rb] = ra,
            Ordering::Equal => { self.parent[rb] = ra; self.rank[ra] += 1; }
        }
        true
    }
}

/// Kruskal's MST. Returns edges and total weight.
pub fn kruskal_mst(n_nodes: usize, edges: &mut [Edge]) -> (Vec<Edge>, f64) {
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal));
    let mut uf = UnionFind::new(n_nodes);
    let mut mst = Vec::new();
    let mut total = 0.0;
    for &(u, v, w) in edges.iter() {
        if uf.union(u, v) {
            mst.push((u, v, w));
            total += w;
            if mst.len() == n_nodes - 1 { break; }
        }
    }
    (mst, total)
}

// ═══════════════════════════════════════════════════════════════════
// TOPOLOGICAL SORT
// ═══════════════════════════════════════════════════════════════════

/// Topological sort (Kahn's algorithm). Returns sorted order or None if cycle exists.
pub fn topological_sort(graph: &Graph) -> Option<Vec<usize>> {
    let n = graph.n_nodes;
    let mut in_degree = vec![0usize; n];
    for u in 0..n {
        for &(v, _) in &graph.adj[u] {
            in_degree[v] += 1;
        }
    }
    let mut queue: VecDeque<usize> = in_degree.iter().enumerate()
        .filter(|(_, d)| **d == 0).map(|(i, _)| i).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &(v, _) in &graph.adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 { queue.push_back(v); }
        }
    }
    if order.len() == n { Some(order) } else { None }
}

// ═══════════════════════════════════════════════════════════════════
// CONNECTED COMPONENTS
// ═══════════════════════════════════════════════════════════════════

/// Connected components (undirected). Returns component label per node.
pub fn connected_components(graph: &Graph) -> Vec<usize> {
    let n = graph.n_nodes;
    let mut label = vec![usize::MAX; n];
    let mut comp = 0;
    for start in 0..n {
        if label[start] != usize::MAX { continue; }
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            if label[u] != usize::MAX { continue; }
            label[u] = comp;
            for &(v, _) in &graph.adj[u] {
                if label[v] == usize::MAX { stack.push(v); }
            }
        }
        comp += 1;
    }
    label
}

/// Number of connected components.
pub fn n_connected_components(graph: &Graph) -> usize {
    let labels = connected_components(graph);
    labels.iter().copied().collect::<HashSet<usize>>().len()
}

/// Strongly connected components (Tarjan's algorithm, directed graph).
/// Returns list of components, each a Vec of node indices.
pub fn tarjan_scc(graph: &Graph) -> Vec<Vec<usize>> {
    let n = graph.n_nodes;
    let mut index_counter = 0usize;
    let mut stack = Vec::new();
    let mut on_stack = vec![false; n];
    let mut index = vec![usize::MAX; n];
    let mut lowlink = vec![0usize; n];
    let mut sccs = Vec::new();

    fn strongconnect(
        u: usize, graph: &Graph, index_counter: &mut usize,
        stack: &mut Vec<usize>, on_stack: &mut Vec<bool>,
        index: &mut Vec<usize>, lowlink: &mut Vec<usize>,
        sccs: &mut Vec<Vec<usize>>,
    ) {
        index[u] = *index_counter;
        lowlink[u] = *index_counter;
        *index_counter += 1;
        stack.push(u);
        on_stack[u] = true;

        for &(v, _) in &graph.adj[u] {
            if index[v] == usize::MAX {
                strongconnect(v, graph, index_counter, stack, on_stack, index, lowlink, sccs);
                lowlink[u] = lowlink[u].min(lowlink[v]);
            } else if on_stack[v] {
                lowlink[u] = lowlink[u].min(index[v]);
            }
        }

        if lowlink[u] == index[u] {
            let mut comp = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack[w] = false;
                comp.push(w);
                if w == u { break; }
            }
            sccs.push(comp);
        }
    }

    for i in 0..n {
        if index[i] == usize::MAX {
            strongconnect(i, graph, &mut index_counter, &mut stack, &mut on_stack,
                         &mut index, &mut lowlink, &mut sccs);
        }
    }
    sccs
}

// ═══════════════════════════════════════════════════════════════════
// PAGERANK
// ═══════════════════════════════════════════════════════════════════

/// PageRank. Returns scores for each node.
pub fn pagerank(graph: &Graph, damping: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = graph.n_nodes;
    if n == 0 { return vec![]; }
    let mut rank = vec![1.0 / n as f64; n];
    let out_degree: Vec<f64> = (0..n).map(|u| graph.adj[u].len() as f64).collect();

    for _ in 0..max_iter {
        let mut new_rank = vec![(1.0 - damping) / n as f64; n];
        for u in 0..n {
            if out_degree[u] > 0.0 {
                let share = damping * rank[u] / out_degree[u];
                for &(v, _) in &graph.adj[u] {
                    new_rank[v] += share;
                }
            } else {
                // Dangling node: distribute equally
                let share = damping * rank[u] / n as f64;
                for r in new_rank.iter_mut() { *r += share; }
            }
        }
        let diff: f64 = rank.iter().zip(&new_rank).map(|(a, b)| (a - b).abs()).sum();
        rank = new_rank;
        if diff < tol { break; }
    }
    rank
}

// ═══════════════════════════════════════════════════════════════════
// CENTRALITY
// ═══════════════════════════════════════════════════════════════════

/// Betweenness centrality (Brandes algorithm).
pub fn betweenness_centrality(graph: &Graph) -> Vec<f64> {
    let n = graph.n_nodes;
    let mut cb = vec![0.0; n];

    for s in 0..n {
        let mut stack = Vec::new();
        let mut pred: Vec<Vec<usize>> = vec![vec![]; n];
        let mut sigma = vec![0.0f64; n];
        sigma[s] = 1.0;
        let mut dist = vec![-1i64; n];
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            for &(w, _) in &graph.adj[v] {
                if dist[w] < 0 {
                    dist[w] = dist[v] + 1;
                    queue.push_back(w);
                }
                if dist[w] == dist[v] + 1 {
                    sigma[w] += sigma[v];
                    pred[w].push(v);
                }
            }
        }

        let mut delta = vec![0.0; n];
        while let Some(w) = stack.pop() {
            for &v in &pred[w] {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
            if w != s { cb[w] += delta[w]; }
        }
    }

    if !graph.directed {
        for v in cb.iter_mut() { *v /= 2.0; }
    }
    cb
}

/// Closeness centrality.
pub fn closeness_centrality(graph: &Graph) -> Vec<f64> {
    let n = graph.n_nodes;
    (0..n).map(|s| {
        let dists = bfs(graph, s);
        let reachable: Vec<f64> = dists.iter().filter(|d| d.is_finite() && **d > 0.0).copied().collect();
        if reachable.is_empty() { return 0.0; }
        let total: f64 = reachable.iter().sum();
        reachable.len() as f64 / total
    }).collect()
}

/// Degree centrality.
pub fn degree_centrality(graph: &Graph) -> Vec<f64> {
    let n = graph.n_nodes;
    if n <= 1 { return vec![0.0; n]; }
    let denom = (n - 1) as f64;
    (0..n).map(|u| graph.degree(u) as f64 / denom).collect()
}

// ═══════════════════════════════════════════════════════════════════
// CYCLE DETECTION
// ═══════════════════════════════════════════════════════════════════

/// Detect if directed graph has a cycle.
pub fn has_cycle_directed(graph: &Graph) -> bool {
    topological_sort(graph).is_none()
}

/// Detect if undirected graph has a cycle.
pub fn has_cycle_undirected(graph: &Graph) -> bool {
    let n = graph.n_nodes;
    let mut uf = UnionFind::new(n);
    let mut seen = HashSet::new();
    for u in 0..n {
        for &(v, _) in &graph.adj[u] {
            if u < v || graph.directed {
                let key = (u.min(v), u.max(v));
                if !seen.insert(key) { continue; }
                if !uf.union(u, v) { return true; }
            }
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════
// BIPARTITENESS
// ═══════════════════════════════════════════════════════════════════

/// Check if graph is bipartite. Returns (is_bipartite, coloring).
pub fn is_bipartite(graph: &Graph) -> (bool, Vec<i8>) {
    let n = graph.n_nodes;
    let mut color = vec![-1i8; n];
    for start in 0..n {
        if color[start] >= 0 { continue; }
        color[start] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for &(v, _) in &graph.adj[u] {
                if color[v] < 0 {
                    color[v] = 1 - color[u];
                    queue.push_back(v);
                } else if color[v] == color[u] {
                    return (false, color);
                }
            }
        }
    }
    (true, color)
}

// ═══════════════════════════════════════════════════════════════════
// GRAPH DENSITY & DIAMETER
// ═══════════════════════════════════════════════════════════════════

/// Graph density.
pub fn graph_density(graph: &Graph) -> f64 {
    let n = graph.n_nodes as f64;
    if n <= 1.0 { return 0.0; }
    let e = graph.n_edges() as f64;
    if graph.directed { e / (n * (n - 1.0)) } else { 2.0 * e / (n * (n - 1.0)) }
}

/// Graph diameter (longest shortest path). Returns f64::INFINITY if disconnected.
pub fn graph_diameter(graph: &Graph) -> f64 {
    let apsp = floyd_warshall(graph);
    let mut diam = 0.0f64;
    for row in &apsp {
        for &d in row {
            if d.is_infinite() { return f64::INFINITY; }
            diam = diam.max(d);
        }
    }
    diam
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Graph {
        // 0 -- 1 -- 2
        // |         |
        // 3 ------- 4
        Graph::from_edges(5, &[
            (0, 1, 1.0), (1, 2, 1.0), (0, 3, 1.0), (2, 4, 1.0), (3, 4, 1.0),
        ], false)
    }

    #[test]
    fn test_bfs() {
        let g = sample_graph();
        let dist = bfs(&g, 0);
        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[1], 1.0);
        assert_eq!(dist[2], 2.0);
        assert_eq!(dist[4], 2.0);
    }

    #[test]
    fn test_bfs_path() {
        let g = sample_graph();
        let path = bfs_path(&g, 0, 4);
        assert_eq!(path.len(), 3); // 0→3→4
    }

    #[test]
    fn test_dfs() {
        let g = sample_graph();
        let order = dfs(&g, 0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 5);
    }

    #[test]
    fn test_dijkstra() {
        let g = Graph::from_edges(4, &[
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0), (2, 3, 1.0),
        ], true);
        let (dist, prev) = dijkstra(&g, 0);
        assert_eq!(dist[2], 3.0); // 0→1→2
        assert_eq!(dist[3], 4.0);
        let path = reconstruct_path(&prev, 0, 3);
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_bellman_ford() {
        let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, -3.0), (0, 2, 4.0)], true);
        let (dist, neg) = bellman_ford(&g, 0);
        assert!(!neg);
        assert_eq!(dist[2], -2.0);
    }

    #[test]
    fn test_floyd_warshall() {
        let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 2.0), (0, 2, 5.0)], true);
        let dist = floyd_warshall(&g);
        assert_eq!(dist[0][2], 3.0);
    }

    #[test]
    fn test_prim_mst() {
        let g = Graph::from_edges(4, &[
            (0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0), (2, 3, 1.0),
        ], false);
        let (edges, total) = prim_mst(&g);
        assert_eq!(edges.len(), 3);
        // MST: 0-1(1) + 2-3(1) + 1-2(2) = 4
        assert!((total - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_kruskal_mst() {
        let mut edges = vec![(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0), (2, 3, 1.0)];
        let (mst, total) = kruskal_mst(4, &mut edges);
        assert_eq!(mst.len(), 3);
        assert!((total - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_topological_sort() {
        let g = Graph::from_edges(4, &[(0, 1, 1.0), (0, 2, 1.0), (1, 3, 1.0), (2, 3, 1.0)], true);
        let order = topological_sort(&g).unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(*order.last().unwrap(), 3);
    }

    #[test]
    fn test_connected_components() {
        // Two components: {0,1,2} and {3,4}
        let mut g = Graph::new(5, false);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(3, 4, 1.0);
        assert_eq!(n_connected_components(&g), 2);
    }

    #[test]
    fn test_tarjan_scc() {
        // Cycle: 0→1→2→0, plus 3→4
        let g = Graph::from_edges(5, &[
            (0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 4, 1.0),
        ], true);
        let sccs = tarjan_scc(&g);
        let big: Vec<&Vec<usize>> = sccs.iter().filter(|c| c.len() == 3).collect();
        assert_eq!(big.len(), 1);
    }

    #[test]
    fn test_pagerank() {
        let g = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], true);
        let pr = pagerank(&g, 0.85, 100, 1e-8);
        // All nodes in a cycle should have equal rank
        assert!((pr[0] - pr[1]).abs() < 1e-4);
        assert!((pr[1] - pr[2]).abs() < 1e-4);
    }

    #[test]
    fn test_is_bipartite() {
        // Square: 0-1-2-3-0 → bipartite
        let g = Graph::from_edges(4, &[
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0),
        ], false);
        assert!(is_bipartite(&g).0);
        // Triangle: not bipartite
        let g2 = Graph::from_edges(3, &[(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)], false);
        assert!(!is_bipartite(&g2).0);
    }

    #[test]
    fn test_graph_density() {
        // Complete graph K4: 6 edges, density = 1.0
        let g = Graph::from_edges(4, &[
            (0,1,1.0),(0,2,1.0),(0,3,1.0),(1,2,1.0),(1,3,1.0),(2,3,1.0),
        ], false);
        assert!((graph_density(&g) - 1.0).abs() < 1e-10);
    }
}
