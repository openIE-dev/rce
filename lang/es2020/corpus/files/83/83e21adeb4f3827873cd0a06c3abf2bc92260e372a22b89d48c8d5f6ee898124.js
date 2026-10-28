// SPDX-License-Identifier: MIT OR Apache-2.0
// Dijkstra's shortest paths on a small weighted graph, choosing the nearest
// unvisited vertex by a linear scan (ties go to the lowest index).
const n = 7;
const edges = [[0, 1, 7], [0, 2, 9], [0, 5, 14], [1, 2, 10], [1, 3, 15], [2, 3, 11], [2, 5, 2], [3, 4, 6], [4, 5, 9], [5, 6, 1], [6, 4, 3]];
const adj = Array.from({ length: n }, () => []);
for (const [a, b, w] of edges) {
  adj[a].push([b, w]);
  adj[b].push([a, w]);
}
const dist = new Array(n).fill(Infinity), prev = new Array(n).fill(-1), done = new Array(n).fill(false);
dist[0] = 0;
for (;;) {
  let u = -1;
  for (let i = 0; i < n; i++) if (!done[i] && dist[i] < Infinity && (u < 0 || dist[i] < dist[u])) u = i;
  if (u < 0) break;
  done[u] = true;
  for (const [v, w] of adj[u]) {
    if (dist[u] + w < dist[v]) {
      dist[v] = dist[u] + w;
      prev[v] = u;
    }
  }
}
for (let v = 0; v < n; v++) {
  const path = [];
  for (let x = v; x >= 0; x = prev[x]) path.unshift(x);
  console.log(v, dist[v], path.join("-"));
}
