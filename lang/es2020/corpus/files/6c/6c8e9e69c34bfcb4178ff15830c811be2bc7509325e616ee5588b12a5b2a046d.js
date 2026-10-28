// SPDX-License-Identifier: MIT OR Apache-2.0
// Kahn's topological sort, always taking the smallest ready vertex, and
// detection of a cycle.
function topo(n, edges) {
  const indeg = new Array(n).fill(0);
  const adj = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    adj[a].push(b);
    indeg[b]++;
  }
  const ready = [];
  for (let v = 0; v < n; v++) if (indeg[v] === 0) ready.push(v);
  const order = [];
  while (ready.length) {
    ready.sort((a, b) => a - b);
    const v = ready.shift();
    order.push(v);
    for (const w of adj[v]) if (--indeg[w] === 0) ready.push(w);
  }
  return [order, order.length === n];
}
console.log(JSON.stringify(topo(8, [[5, 2], [5, 0], [4, 0], [4, 1], [2, 3], [3, 1], [7, 6], [6, 3]])));
console.log(JSON.stringify(topo(4, [[0, 1], [1, 2], [2, 0], [2, 3]])));
console.log(JSON.stringify(topo(3, [])));
