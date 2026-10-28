# SPDX-License-Identifier: MIT OR Apache-2.0
# Kahn's topological sort, always taking the smallest ready vertex, and
# detection of a cycle.
import heapq


def topo(n, edges):
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    ready = [v for v in range(n) if indeg[v] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        v = heapq.heappop(ready)
        order.append(v)
        for w in adj[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)
    return order, len(order) == n


print(topo(8, [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1), (7, 6), (6, 3)]))
print(topo(4, [(0, 1), (1, 2), (2, 0), (2, 3)]))
print(topo(3, []))
