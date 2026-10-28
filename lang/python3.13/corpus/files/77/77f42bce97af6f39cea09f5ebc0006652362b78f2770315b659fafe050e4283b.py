# SPDX-License-Identifier: MIT OR Apache-2.0
# Dijkstra's shortest paths with heapq; ties in the queue are broken by
# vertex number.
import heapq

edges = [(0, 1, 7), (0, 2, 9), (0, 5, 14), (1, 2, 10), (1, 3, 15), (2, 3, 11), (2, 5, 2), (3, 4, 6), (4, 5, 9), (5, 6, 1), (6, 4, 3)]
n = 7
adj = [[] for _ in range(n)]
for a, b, w in edges:
    adj[a].append((b, w))
    adj[b].append((a, w))
dist = [None] * n
prev = [None] * n
dist[0] = 0
pq = [(0, 0)]
while pq:
    d, u = heapq.heappop(pq)
    if d > dist[u]:
        continue
    for v, w in adj[u]:
        if dist[v] is None or d + w < dist[v]:
            dist[v] = d + w
            prev[v] = u
            heapq.heappush(pq, (d + w, v))
for v in range(n):
    path = []
    x = v
    while x is not None:
        path.insert(0, x)
        x = prev[x]
    print(v, dist[v], path)
