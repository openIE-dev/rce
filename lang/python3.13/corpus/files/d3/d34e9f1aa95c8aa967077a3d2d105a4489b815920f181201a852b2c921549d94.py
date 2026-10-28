# SPDX-License-Identifier: MIT OR Apache-2.0
# Breadth-first search for the shortest path through a grid maze.
from collections import deque

maze = [
    "S.#.......",
    ".##.####..",
    "....#..#..",
    "#.#.#.##.#",
    "..#...#...",
    ".####.#.#.",
    "......#.#E",
]
h, w = len(maze), len(maze[0])
dist = [[-1] * w for _ in range(h)]
dist[0][0] = 0
queue = deque([(0, 0)])
while queue:
    r, c = queue.popleft()
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and maze[nr][nc] != "#" and dist[nr][nc] < 0:
            dist[nr][nc] = dist[r][c] + 1
            queue.append((nr, nc))
print(dist[h - 1][w - 1])
for row in dist:
    print("".join(f"{d:3d}" for d in row))
