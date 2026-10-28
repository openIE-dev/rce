// SPDX-License-Identifier: MIT OR Apache-2.0
// Breadth-first search for the shortest path through a grid maze.
const maze = [
  "S.#.......",
  ".##.####..",
  "....#..#..",
  "#.#.#.##.#",
  "..#...#...",
  ".####.#.#.",
  "......#.#E",
];
const h = maze.length, w = maze[0].length;
const dist = maze.map((row) => Array.from(row, () => -1));
dist[0][0] = 0;
const queue = [[0, 0]];
for (let head = 0; head < queue.length; head++) {
  const [r, c] = queue[head];
  for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    const nr = r + dr, nc = c + dc;
    if (nr >= 0 && nr < h && nc >= 0 && nc < w && maze[nr][nc] !== "#" && dist[nr][nc] < 0) {
      dist[nr][nc] = dist[r][c] + 1;
      queue.push([nr, nc]);
    }
  }
}
console.log(dist[h - 1][w - 1]);
for (const row of dist) console.log(row.map((d) => String(d).padStart(3)).join(""));
