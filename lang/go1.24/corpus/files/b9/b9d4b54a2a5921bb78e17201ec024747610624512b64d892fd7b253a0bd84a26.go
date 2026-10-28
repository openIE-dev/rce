// SPDX-License-Identifier: MIT OR Apache-2.0
// Breadth-first search for the shortest path through a grid maze.
package main

import "fmt"

func main() {
	maze := []string{
		"S.#.......",
		".##.####..",
		"....#..#..",
		"#.#.#.##.#",
		"..#...#...",
		".####.#.#.",
		"......#.#E",
	}
	h, w := len(maze), len(maze[0])
	dist := make([][]int, h)
	for i := range dist {
		dist[i] = make([]int, w)
		for j := range dist[i] {
			dist[i][j] = -1
		}
	}
	type cell struct{ r, c int }
	queue := []cell{{0, 0}}
	dist[0][0] = 0
	dirs := []cell{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for _, d := range dirs {
			n := cell{cur.r + d.r, cur.c + d.c}
			if n.r < 0 || n.r >= h || n.c < 0 || n.c >= w || maze[n.r][n.c] == '#' || dist[n.r][n.c] >= 0 {
				continue
			}
			dist[n.r][n.c] = dist[cur.r][cur.c] + 1
			queue = append(queue, n)
		}
	}
	fmt.Println(dist[h-1][w-1])
	for _, row := range dist {
		for _, d := range row {
			fmt.Printf("%3d", d)
		}
		fmt.Println()
	}
}
