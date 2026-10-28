// SPDX-License-Identifier: MIT OR Apache-2.0
// Dijkstra's shortest paths on a small weighted graph, choosing the nearest
// unvisited vertex by a linear scan (ties go to the lowest index).
package main

import "fmt"

func main() {
	const inf = 1 << 60
	n := 7
	edges := [][3]int{{0, 1, 7}, {0, 2, 9}, {0, 5, 14}, {1, 2, 10}, {1, 3, 15}, {2, 3, 11}, {2, 5, 2}, {3, 4, 6}, {4, 5, 9}, {5, 6, 1}, {6, 4, 3}}
	adj := make([][][2]int, n)
	for _, e := range edges {
		adj[e[0]] = append(adj[e[0]], [2]int{e[1], e[2]})
		adj[e[1]] = append(adj[e[1]], [2]int{e[0], e[2]})
	}
	dist := make([]int, n)
	prev := make([]int, n)
	done := make([]bool, n)
	for i := range dist {
		dist[i], prev[i] = inf, -1
	}
	dist[0] = 0
	for {
		u := -1
		for i := 0; i < n; i++ {
			if !done[i] && dist[i] < inf && (u < 0 || dist[i] < dist[u]) {
				u = i
			}
		}
		if u < 0 {
			break
		}
		done[u] = true
		for _, e := range adj[u] {
			if dist[u]+e[1] < dist[e[0]] {
				dist[e[0]] = dist[u] + e[1]
				prev[e[0]] = u
			}
		}
	}
	for v := 0; v < n; v++ {
		path := []int{}
		for x := v; x >= 0; x = prev[x] {
			path = append([]int{x}, path...)
		}
		fmt.Println(v, dist[v], path)
	}
}
