// SPDX-License-Identifier: MIT OR Apache-2.0
// Kahn's topological sort, always taking the smallest ready vertex, and
// detection of a cycle.
package main

import (
	"fmt"
	"sort"
)

func topo(n int, edges [][2]int) ([]int, bool) {
	indeg := make([]int, n)
	adj := make([][]int, n)
	for _, e := range edges {
		adj[e[0]] = append(adj[e[0]], e[1])
		indeg[e[1]]++
	}
	var ready, order []int
	for v := 0; v < n; v++ {
		if indeg[v] == 0 {
			ready = append(ready, v)
		}
	}
	for len(ready) > 0 {
		sort.Ints(ready)
		v := ready[0]
		ready = ready[1:]
		order = append(order, v)
		for _, w := range adj[v] {
			indeg[w]--
			if indeg[w] == 0 {
				ready = append(ready, w)
			}
		}
	}
	return order, len(order) == n
}

func main() {
	fmt.Println(topo(8, [][2]int{{5, 2}, {5, 0}, {4, 0}, {4, 1}, {2, 3}, {3, 1}, {7, 6}, {6, 3}}))
	fmt.Println(topo(4, [][2]int{{0, 1}, {1, 2}, {2, 0}, {2, 3}}))
	fmt.Println(topo(3, nil))
}
