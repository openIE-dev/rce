// SPDX-License-Identifier: MIT OR Apache-2.0
// The 0/1 knapsack problem by dynamic programming over capacities.
package main

import "fmt"

func main() {
	weights := []int{12, 7, 11, 8, 9, 5, 3, 14}
	values := []int{24, 13, 23, 15, 16, 9, 5, 30}
	for _, capacity := range []int{0, 10, 26, 40, 69} {
		best := make([]int, capacity+1)
		for i := range weights {
			for c := capacity; c >= weights[i]; c-- {
				if v := best[c-weights[i]] + values[i]; v > best[c] {
					best[c] = v
				}
			}
		}
		fmt.Println(capacity, best[capacity])
	}
}
