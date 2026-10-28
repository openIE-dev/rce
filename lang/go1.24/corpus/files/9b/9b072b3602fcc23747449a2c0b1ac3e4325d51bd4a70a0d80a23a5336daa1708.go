// SPDX-License-Identifier: MIT OR Apache-2.0
// The number of ways to place n non-attacking queens, for n from 1 to 9,
// and the first solution for n = 8.
package main

import "fmt"

func solve(n, row int, cols, d1, d2 []bool, pos []int, first *[]int) int {
	if row == n {
		if *first == nil {
			*first = append([]int(nil), pos...)
		}
		return 1
	}
	count := 0
	for c := 0; c < n; c++ {
		if cols[c] || d1[row+c] || d2[row-c+n-1] {
			continue
		}
		cols[c], d1[row+c], d2[row-c+n-1] = true, true, true
		pos[row] = c
		count += solve(n, row+1, cols, d1, d2, pos, first)
		cols[c], d1[row+c], d2[row-c+n-1] = false, false, false
	}
	return count
}

func main() {
	for n := 1; n <= 9; n++ {
		var first []int
		k := solve(n, 0, make([]bool, n), make([]bool, 2*n), make([]bool, 2*n), make([]int, n), &first)
		fmt.Println(n, k)
		if n == 8 {
			fmt.Println(first)
		}
	}
}
