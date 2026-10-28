// SPDX-License-Identifier: MIT OR Apache-2.0
// The longest Collatz chain that starts below 100000, with a memo table.
package main

import "fmt"

func main() {
	const limit = 100000
	memo := make([]int, limit)
	memo[1] = 1
	best, bestN := 0, 0
	for start := 2; start < limit; start++ {
		n, steps := start, 0
		for n >= limit || memo[n] == 0 {
			if n%2 == 0 {
				n /= 2
			} else {
				n = 3*n + 1
			}
			steps++
		}
		memo[start] = steps + memo[n]
		if memo[start] > best {
			best, bestN = memo[start], start
		}
	}
	fmt.Println(bestN, best)
	fmt.Println(memo[27], memo[97], memo[871])
}
