// SPDX-License-Identifier: MIT OR Apache-2.0
// Levenshtein distance between strings, counted over runes.
package main

import "fmt"

func lev(a, b string) int {
	ra, rb := []rune(a), []rune(b)
	prev := make([]int, len(rb)+1)
	for j := range prev {
		prev[j] = j
	}
	for i := 1; i <= len(ra); i++ {
		cur := make([]int, len(rb)+1)
		cur[0] = i
		for j := 1; j <= len(rb); j++ {
			cost := 1
			if ra[i-1] == rb[j-1] {
				cost = 0
			}
			cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
		}
		prev = cur
	}
	return prev[len(rb)]
}

func main() {
	pairs := [][2]string{{"kitten", "sitting"}, {"flaw", "lawn"}, {"", "abc"}, {"abc", ""}, {"same", "same"}, {"héllo", "hello"}, {"日本語", "日本"}, {"intention", "execution"}}
	for _, p := range pairs {
		fmt.Println(p[0], p[1], lev(p[0], p[1]))
	}
}
