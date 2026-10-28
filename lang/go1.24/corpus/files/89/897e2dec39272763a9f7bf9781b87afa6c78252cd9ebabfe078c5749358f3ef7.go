// SPDX-License-Identifier: MIT OR Apache-2.0
// Knuth-Morris-Pratt substring search, checked against strings.Index.
package main

import (
	"fmt"
	"strings"
)

func failure(p string) []int {
	f := make([]int, len(p))
	k := 0
	for i := 1; i < len(p); i++ {
		for k > 0 && p[i] != p[k] {
			k = f[k-1]
		}
		if p[i] == p[k] {
			k++
		}
		f[i] = k
	}
	return f
}

func findAll(text, p string) []int {
	if p == "" {
		return nil
	}
	f := failure(p)
	var out []int
	k := 0
	for i := 0; i < len(text); i++ {
		for k > 0 && text[i] != p[k] {
			k = f[k-1]
		}
		if text[i] == p[k] {
			k++
		}
		if k == len(p) {
			out = append(out, i-len(p)+1)
			k = f[k-1]
		}
	}
	return out
}

func main() {
	fmt.Println(failure("ababaca"), failure("aaaa"))
	text := "abababcabababcababab"
	for _, p := range []string{"abab", "ababc", "c", "x", "abababcabababcababab"} {
		all := findAll(text, p)
		first := -1
		if len(all) > 0 {
			first = all[0]
		}
		fmt.Println(p, all, first == strings.Index(text, p))
	}
}
