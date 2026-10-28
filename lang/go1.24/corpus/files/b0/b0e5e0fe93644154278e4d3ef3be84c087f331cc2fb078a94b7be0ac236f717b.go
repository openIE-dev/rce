// SPDX-License-Identifier: MIT OR Apache-2.0
// The order of a range over a map is not specified, so the program sorts
// the keys before it prints.
package main

import (
	"fmt"
	"maps"
	"slices"
	"sort"
)

func main() {
	m := map[string]int{}
	words := []string{"delta", "alpha", "charlie", "bravo", "echo", "alpha", "delta", "alpha"}
	for _, w := range words {
		m[w]++
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Println(k, m[k])
	}
	fmt.Println(slices.Sorted(maps.Keys(m)))
	total := 0
	for _, v := range m {
		total += v
	}
	fmt.Println(total)
}
