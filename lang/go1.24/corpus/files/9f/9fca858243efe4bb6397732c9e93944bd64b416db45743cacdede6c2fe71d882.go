// SPDX-License-Identifier: MIT OR Apache-2.0
// Sorting with the sort and slices packages, with and without a comparison.
package main

import (
	"fmt"
	"slices"
	"sort"
	"strings"
)

type person struct {
	Name string
	Age  int
}

func main() {
	xs := []int{5, 2, 9, 1, 5, 6, -3, 0}
	sort.Ints(xs)
	fmt.Println(xs, sort.SearchInts(xs, 5))
	ws := []string{"pear", "Apple", "fig", "banana", "apple"}
	sort.Strings(ws)
	fmt.Println(ws)
	ps := []person{{"Ann", 30}, {"Bob", 25}, {"Cid", 30}, {"Dee", 25}, {"Eve", 35}}
	sort.SliceStable(ps, func(i, j int) bool { return ps[i].Age < ps[j].Age })
	fmt.Println(ps)
	ys := []int{3, 1, 2}
	slices.Sort(ys)
	fmt.Println(ys, slices.Contains(ys, 2), slices.Index(ys, 3), slices.Max(ys), slices.Min(ys))
	zs := []string{"bb", "a", "ccc"}
	slices.SortFunc(zs, func(a, b string) int { return len(b) - len(a) })
	fmt.Println(zs)
	vs := []string{"b", "A", "c", "B", "a"}
	slices.SortStableFunc(vs, func(a, b string) int { return strings.Compare(strings.ToLower(a), strings.ToLower(b)) })
	fmt.Println(vs)
	slices.Reverse(vs)
	fmt.Println(vs, len(vs))
}
