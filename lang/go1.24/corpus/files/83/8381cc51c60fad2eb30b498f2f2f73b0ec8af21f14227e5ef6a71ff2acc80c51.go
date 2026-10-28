// SPDX-License-Identifier: MIT OR Apache-2.0
// Stable sorts keep equal elements in input order; a sort whose order is
// total gives one result whatever the algorithm.
package main

import (
	"fmt"
	"sort"
)

type item struct {
	k int
	v string
}

func main() {
	items := []item{{2, "a"}, {1, "b"}, {2, "c"}, {1, "d"}, {0, "e"}, {2, "f"}, {1, "g"}}
	stable := append([]item(nil), items...)
	sort.SliceStable(stable, func(i, j int) bool { return stable[i].k < stable[j].k })
	fmt.Println(stable)
	total := append([]item(nil), items...)
	sort.Slice(total, func(i, j int) bool {
		if total[i].k != total[j].k {
			return total[i].k > total[j].k
		}
		return total[i].v < total[j].v
	})
	fmt.Println(total)
	fs := []float64{3, -1, 0, 1e-300, -1e-300, 2}
	sort.Float64s(fs)
	fmt.Println(fs)
	words := []string{"b", "B", "a", "A", "_", "1"}
	sort.Strings(words)
	fmt.Println(words)
}
