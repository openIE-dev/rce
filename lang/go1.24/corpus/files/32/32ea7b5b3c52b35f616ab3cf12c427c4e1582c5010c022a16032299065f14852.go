// SPDX-License-Identifier: MIT OR Apache-2.0
// A stable merge sort of records by key; equal keys keep their order.
package main

import "fmt"

type rec struct {
	key int
	tag string
}

func mergesort(a []rec) []rec {
	if len(a) <= 1 {
		return a
	}
	m := len(a) / 2
	l, r := mergesort(a[:m]), mergesort(a[m:])
	out := make([]rec, 0, len(a))
	i, j := 0, 0
	for i < len(l) && j < len(r) {
		if r[j].key < l[i].key {
			out = append(out, r[j])
			j++
		} else {
			out = append(out, l[i])
			i++
		}
	}
	out = append(out, l[i:]...)
	return append(out, r[j:]...)
}

func main() {
	in := []rec{{3, "a"}, {1, "b"}, {2, "c"}, {3, "d"}, {1, "e"}, {2, "f"}, {0, "g"}, {3, "h"}}
	fmt.Println(mergesort(in))
	fmt.Println(in)
}
