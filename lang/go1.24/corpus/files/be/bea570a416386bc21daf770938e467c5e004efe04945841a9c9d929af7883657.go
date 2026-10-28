// SPDX-License-Identifier: MIT OR Apache-2.0
// Slices: append, len, indexing, slicing, copy and iteration.
package main

import "fmt"

func main() {
	var s []int
	for i := 1; i <= 5; i++ {
		s = append(s, i*i)
	}
	fmt.Println(s, len(s), s[0], s[len(s)-1])
	fmt.Println(s[1:3], s[:2], s[3:], len(s[1:3]))
	t := make([]int, 3)
	n := copy(t, s)
	fmt.Println(t, n)
	s = append(s, t...)
	fmt.Println(s)
	sum := 0
	for _, v := range s {
		sum += v
	}
	fmt.Println(sum)
	for i := range 3 {
		fmt.Print(i, " ")
	}
	fmt.Println()
	m := [][]string{{"a"}, {"b", "c"}}
	m[1] = append(m[1], "d")
	fmt.Println(m, len(m[1]))
	r := s[:0]
	for _, v := range s {
		if v%2 == 0 {
			r = append(r, v)
		}
	}
	fmt.Println(r)
}
