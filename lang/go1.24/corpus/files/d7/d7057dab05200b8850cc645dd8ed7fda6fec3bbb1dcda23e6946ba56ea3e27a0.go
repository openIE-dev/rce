// SPDX-License-Identifier: MIT OR Apache-2.0
// Slices share their backing array: appends within capacity are seen
// through other slices, and a full slice expression limits capacity.
package main

import "fmt"

func main() {
	base := make([]int, 3, 10)
	a := base[:2]
	b := append(a, 100)
	fmt.Println(base, a, b, len(b), cap(b))
	c := base[:2:2]
	d := append(c, 200)
	d[0] = -1
	fmt.Println(base, c, d, cap(c))
	arr := [5]int{1, 2, 3, 4, 5}
	s := arr[1:4]
	s[0] = 20
	fmt.Println(arr, s, len(s), cap(s))
	t := s[1:cap(s)]
	fmt.Println(t)
	m := []int{1, 2, 3}
	n := m
	n[0] = 9
	fmt.Println(m, n)
	o := make([]int, len(m))
	copy(o, m)
	o[1] = 8
	fmt.Println(m, o)
	overlap := []int{1, 2, 3, 4, 5}
	copy(overlap[1:], overlap)
	fmt.Println(overlap)
}
