// SPDX-License-Identifier: MIT OR Apache-2.0
// All permutations of four letters in lexicographic order by the
// next-permutation step.
package main

import "fmt"

func next(a []byte) bool {
	i := len(a) - 2
	for i >= 0 && a[i] >= a[i+1] {
		i--
	}
	if i < 0 {
		return false
	}
	j := len(a) - 1
	for a[j] <= a[i] {
		j--
	}
	a[i], a[j] = a[j], a[i]
	for l, r := i+1, len(a)-1; l < r; l, r = l+1, r-1 {
		a[l], a[r] = a[r], a[l]
	}
	return true
}

func main() {
	a := []byte("abcd")
	n := 0
	for {
		fmt.Print(string(a), " ")
		n++
		if n%6 == 0 {
			fmt.Println()
		}
		if !next(a) {
			break
		}
	}
	fmt.Println(n)
	b := []byte("aabb")
	m := 1
	for next(b) {
		m++
	}
	fmt.Println(m, string(b))
}
