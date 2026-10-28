// SPDX-License-Identifier: MIT OR Apache-2.0
// Binary search over a sorted slice, for present and absent values.
package main

import "fmt"

func search(xs []int, target int) (int, bool) {
	lo, hi := 0, len(xs)
	for lo < hi {
		mid := lo + (hi-lo)/2
		if xs[mid] < target {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	return lo, lo < len(xs) && xs[lo] == target
}

func main() {
	xs := make([]int, 0, 50)
	for i := 0; i < 50; i++ {
		xs = append(xs, i*i-100)
	}
	for _, t := range []int{-100, -99, 0, 1, 44, 2301, 2302, 5000, -1000} {
		i, ok := search(xs, t)
		fmt.Println(t, i, ok)
	}
}
