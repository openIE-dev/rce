// SPDX-License-Identifier: MIT OR Apache-2.0
// Quicksort with Lomuto partitioning on a sequence from a fixed linear
// congruential generator.
package main

import "fmt"

func quicksort(a []int) {
	if len(a) < 2 {
		return
	}
	p := a[len(a)-1]
	i := 0
	for j := 0; j < len(a)-1; j++ {
		if a[j] < p {
			a[i], a[j] = a[j], a[i]
			i++
		}
	}
	a[i], a[len(a)-1] = a[len(a)-1], a[i]
	quicksort(a[:i])
	quicksort(a[i+1:])
}

func main() {
	var s uint32 = 12345
	a := make([]int, 40)
	for i := range a {
		s = s*1103515245 + 12345
		a[i] = int(s>>16) % 1000
	}
	fmt.Println(a)
	quicksort(a)
	fmt.Println(a)
	ok := true
	for i := 1; i < len(a); i++ {
		if a[i-1] > a[i] {
			ok = false
		}
	}
	fmt.Println(ok)
}
