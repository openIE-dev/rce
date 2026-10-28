// SPDX-License-Identifier: MIT OR Apache-2.0
// Indexing past the end of a slice is a run-time panic.
package main

import "fmt"

func main() {
	xs := []int{10, 20, 30}
	for i := 0; i <= len(xs); i++ {
		fmt.Println(i, xs[i])
	}
}
