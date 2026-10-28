// SPDX-License-Identifier: MIT OR Apache-2.0
// A goroutine's stack grows as needed: recursion 1000000 deep completes
// within the default maximum stack size.
package main

import "fmt"

func sum(n int) int {
	if n == 0 {
		return 0
	}
	return n + sum(n-1)
}

func main() {
	fmt.Println(sum(1000), sum(100000), sum(1000000))
}
