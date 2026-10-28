// SPDX-License-Identifier: MIT OR Apache-2.0
// A panic that no deferred function recovers ends the program with exit
// status 2, after the deferred calls run and after what was printed.
package main

import "fmt"

func check(n int) {
	if n > 2 {
		panic(fmt.Sprintf("n too large: %d", n))
	}
	fmt.Println("ok", n)
}

func main() {
	defer fmt.Println("deferred runs")
	for i := 0; i < 5; i++ {
		check(i)
	}
}
