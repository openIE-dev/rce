// SPDX-License-Identifier: MIT OR Apache-2.0
// Fibonacci numbers in uint64 and int64: past F(93) the sums wrap modulo
// 2^64, and int64 wraps into negative values past F(92).
package main

import "fmt"

func main() {
	var a, b uint64 = 0, 1
	var c, d int64 = 0, 1
	for i := 0; i <= 100; i++ {
		if i >= 88 {
			fmt.Println(i, a, c)
		}
		a, b = b, a+b
		c, d = d, c+d
	}
}
