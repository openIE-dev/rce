// SPDX-License-Identifier: MIT OR Apache-2.0
// Two linear congruential generators with fixed seeds: one modulo 2^31 - 1
// (MINSTD) and one modulo 2^64 (wrapping uint64 arithmetic).
package main

import "fmt"

func main() {
	x := int64(1)
	for i := 0; i < 10; i++ {
		x = x * 48271 % 2147483647
		fmt.Print(x, " ")
	}
	fmt.Println()
	var s uint64 = 42
	for i := 0; i < 10; i++ {
		s = s*6364136223846793005 + 1442695040888963407
		fmt.Print(s>>33, " ")
	}
	fmt.Println()
	hist := make([]int, 10)
	for i := 0; i < 100000; i++ {
		s = s*6364136223846793005 + 1442695040888963407
		hist[(s>>33)%10]++
	}
	fmt.Println(hist)
}
