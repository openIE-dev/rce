// SPDX-License-Identifier: MIT OR Apache-2.0
// Recursion with no base case passes the goroutine's maximum stack size,
// set here to 1 MiB; the runtime ends the program.
package main

import (
	"fmt"
	"runtime/debug"
)

func down(n int) int {
	return down(n+1) + 1
}

func main() {
	debug.SetMaxStack(1 << 20)
	fmt.Println("start")
	fmt.Println(down(0))
}
