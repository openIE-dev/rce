// SPDX-License-Identifier: MIT OR Apache-2.0
// Integer division by a zero held in a variable is a run-time panic.
package main

import "fmt"

func divide(a, b int) int {
	return a / b
}

func main() {
	fmt.Println(divide(10, 3))
	fmt.Println(divide(10, 0))
	fmt.Println("not reached")
}
