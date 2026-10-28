// SPDX-License-Identifier: MIT OR Apache-2.0
// Writing to a nil map is a run-time panic.
package main

import "fmt"

func main() {
	var m map[string]int
	fmt.Println(len(m))
	m["x"] = 1
	fmt.Println("not reached")
}
