// SPDX-License-Identifier: MIT OR Apache-2.0
// A shift by a negative count held in a signed variable is a run-time
// panic.
package main

import "fmt"

func main() {
	x := 1
	for _, s := range []int{2, 0, -1} {
		fmt.Println(x << s)
	}
}
