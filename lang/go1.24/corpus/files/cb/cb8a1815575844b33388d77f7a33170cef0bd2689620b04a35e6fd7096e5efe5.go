// SPDX-License-Identifier: MIT OR Apache-2.0
// Receiving from a channel that no goroutine will ever send on: the runtime
// finds every goroutine asleep and ends the program.
package main

import "fmt"

func main() {
	ch := make(chan int)
	fmt.Println("waiting")
	fmt.Println(<-ch)
}
