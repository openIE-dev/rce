// SPDX-License-Identifier: MIT OR Apache-2.0
// os.Exit ends the program at once with the given status; deferred calls
// do not run.
package main

import (
	"fmt"
	"os"
)

func main() {
	defer fmt.Println("never printed")
	fmt.Println("about to exit")
	os.Exit(3)
}
