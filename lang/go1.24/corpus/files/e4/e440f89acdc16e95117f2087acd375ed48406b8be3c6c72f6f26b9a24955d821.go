// SPDX-License-Identifier: MIT OR Apache-2.0
// A program writes to both standard output and standard error and exits
// with status 0.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Println("to stdout 1")
	fmt.Fprintln(os.Stderr, "to stderr")
	fmt.Println("to stdout 2")
}
