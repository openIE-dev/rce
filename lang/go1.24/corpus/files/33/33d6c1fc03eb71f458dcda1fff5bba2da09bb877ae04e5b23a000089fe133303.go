// SPDX-License-Identifier: MIT OR Apache-2.0
// if, else if and else, with an initialising statement.
package main

import "fmt"

func classify(n int) string {
	if n < 0 {
		return "negative"
	} else if n == 0 {
		return "zero"
	} else if n%2 == 0 {
		return "even"
	} else {
		return "odd"
	}
}

func main() {
	for _, n := range []int{-5, 0, 4, 7} {
		fmt.Println(n, classify(n))
	}
	if x := 10 * 3; x > 20 {
		fmt.Println("big", x)
	} else {
		fmt.Println("small", x)
	}
	grade := 83
	if grade >= 90 {
		fmt.Println("A")
	} else if grade >= 80 {
		fmt.Println("B")
	} else {
		fmt.Println("C")
	}
}
