// SPDX-License-Identifier: MIT OR Apache-2.0
// The three forms of for, nested loops and range over integers and strings.
package main

import "fmt"

func main() {
	for i := 0; i < 3; i++ {
		fmt.Print(i, " ")
	}
	fmt.Println()
	n := 1
	for n < 100 {
		n *= 3
	}
	fmt.Println(n)
	k := 0
	for {
		k++
		if k == 4 {
			break
		}
	}
	fmt.Println(k)
	for i := 1; i <= 3; i++ {
		for j := 1; j <= 3; j++ {
			fmt.Print(i*j, " ")
		}
		fmt.Println()
	}
	for i := 10; i > 0; i -= 3 {
		fmt.Print(i, " ")
	}
	fmt.Println()
	for i, c := range "go!" {
		fmt.Print(i, ":", string(c), " ")
	}
	fmt.Println()
	for i := range 4 {
		fmt.Print(i)
	}
	fmt.Println()
}
