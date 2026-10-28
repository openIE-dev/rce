// SPDX-License-Identifier: MIT OR Apache-2.0
// break and continue, with and without labels, and goto.
package main

import "fmt"

func main() {
	for i := 0; i < 10; i++ {
		if i%2 == 1 {
			continue
		}
		if i > 6 {
			break
		}
		fmt.Print(i, " ")
	}
	fmt.Println()
outer:
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			if j == 2 {
				continue outer
			}
			if i == 3 {
				break outer
			}
			fmt.Print(i, j, " ")
		}
	}
	fmt.Println()
	i := 0
loop:
	if i < 3 {
		fmt.Print("g", i, " ")
		i++
		goto loop
	}
	fmt.Println()
	found := -1
search:
	for r, row := range [][]int{{1, 2}, {3, 4}, {5, 6}} {
		for _, v := range row {
			if v == 4 {
				found = r
				break search
			}
		}
	}
	fmt.Println(found)
}
