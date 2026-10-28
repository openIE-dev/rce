// SPDX-License-Identifier: MIT OR Apache-2.0
// How fmt.Println prints values of the built-in kinds.
package main

import "fmt"

type point struct {
	X, Y int
}

func main() {
	var nilSlice []int
	var nilMap map[string]int
	var nilErr error
	fmt.Println(42, -7, 3.5, true, "text", 'x', byte(65))
	fmt.Println([]int{1, 2, 3}, [3]string{"a", "b", "c"}, nilSlice, len(nilSlice))
	fmt.Println(map[string]int{"b": 2, "a": 1, "c": 3}, nilMap)
	fmt.Println(point{1, 2}, []point{{3, 4}, {5, 6}})
	fmt.Println(nilErr, [][]int{{1}, {2, 3}, {}})
	fmt.Println("a", "b", 1, 2, "c")
	fmt.Print("x", "y", 1, 2, "z", "\n")
	fmt.Print(1, 2, "\n")
	fmt.Println(struct{}{}, [0]int{}, []string{""}, []string{"", ""})
}
