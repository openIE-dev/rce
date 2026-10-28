// SPDX-License-Identifier: MIT OR Apache-2.0
// The towers of Hanoi: the moves for four discs and the count for twenty.
package main

import "fmt"

func hanoi(n int, from, to, via string, moves *[]string) {
	if n == 0 {
		return
	}
	hanoi(n-1, from, via, to, moves)
	*moves = append(*moves, from+to)
	hanoi(n-1, via, to, from, moves)
}

func count(n int) int {
	if n == 0 {
		return 0
	}
	return 2*count(n-1) + 1
}

func main() {
	var moves []string
	hanoi(4, "A", "C", "B", &moves)
	fmt.Println(len(moves), moves)
	var big []string
	hanoi(20, "A", "C", "B", &big)
	fmt.Println(len(big), count(20), big[len(big)-1])
}
