// SPDX-License-Identifier: MIT OR Apache-2.0
// Functions with several results, named results, variadic parameters and
// functions as values.
package main

import "fmt"

func divmod(a, b int) (int, int) {
	return a / b, a % b
}

func split(sum int) (x, y int) {
	x = sum * 4 / 9
	y = sum - x
	return
}

func sum(nums ...int) int {
	t := 0
	for _, n := range nums {
		t += n
	}
	return t
}

func apply(f func(int) int, xs []int) []int {
	out := make([]int, len(xs))
	for i, x := range xs {
		out[i] = f(x)
	}
	return out
}

func swap(a, b string) (string, string) { return b, a }

func main() {
	q, r := divmod(17, 5)
	fmt.Println(q, r)
	fmt.Println(split(17))
	fmt.Println(sum(), sum(1), sum(1, 2, 3), sum([]int{4, 5, 6}...))
	sq := func(x int) int { return x * x }
	fmt.Println(apply(sq, []int{1, 2, 3}), apply(func(x int) int { return -x }, []int{4}))
	a, b := swap("left", "right")
	fmt.Println(a, b)
	_, onlyR := divmod(9, 4)
	fmt.Println(onlyR)
	var f func() int
	fmt.Println(f == nil)
}
