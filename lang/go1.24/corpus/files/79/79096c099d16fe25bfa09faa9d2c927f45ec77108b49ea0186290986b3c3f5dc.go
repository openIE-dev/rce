// SPDX-License-Identifier: MIT OR Apache-2.0
// Closures that capture variables; since go1.22 each iteration of a for
// loop has its own loop variable.
package main

import "fmt"

func counter() func() int {
	n := 0
	return func() int {
		n++
		return n
	}
}

func adder(base int) func(int) int {
	return func(x int) int { return base + x }
}

func main() {
	c1, c2 := counter(), counter()
	fmt.Println(c1(), c1(), c1(), c2())
	add5 := adder(5)
	fmt.Println(add5(1), add5(10))
	var fs []func() int
	for i := 0; i < 3; i++ {
		fs = append(fs, func() int { return i * 10 })
	}
	for _, f := range fs {
		fmt.Print(f(), " ")
	}
	fmt.Println()
	shared := 0
	inc := func() { shared++ }
	inc()
	inc()
	fmt.Println(shared)
	var gs []func() string
	for _, s := range []string{"a", "b"} {
		gs = append(gs, func() string { return s })
	}
	fmt.Println(gs[0](), gs[1]())
	fib := func() func() int {
		a, b := 0, 1
		return func() int {
			r := a
			a, b = b, a+b
			return r
		}
	}()
	for i := 0; i < 8; i++ {
		fmt.Print(fib(), " ")
	}
	fmt.Println()
}
