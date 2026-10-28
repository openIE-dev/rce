// SPDX-License-Identifier: MIT OR Apache-2.0
// Recursive functions: factorial, Fibonacci, Ackermann and mutual recursion.
package main

import "fmt"

func fact(n int) int {
	if n <= 1 {
		return 1
	}
	return n * fact(n-1)
}

func fib(n int) int {
	if n < 2 {
		return n
	}
	return fib(n-1) + fib(n-2)
}

func ack(m, n int) int {
	if m == 0 {
		return n + 1
	}
	if n == 0 {
		return ack(m-1, 1)
	}
	return ack(m-1, ack(m, n-1))
}

func isEven(n int) bool {
	if n == 0 {
		return true
	}
	return isOdd(n - 1)
}

func isOdd(n int) bool {
	if n == 0 {
		return false
	}
	return isEven(n - 1)
}

func main() {
	for i := 0; i <= 20; i += 5 {
		fmt.Println(i, fact(i))
	}
	for i := 0; i < 15; i++ {
		fmt.Print(fib(i), " ")
	}
	fmt.Println()
	fmt.Println(ack(2, 3), ack(3, 3))
	fmt.Println(isEven(10), isOdd(7), isEven(7))
}
