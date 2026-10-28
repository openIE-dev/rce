// SPDX-License-Identifier: MIT OR Apache-2.0
// Euclid's greatest common divisor, the least common multiple, and the
// extended algorithm's Bezout coefficients.
package main

import "fmt"

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a < 0 {
		return -a
	}
	return a
}

func egcd(a, b int) (int, int, int) {
	if b == 0 {
		return a, 1, 0
	}
	g, x, y := egcd(b, a%b)
	return g, y, x - (a/b)*y
}

func main() {
	pairs := [][2]int{{12, 18}, {17, 5}, {0, 9}, {-24, 36}, {1071, 462}, {2147483647, 2147483646}}
	for _, p := range pairs {
		g := gcd(p[0], p[1])
		l := 0
		if g != 0 {
			l = p[0] / g * p[1]
		}
		fmt.Println(p[0], p[1], g, l)
	}
	g, x, y := egcd(240, 46)
	fmt.Println(g, x, y, 240*x+46*y)
	acc := 1
	for i := 1; i <= 20; i++ {
		acc = acc / gcd(acc, i) * i
	}
	fmt.Println(acc)
}
