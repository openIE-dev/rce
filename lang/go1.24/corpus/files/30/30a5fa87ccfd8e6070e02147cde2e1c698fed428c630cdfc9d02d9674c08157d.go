// SPDX-License-Identifier: MIT OR Apache-2.0
// Addition, subtraction, multiplication and precedence on int.
package main

import "fmt"

func main() {
	a, b, c := 17, 5, -3
	fmt.Println(a+b, a-b, a*b, b-a, a*c)
	fmt.Println(a+b*c, (a+b)*c, a-b-c, a-(b-c))
	fmt.Println(-a, - -a, +c)
	x := 1
	for i := 0; i < 10; i++ {
		x = x*3 + i
	}
	fmt.Println(x)
	x += 5
	x -= 2
	x *= 4
	fmt.Println(x)
	x++
	x--
	x--
	fmt.Println(x)
}
