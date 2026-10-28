// SPDX-License-Identifier: MIT OR Apache-2.0
// Untyped constant expressions are evaluated exactly; the same arithmetic
// on variables is rounded to float64 or wraps.
package main

import "fmt"

const huge = 1 << 100

func main() {
	fmt.Println(huge>>98, huge/(1<<97))
	fmt.Println(0.1+0.2 == 0.3)
	a, b := 0.1, 0.2
	fmt.Println(a+b == 0.3)
	const third = 1.0 / 3.0
	fmt.Println(third+third+third == 1)
	x := third
	fmt.Println(x+x+x == 1)
	const c = 'a' + 1
	fmt.Println(c, string(rune(c)))
	var u8 uint8 = 200
	fmt.Println(u8+100, 200+100)
	const f = 1e300 * 1e300 / 1e300
	fmt.Println(f)
	fmt.Println(7/2, 7/2.0, 7.0/2)
	var i = 7
	fmt.Println(i/2, float64(i)/2)
}
