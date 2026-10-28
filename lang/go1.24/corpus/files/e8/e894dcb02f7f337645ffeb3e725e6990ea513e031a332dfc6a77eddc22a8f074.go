// SPDX-License-Identifier: MIT OR Apache-2.0
// IEEE 754 binary64 arithmetic on float64 values held in variables.
package main

import (
	"fmt"
	"math"
)

func main() {
	a, b := 0.1, 0.2
	fmt.Println(a+b, a+b == 0.3, a*3, 1-0.9)
	one, three := 1.0, 3.0
	fmt.Println(one/three, 2*one/three, one/three*3)
	big := 1e300
	fmt.Println(big*1e10, -big*1e10, big*big/big)
	fmt.Println(math.Sqrt(2), math.Sqrt(2)*math.Sqrt(2), math.Abs(-2.5))
	fmt.Println(math.Floor(-2.5), math.Ceil(-2.5), math.Trunc(-2.5), math.Round(-2.5), math.RoundToEven(-2.5))
	fmt.Println(math.Mod(7.5, 2), math.Mod(-7.5, 2), math.Remainder(7.5, 2))
	x := 0.0
	for i := 0; i < 10; i++ {
		x += 0.1
	}
	fmt.Println(x, x == 1)
	fmt.Println(math.MaxFloat64, math.SmallestNonzeroFloat64, math.Pi, math.E)
}
