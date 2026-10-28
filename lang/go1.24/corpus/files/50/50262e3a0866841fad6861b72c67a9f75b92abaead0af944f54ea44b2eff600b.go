// SPDX-License-Identifier: MIT OR Apache-2.0
// Floating-point division by zero gives infinities and NaN, with no panic.
package main

import (
	"fmt"
	"math"
)

func main() {
	zero := 0.0
	negZero := math.Copysign(0, -1)
	fmt.Println(1/zero, -1/zero, zero/zero, 1/negZero)
	inf := math.Inf(1)
	fmt.Println(inf-inf, inf*0, inf+1, -inf < inf, math.IsNaN(inf-inf))
	nan := math.NaN()
	fmt.Println(nan+1, math.Max(nan, 1), math.Min(-0.0, 0), math.Max(negZero, 0))
	fmt.Println(math.Sqrt(-1), math.Log(0), math.Log(-1), math.Pow(0, -1))
	fmt.Printf("%v %v %.2f %e %g\n", nan, -inf, inf, nan, negZero)
	fmt.Println(math.Float64bits(negZero), math.Float64bits(inf))
	var f32zero float32
	fmt.Println(float32(1)/f32zero, f32zero/f32zero)
}
