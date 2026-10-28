// SPDX-License-Identifier: MIT OR Apache-2.0
// float32 arithmetic rounds each result to binary32.
package main

import (
	"fmt"
	"math"
)

func main() {
	var a, b float32 = 0.1, 0.2
	fmt.Println(a+b, a+b == 0.3, float64(a+b))
	var x float32 = 16777216
	fmt.Println(x+1, x+2, x+1 == x)
	var big float32 = math.MaxFloat32
	fmt.Println(big*2, -big*2, big+1 == big)
	var tiny float32 = math.SmallestNonzeroFloat32
	fmt.Println(tiny, tiny/2, tiny*0.5 == 0)
	var s float32
	for i := 0; i < 10; i++ {
		s += 0.1
	}
	fmt.Println(s, s == 1)
	fmt.Printf("%v %g %e %.10f\n", float32(1)/3, float32(1)/3, float32(1)/3, float32(1)/3)
	fmt.Println(math.Float32bits(float32(1)/3), math.Float32frombits(0x7f7fffff))
}
