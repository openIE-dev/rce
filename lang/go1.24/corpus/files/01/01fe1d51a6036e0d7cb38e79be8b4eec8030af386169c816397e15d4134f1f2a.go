// SPDX-License-Identifier: MIT OR Apache-2.0
// Conversions between floats and integers for values each type can hold:
// float to integer truncates toward zero.
package main

import (
	"fmt"
	"math"
)

func main() {
	for _, f := range []float64{2.9, -2.9, 0.5, -0.5, 1e15 + 0.5, 9007199254740993, -9.99} {
		fmt.Println(f, int64(f), math.Trunc(f), math.Floor(f), math.Ceil(f), math.Round(f), math.RoundToEven(f))
	}
	big := int64(1)<<53 + 1
	fmt.Println(big, float64(big), int64(float64(big)), float32(big))
	fmt.Println(uint8(float64(200)), int8(float64(-100)), uint64(float64(1<<63)))
	fmt.Println(float64(math.MaxInt64), float64(math.MaxUint64), float32(math.MaxFloat32))
	var f32 float32 = 1.1
	fmt.Println(f32, float64(f32), float32(float64(f32)) == f32)
	fmt.Println(math.Float32bits(0.1), math.Float64bits(0.1), math.Float64frombits(0x3ff0000000000001))
}
