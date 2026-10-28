// SPDX-License-Identifier: MIT OR Apache-2.0
// Conversions between integer types keep the low bits and sign-extend or
// zero-extend; conversions to string treat an integer as a code point.
package main

import (
	"fmt"
	"math"
)

func main() {
	vals := []int64{0, 1, -1, 127, 128, 255, 256, -129, 65535, 65536, math.MaxInt32, math.MinInt32 - 1, math.MaxInt64, math.MinInt64}
	for _, v := range vals {
		fmt.Println(v, int8(v), uint8(v), int16(v), uint16(v), int32(v), uint32(v), uint64(v))
	}
	var b byte = 'A'
	r := rune(b) + 0x1F5FF
	fmt.Println(string(rune(b)), r, string(r), string(rune(0x41)))
	var u uint32 = 0xFFFFFFFF
	fmt.Println(int32(u), int64(u), int64(int32(u)))
}
