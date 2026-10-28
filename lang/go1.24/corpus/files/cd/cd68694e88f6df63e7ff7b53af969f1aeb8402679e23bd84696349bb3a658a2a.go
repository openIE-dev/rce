// SPDX-License-Identifier: MIT OR Apache-2.0
// The most negative value divided by -1: the quotient equals the dividend
// and the remainder is 0, with no panic, for every signed size.
package main

import (
	"fmt"
	"math"
)

func main() {
	var a8 int8 = math.MinInt8
	var a16 int16 = math.MinInt16
	var a32 int32 = math.MinInt32
	var a64 int64 = math.MinInt64
	var d8, d16, d32, d64 = int8(-1), int16(-1), int32(-1), int64(-1)
	fmt.Println(a8/d8, a8%d8)
	fmt.Println(a16/d16, a16%d16)
	fmt.Println(a32/d32, a32%d32)
	fmt.Println(a64/d64, a64%d64)
	var mi int = math.MinInt
	var neg int = -1
	fmt.Println(mi/neg, mi%neg)
}
