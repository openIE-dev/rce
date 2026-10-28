// SPDX-License-Identifier: MIT OR Apache-2.0
// Arithmetic on sized integers held in variables wraps modulo 2^n: the
// specification's rule for integer overflow.
package main

import (
	"fmt"
	"math"
)

func main() {
	var i8 int8 = 127
	var u8 uint8 = 255
	var i16 int16 = math.MaxInt16
	var u16 uint16 = 0
	var i32 int32 = math.MaxInt32
	var u32 uint32 = math.MaxUint32
	var i64 int64 = math.MaxInt64
	var u64 uint64 = 0
	i8++
	u8++
	i16 += 2
	u16--
	i32 *= 2
	u32 += 2
	i64++
	u64--
	fmt.Println(i8, u8, i16, u16, i32, u32, i64, u64)
	var m8 int8 = -128
	fmt.Println(-m8, m8-1, m8*m8, m8*-1)
	var big int64 = math.MinInt64
	fmt.Println(-big, big-1, big*-1, big+big)
	var n int = math.MaxInt
	n++
	fmt.Println(n == math.MinInt, n)
	var m uint = 0
	m -= 1
	fmt.Println(m, m == math.MaxUint)
	x := int32(46341)
	fmt.Println(x*x, int64(x)*int64(x))
}
