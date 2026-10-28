// SPDX-License-Identifier: MIT OR Apache-2.0
// Shifts by counts at and past the operand's width, and arithmetic right
// shifts of negative values.
package main

import "fmt"

func main() {
	var u8 uint8 = 0xff
	var i8 int8 = -128
	var u32 uint32 = 1
	var i64 int64 = -1
	for _, s := range []uint{0, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 200} {
		fmt.Println(s, u8<<s, u8>>s, i8<<s, i8>>s, u32<<s, u32>>s, i64<<s, i64>>s)
	}
	var k int = 3
	fmt.Println(1<<k, -16>>k, uint64(1)<<63)
}
