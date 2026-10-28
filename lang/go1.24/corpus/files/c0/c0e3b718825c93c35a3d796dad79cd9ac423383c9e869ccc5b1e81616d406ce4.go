// SPDX-License-Identifier: MIT OR Apache-2.0
// Integer literals in every base, digit separators, and printing in bases.
package main

import (
	"fmt"
	"strconv"
)

func main() {
	fmt.Println(255, 0xff, 0XFF, 0o377, 0377, 0b11111111, 1_000_000, 0x_ff_ff)
	fmt.Println('a', '\n', '\x7f', 'é', '\U0001F600')
	n := 48879
	fmt.Printf("%d %x %X %o %b %#x %#o %#b\n", n, n, n, n, n, n, n, n)
	fmt.Println(strconv.FormatInt(-255, 2), strconv.FormatInt(1295, 36), strconv.Itoa(-12))
	v, _ := strconv.ParseInt("-ff", 16, 64)
	w, _ := strconv.ParseInt("0x1f", 0, 64)
	u, _ := strconv.ParseUint("777", 8, 64)
	fmt.Println(v, w, u)
	fmt.Println(1.5e3, 0x1p-2, 1_0.5)
}
