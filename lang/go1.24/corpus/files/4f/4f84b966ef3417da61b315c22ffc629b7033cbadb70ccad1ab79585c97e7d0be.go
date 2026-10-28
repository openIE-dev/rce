// SPDX-License-Identifier: MIT OR Apache-2.0
// Constant declarations, iota, and typed and untyped constants.
package main

import "fmt"

type weekday int

const (
	sunday weekday = iota
	monday
	tuesday
)

const (
	_  = iota
	kb = 1 << (10 * iota)
	mb
	gb
)

const (
	a = iota * 2
	b
	c = "text"
	d
	e = iota
)

func (d weekday) String() string {
	return [...]string{"Sun", "Mon", "Tue"}[d]
}

func main() {
	fmt.Println(sunday, monday, tuesday, int(tuesday))
	fmt.Println(kb, mb, gb)
	fmt.Println(a, b, c, d, e)
	const big = 1 << 70
	fmt.Println(big >> 65)
	const third = 1.0 / 3
	fmt.Println(third, third*3 == 1)
	var f32 float32 = third
	fmt.Println(f32)
}
