// SPDX-License-Identifier: MIT OR Apache-2.0
// strconv at its edges: out-of-range and malformed input, bases and
// quoting.
package main

import (
	"fmt"
	"strconv"
)

func main() {
	for _, s := range []string{"9223372036854775807", "9223372036854775808", "-9223372036854775808", "-9223372036854775809", " 1", "+1", "0x10", "1_000", ""} {
		n, err := strconv.ParseInt(s, 10, 64)
		fmt.Printf("%q %d %v\n", s, n, err)
	}
	for _, s := range []string{"0x10", "0b101", "0o17", "017", "1_000", "-0x8000000000000000"} {
		n, err := strconv.ParseInt(s, 0, 64)
		fmt.Printf("%q %d %v\n", s, n, err)
	}
	for _, s := range []string{"1e400", "-1e400", "1e-400", "NaN", "inf", "0x1p-2", "1_0.5", ".5", "5.", "1e", "0.1"} {
		f, err := strconv.ParseFloat(s, 64)
		fmt.Printf("%q %v %v\n", s, f, err)
	}
	n8, err := strconv.ParseInt("200", 10, 8)
	fmt.Println(n8, err)
	u, err := strconv.ParseUint("-1", 10, 64)
	fmt.Println(u, err)
	b, err := strconv.ParseBool("T")
	fmt.Println(b, err)
	fmt.Println(strconv.Quote("tab\there \"q\" é \x00 \u2028"), strconv.QuoteToASCII("é😀"), strconv.QuoteRune('☺'))
	s, err := strconv.Unquote(`"a\tbé"`)
	fmt.Println(s, err)
}
