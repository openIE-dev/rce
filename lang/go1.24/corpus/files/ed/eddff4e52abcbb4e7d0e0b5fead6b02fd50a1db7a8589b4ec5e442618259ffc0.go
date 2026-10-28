// SPDX-License-Identifier: MIT OR Apache-2.0
// Run-length encoding and decoding over runes.
package main

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

func encode(s string) string {
	r := []rune(s)
	var b strings.Builder
	for i := 0; i < len(r); {
		j := i
		for j < len(r) && r[j] == r[i] {
			j++
		}
		b.WriteString(strconv.Itoa(j - i))
		b.WriteRune(r[i])
		i = j
	}
	return b.String()
}

func decode(s string) string {
	var b strings.Builder
	n := 0
	for _, c := range s {
		if unicode.IsDigit(c) {
			n = n*10 + int(c-'0')
			continue
		}
		b.WriteString(strings.Repeat(string(c), n))
		n = 0
	}
	return b.String()
}

func main() {
	for _, s := range []string{"", "a", "aaabccddddde", "WWWWWWWWWWWWBWWWWWWWWWWWWBBBWWWWWWWWWWWWWWWWWWWWWWWWBWWWWWWWWWWWWWW", "ééé日日x"} {
		e := encode(s)
		fmt.Printf("%q %q %v\n", s, e, decode(e) == s)
	}
}
