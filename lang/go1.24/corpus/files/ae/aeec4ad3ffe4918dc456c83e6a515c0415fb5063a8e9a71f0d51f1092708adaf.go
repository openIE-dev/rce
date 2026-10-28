// SPDX-License-Identifier: MIT OR Apache-2.0
// Building a large string, and checking it by length and checksum rather
// than printing it.
package main

import (
	"fmt"
	"strings"
)

func main() {
	var b strings.Builder
	for i := 0; i < 200000; i++ {
		b.WriteString(fmt.Sprint(i % 97))
		b.WriteByte(',')
	}
	s := b.String()
	var sum uint32
	for i := 0; i < len(s); i++ {
		sum = sum*31 + uint32(s[i])
	}
	fmt.Println(len(s), sum, s[:20], strings.Count(s, "96,"))
	r := strings.Repeat("ab", 1<<20)
	fmt.Println(len(r), strings.LastIndex(r, "ba"))
}
