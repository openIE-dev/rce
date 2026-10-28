// SPDX-License-Identifier: MIT OR Apache-2.0
// The CRC-32 of IEEE 802.3 computed with a table built at run time, checked
// against hash/crc32.
package main

import (
	"fmt"
	"hash/crc32"
)

func main() {
	var table [256]uint32
	for i := range table {
		c := uint32(i)
		for k := 0; k < 8; k++ {
			if c&1 == 1 {
				c = 0xEDB88320 ^ (c >> 1)
			} else {
				c >>= 1
			}
		}
		table[i] = c
	}
	for _, s := range []string{"", "a", "123456789", "The quick brown fox jumps over the lazy dog"} {
		c := ^uint32(0)
		for i := 0; i < len(s); i++ {
			c = table[byte(c)^s[i]] ^ (c >> 8)
		}
		c = ^c
		fmt.Printf("%08x %v\n", c, c == crc32.ChecksumIEEE([]byte(s)))
	}
}
