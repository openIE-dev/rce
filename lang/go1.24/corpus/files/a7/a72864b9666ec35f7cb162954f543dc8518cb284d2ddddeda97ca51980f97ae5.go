// SPDX-License-Identifier: MIT OR Apache-2.0
// FNV-1a hashes in 32 and 64 bits; the multiplications wrap modulo 2^32
// and 2^64. Checked against hash/fnv.
package main

import (
	"fmt"
	"hash/fnv"
)

func fnv32a(s string) uint32 {
	h := uint32(2166136261)
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return h
}

func fnv64a(s string) uint64 {
	h := uint64(14695981039346656037)
	for i := 0; i < len(s); i++ {
		h ^= uint64(s[i])
		h *= 1099511628211
	}
	return h
}

func main() {
	for _, s := range []string{"", "a", "foobar", "hello, world", "日本"} {
		h32 := fnv.New32a()
		h32.Write([]byte(s))
		h64 := fnv.New64a()
		h64.Write([]byte(s))
		fmt.Printf("%q %08x %016x %v %v\n", s, fnv32a(s), fnv64a(s), fnv32a(s) == h32.Sum32(), fnv64a(s) == h64.Sum64())
	}
}
