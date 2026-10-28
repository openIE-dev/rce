// SPDX-License-Identifier: MIT OR Apache-2.0
// Concatenation, length in bytes, indexing a byte and slicing.
package main

import "fmt"

func main() {
	s := "hello"
	t := s + ", " + "world"
	fmt.Println(t, len(t))
	fmt.Println(s[0], s[1:3], s[:2], s[3:], s[:])
	fmt.Println(string(s[0]), s[len(s)-1])
	b := []byte(s)
	b[0] = 'j'
	fmt.Println(string(b), s)
	r := []rune("héllo")
	fmt.Println(len("héllo"), len(r), string(r[1]))
	fmt.Println(s == "hello", s+"" == s)
	u := ""
	for i := 0; i < 5; i++ {
		u += string(rune('a' + i))
	}
	fmt.Println(u)
}
