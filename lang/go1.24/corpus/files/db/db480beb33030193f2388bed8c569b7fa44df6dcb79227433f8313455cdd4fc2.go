// SPDX-License-Identifier: MIT OR Apache-2.0
// Strings hold any bytes: ranging over invalid UTF-8 gives U+FFFD one byte
// at a time, and converting an invalid code point to a string gives U+FFFD.
package main

import (
	"fmt"
	"unicode/utf8"
)

func main() {
	s := "a\xffb\xe6\x97c"
	fmt.Println(len(s), utf8.RuneCountInString(s), utf8.ValidString(s))
	for i, r := range s {
		fmt.Print(i, ":", r, " ")
	}
	fmt.Println()
	fmt.Printf("%q %+q %x\n", s, s, s)
	fmt.Println([]rune(s), []byte(s))
	fmt.Println(string(rune(-1)), string(rune(0x10FFFF+1)) == "\ufffd", string(rune(0xD800)) == "\ufffd")
	fmt.Println(string([]rune{72, 105, 0x1F600}))
	b := []byte("héllo")
	fmt.Println(b, string(b[1:2]), utf8.Valid(b[1:2]))
	r, size := utf8.DecodeRuneInString("\xc3")
	fmt.Println(r, size)
	fmt.Printf("%q\n", string([]byte{0xe2, 0x82, 0xac}))
}
