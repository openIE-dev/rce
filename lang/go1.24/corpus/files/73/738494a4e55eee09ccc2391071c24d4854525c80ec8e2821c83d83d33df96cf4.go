// SPDX-License-Identifier: MIT OR Apache-2.0
// Strings are bytes in UTF-8: len counts bytes, ranging decodes runes.
package main

import (
	"fmt"
	"unicode/utf8"
)

func main() {
	words := []string{"abc", "héllo", "日本語", "😀", "e\u0301", "👍🏽", "é"}
	for _, w := range words {
		fmt.Println(w, len(w), utf8.RuneCountInString(w), len([]rune(w)))
	}
	for i, r := range "aé日😀" {
		fmt.Println(i, r, string(r), utf8.RuneLen(r))
	}
	fmt.Printf("% x\n", "日本")
	fmt.Println(utf8.ValidString("ok"), utf8.ValidString("\xff"))
}
