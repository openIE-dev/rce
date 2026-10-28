// SPDX-License-Identifier: MIT OR Apache-2.0
// Case mapping and classification of runes with the unicode and strings
// packages, which map one rune to one rune.
package main

import (
	"fmt"
	"strings"
	"unicode"
)

func main() {
	words := []string{"straße", "İstanbul", "ǅemal", "ΣΊΣΥΦΟΣ", "ﬁne", "Ⅻ", "ǈ"}
	for _, w := range words {
		fmt.Println(w, strings.ToUpper(w), strings.ToLower(w), strings.ToTitle(w), len(strings.ToLower(w)))
	}
	for _, r := range []rune{'a', 'Z', '5', '٣', '²', '½', ' ', '\u00a0', '\u2028', 'é', '_', '日'} {
		fmt.Printf("%U %v %v %v %v %v %v\n", r, unicode.IsLetter(r), unicode.IsDigit(r), unicode.IsNumber(r), unicode.IsSpace(r), unicode.IsUpper(r), unicode.IsPunct(r))
	}
	fmt.Println(strings.EqualFold("straße", "STRASSE"), strings.EqualFold("Σ", "ς"), strings.EqualFold("K", "\u212a"))
	fmt.Println(unicode.Version)
}
