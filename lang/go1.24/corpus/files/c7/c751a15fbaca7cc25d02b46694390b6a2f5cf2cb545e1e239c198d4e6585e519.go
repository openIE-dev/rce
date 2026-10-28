// SPDX-License-Identifier: MIT OR Apache-2.0
// Word frequencies of a fixed text, sorted by count and then by word.
package main

import (
	"fmt"
	"sort"
	"strings"
	"unicode"
)

const text = `It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it was the
epoch of incredulity, it was the season of Light, it was the season of Darkness`

func main() {
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool { return !unicode.IsLetter(r) })
	counts := map[string]int{}
	for _, w := range words {
		counts[w]++
	}
	type wc struct {
		w string
		c int
	}
	var list []wc
	for w, c := range counts {
		list = append(list, wc{w, c})
	}
	sort.Slice(list, func(i, j int) bool {
		if list[i].c != list[j].c {
			return list[i].c > list[j].c
		}
		return list[i].w < list[j].w
	})
	for _, e := range list {
		fmt.Println(e.c, e.w)
	}
	fmt.Println(len(words), len(counts))
}
