package main

import (
	"fmt"
	"sort"
	"strings"
)

func main() {
	words := strings.Fields("the quick brown fox jumps over the lazy dog the end")
	count := map[string]int{}
	for _, w := range words {
		count[w]++
	}
	keys := make([]string, 0, len(count))
	for k := range count {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		fmt.Println(k, count[k])
	}
}
