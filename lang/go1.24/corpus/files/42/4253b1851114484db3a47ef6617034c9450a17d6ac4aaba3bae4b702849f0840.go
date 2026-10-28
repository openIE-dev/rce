// SPDX-License-Identifier: MIT OR Apache-2.0
// Strings compare by bytes, so the order of UTF-8 encodings is the order of
// code points.
package main

import (
	"fmt"
	"sort"
	"strings"
)

func main() {
	pairs := [][2]string{{"\uffff", "\U00010000"}, {"é", "f"}, {"Z", "a"}, {"abc", "abd"}, {"ab", "abc"}, {"\x00", ""}, {"é", "e\u0301"}}
	for _, p := range pairs {
		fmt.Printf("%+q %+q %v %d\n", p[0], p[1], p[0] < p[1], strings.Compare(p[0], p[1]))
	}
	ws := []string{"\U0001F600", "\uffff", "zebra", "Zebra", "éclair", "eclair", "", "10", "9"}
	sort.Strings(ws)
	fmt.Printf("%+q\n", ws)
}
