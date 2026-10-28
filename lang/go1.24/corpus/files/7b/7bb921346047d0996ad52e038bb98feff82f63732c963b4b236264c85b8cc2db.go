// SPDX-License-Identifier: MIT OR Apache-2.0
// Conversion to and from Roman numerals, checked by a round trip.
package main

import (
	"fmt"
	"strings"
)

var vals = []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
var syms = []string{"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"}

func toRoman(n int) string {
	var b strings.Builder
	for i, v := range vals {
		for n >= v {
			b.WriteString(syms[i])
			n -= v
		}
	}
	return b.String()
}

func fromRoman(s string) int {
	value := map[byte]int{'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
	total := 0
	for i := 0; i < len(s); i++ {
		v := value[s[i]]
		if i+1 < len(s) && value[s[i+1]] > v {
			total -= v
		} else {
			total += v
		}
	}
	return total
}

func main() {
	for _, n := range []int{1, 4, 9, 14, 40, 90, 400, 1994, 2024, 3999} {
		r := toRoman(n)
		fmt.Println(n, r, fromRoman(r))
	}
	bad := 0
	for n := 1; n < 4000; n++ {
		if fromRoman(toRoman(n)) != n {
			bad++
		}
	}
	fmt.Println(bad)
}
