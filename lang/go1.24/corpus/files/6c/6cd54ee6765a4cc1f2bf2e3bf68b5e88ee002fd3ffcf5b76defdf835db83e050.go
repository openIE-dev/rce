// SPDX-License-Identifier: MIT OR Apache-2.0
// Maps: insertion, lookup of a missing key, the comma-ok form, deletion,
// len, and printing (fmt prints a map's entries in key order).
package main

import "fmt"

func main() {
	ages := map[string]int{"ann": 31, "bob": 27}
	ages["cid"] = 45
	ages["ann"]++
	fmt.Println(ages, len(ages))
	v, ok := ages["dee"]
	fmt.Println(v, ok, ages["dee"], len(ages))
	v, ok = ages["bob"]
	fmt.Println(v, ok)
	delete(ages, "bob")
	delete(ages, "nobody")
	fmt.Println(ages, len(ages))
	counts := map[rune]int{}
	for _, r := range "mississippi" {
		counts[r]++
	}
	fmt.Println(counts)
	nested := map[int][]string{}
	nested[2] = append(nested[2], "b")
	nested[1] = append(nested[1], "a")
	nested[2] = append(nested[2], "c")
	fmt.Println(nested)
	var bools = map[bool]int{true: 1, false: 0}
	fmt.Println(bools)
	clear(ages)
	fmt.Println(ages, len(ages))
}
