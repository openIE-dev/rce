// SPDX-License-Identifier: MIT OR Apache-2.0
// Boolean operators, and the short-circuit evaluation of && and ||.
package main

import "fmt"

var calls []string

func t(name string) bool {
	calls = append(calls, name)
	return true
}

func f(name string) bool {
	calls = append(calls, name)
	return false
}

func main() {
	fmt.Println(true && false, true || false, !true, !false)
	r1 := f("a") && t("b")
	r2 := t("c") || f("d")
	r3 := t("e") && f("f") || t("g")
	r4 := !(f("h") || f("i"))
	fmt.Println(r1, r2, r3, r4)
	fmt.Println(calls)
	x := 5
	fmt.Println(x > 3 && x < 10, x < 3 || x == 5, x != 5)
}
