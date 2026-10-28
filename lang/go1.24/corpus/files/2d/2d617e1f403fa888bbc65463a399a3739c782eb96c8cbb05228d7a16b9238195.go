// SPDX-License-Identifier: MIT OR Apache-2.0
// Expression switches, switches with no tag, fallthrough, and type switches.
package main

import "fmt"

func kind(v interface{}) string {
	switch x := v.(type) {
	case nil:
		return "nil"
	case int:
		return fmt.Sprint("int ", x+1)
	case string:
		return "string " + x
	case []int:
		return fmt.Sprint("slice of ", len(x))
	case error:
		return "error"
	default:
		return fmt.Sprintf("other %T", x)
	}
}

func main() {
	for _, d := range []int{1, 2, 3, 6, 7} {
		switch d {
		case 1, 7:
			fmt.Println(d, "weekend")
		case 2:
			fmt.Println(d, "monday")
			fallthrough
		case 3:
			fmt.Println(d, "early week")
		default:
			fmt.Println(d, "late week")
		}
	}
	x := 15
	switch {
	case x > 10 && x < 20:
		fmt.Println("teens")
	case x > 5:
		fmt.Println("not reached")
	}
	switch y := x % 4; y {
	case 0:
		fmt.Println("divisible")
	default:
		fmt.Println("remainder", y)
	}
	fmt.Println(kind(nil), kind(4), kind("s"), kind([]int{1, 2}), kind(2.5), kind(fmt.Errorf("e")))
}
