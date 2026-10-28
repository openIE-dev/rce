// SPDX-License-Identifier: MIT OR Apache-2.0
// Integer division truncates toward zero, and the remainder has the sign of
// the dividend, for every combination of signs.
package main

import "fmt"

func main() {
	pairs := [][2]int{{7, 2}, {-7, 2}, {7, -2}, {-7, -2}, {6, 3}, {-6, 3}, {0, 5}, {0, -5}, {1, 7}, {-1, 7}, {13, 13}, {-13, 13}}
	for _, p := range pairs {
		a, b := p[0], p[1]
		fmt.Printf("%d / %d = %d, %d %% %d = %d\n", a, b, a/b, a, b, a%b)
	}
	u, v := uint(7), uint(2)
	fmt.Println(u/v, u%v)
	fmt.Println(7>>1, -7>>1, -7/2)
}
