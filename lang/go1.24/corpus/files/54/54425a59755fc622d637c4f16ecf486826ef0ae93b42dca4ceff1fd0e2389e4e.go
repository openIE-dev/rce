// SPDX-License-Identifier: MIT OR Apache-2.0
// Comparison of integers, floats, strings, arrays and structs.
package main

import (
	"fmt"
	"math"
)

type pair struct {
	A int
	B string
}

func main() {
	fmt.Println(3 < 5, 5 <= 5, 7 > 7, -1 >= -2, 4 == 4, 4 != 4)
	nan := math.NaN()
	fmt.Println(nan == nan, nan < 1, nan > 1, nan != nan)
	negZero := math.Copysign(0, -1)
	fmt.Println(negZero == 0, math.Signbit(negZero), 1/negZero)
	fmt.Println("apple" < "banana", "apple" < "Apple", "a" < "aa", "" < "a", "Z" < "a")
	fmt.Println([3]int{1, 2, 3} == [3]int{1, 2, 3}, [2]string{"x", "y"} != [2]string{"x", "z"})
	fmt.Println(pair{1, "a"} == pair{1, "a"}, pair{1, "a"} == pair{1, "b"})
	var i interface{} = 3
	var j interface{} = 3.0
	fmt.Println(i == 3, j == 3.0, i == j)
}
