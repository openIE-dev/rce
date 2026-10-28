// SPDX-License-Identifier: MIT OR Apache-2.0
// Printing of special and extreme float64 values with each verb.
package main

import (
	"fmt"
	"math"
	"strconv"
)

func main() {
	vals := []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.Copysign(0, -1), 5e-324, math.MaxFloat64, 2.2250738585072014e-308, 1e21, 1e20, 123456789.0, 0.000001, 1.0000000000000002, 0.30000000000000004, 100, 1e100, 4.35, 2.675}
	for _, v := range vals {
		fmt.Printf("%v|%g|%e|%.3f|%.17g|%s|%s|%s\n", v, v, v, v, v,
			strconv.FormatFloat(v, 'g', -1, 64), strconv.FormatFloat(v, 'f', -1, 64), strconv.FormatFloat(v, 'b', -1, 64))
	}
	fmt.Println(strconv.FormatFloat(0.1, 'g', -1, 32), strconv.FormatFloat(1.0/3, 'e', 3, 64), strconv.FormatFloat(255, 'x', -1, 64))
}
