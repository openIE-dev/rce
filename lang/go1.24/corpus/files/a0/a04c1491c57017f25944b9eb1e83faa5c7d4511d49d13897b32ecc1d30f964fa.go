// SPDX-License-Identifier: MIT OR Apache-2.0
// The text that fmt gives float64 values under %v, %g, %e, %f and fixed
// precisions.
package main

import "fmt"

func main() {
	vals := []float64{0, 1, -1, 0.5, 0.1, 1.0 / 3, 100, 123456, 1234567, 1e6, 1e20, 1e21, 1e-4, 1e-5, 12345.678, 2.5e-8}
	for _, v := range vals {
		fmt.Printf("%v|%g|%e|%f|%.2f|%8.3f|%-8.1f|%G\n", v, v, v, v, v, v, v, v)
	}
	fmt.Println(float32(0.1), float64(float32(0.1)), float32(16777217))
	fmt.Printf("%5.1f|%05.1f|%+.1f|% .1f\n", 3.14159, 3.14159, 3.14159, 3.14159)
}
