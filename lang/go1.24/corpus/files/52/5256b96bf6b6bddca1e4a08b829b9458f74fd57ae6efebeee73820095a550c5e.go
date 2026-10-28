// SPDX-License-Identifier: MIT OR Apache-2.0
// Prime factorisation by trial division, including a large uint64.
package main

import "fmt"

func factors(n uint64) []uint64 {
	var fs []uint64
	for p := uint64(2); p*p <= n; p++ {
		for n%p == 0 {
			fs = append(fs, p)
			n /= p
		}
	}
	if n > 1 {
		fs = append(fs, n)
	}
	return fs
}

func main() {
	for _, n := range []uint64{1, 2, 12, 97, 360, 1001, 65536, 600851475143, 9999999967, 18446744073709551615} {
		fmt.Println(n, factors(n))
	}
}
