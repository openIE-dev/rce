// SPDX-License-Identifier: MIT OR Apache-2.0
// The sieve of Eratosthenes: the primes below 200 and the count below 100000.
package main

import "fmt"

func sieve(n int) []bool {
	composite := make([]bool, n)
	for i := 2; i*i < n; i++ {
		if !composite[i] {
			for j := i * i; j < n; j += i {
				composite[j] = true
			}
		}
	}
	return composite
}

func main() {
	c := sieve(200)
	for i := 2; i < 200; i++ {
		if !c[i] {
			fmt.Print(i, " ")
		}
	}
	fmt.Println()
	big := sieve(100000)
	count := 0
	for i := 2; i < len(big); i++ {
		if !big[i] {
			count++
		}
	}
	fmt.Println(count)
}
