// SPDX-License-Identifier: MIT OR Apache-2.0
// Integer matrix multiplication, powers and the determinant by cofactor
// expansion.
package main

import "fmt"

type mat [][]int64

func mul(a, b mat) mat {
	n, m, p := len(a), len(b), len(b[0])
	c := make(mat, n)
	for i := range c {
		c[i] = make([]int64, p)
		for j := 0; j < p; j++ {
			for k := 0; k < m; k++ {
				c[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return c
}

func det(a mat) int64 {
	n := len(a)
	if n == 1 {
		return a[0][0]
	}
	var d int64
	sign := int64(1)
	for c := 0; c < n; c++ {
		sub := make(mat, 0, n-1)
		for r := 1; r < n; r++ {
			row := append(append([]int64{}, a[r][:c]...), a[r][c+1:]...)
			sub = append(sub, row)
		}
		d += sign * a[0][c] * det(sub)
		sign = -sign
	}
	return d
}

func main() {
	a := mat{{1, 2, 3}, {4, 5, 6}}
	b := mat{{7, 8}, {9, 10}, {11, 12}}
	fmt.Println(mul(a, b))
	f := mat{{1, 1}, {1, 0}}
	r := mat{{1, 0}, {0, 1}}
	for i := 0; i < 90; i++ {
		r = mul(r, f)
	}
	fmt.Println(r)
	fmt.Println(det(mat{{2, -3, 1}, {2, 0, -1}, {1, 4, 5}}), det(mat{{1, 2}, {2, 4}}), det(mat{{6, 1, 1, 3}, {4, -2, 5, 1}, {2, 8, 7, 6}, {3, 1, 9, 7}}))
}
