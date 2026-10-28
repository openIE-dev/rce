// SPDX-License-Identifier: MIT OR Apache-2.0
// The longest common subsequence by dynamic programming, with one
// subsequence recovered from the table.
package main

import "fmt"

func lcs(a, b string) (int, string) {
	m, n := len(a), len(b)
	t := make([][]int, m+1)
	for i := range t {
		t[i] = make([]int, n+1)
	}
	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				t[i][j] = t[i-1][j-1] + 1
			} else if t[i-1][j] >= t[i][j-1] {
				t[i][j] = t[i-1][j]
			} else {
				t[i][j] = t[i][j-1]
			}
		}
	}
	out := []byte{}
	for i, j := m, n; i > 0 && j > 0; {
		if a[i-1] == b[j-1] {
			out = append([]byte{a[i-1]}, out...)
			i--
			j--
		} else if t[i-1][j] >= t[i][j-1] {
			i--
		} else {
			j--
		}
	}
	return t[m][n], string(out)
}

func main() {
	pairs := [][2]string{{"ABCBDAB", "BDCABA"}, {"AGGTAB", "GXTXAYB"}, {"", "abc"}, {"same", "same"}, {"kitten", "sitting"}}
	for _, p := range pairs {
		n, s := lcs(p[0], p[1])
		fmt.Printf("%q %q %d %q\n", p[0], p[1], n, s)
	}
}
