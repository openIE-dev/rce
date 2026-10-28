// SPDX-License-Identifier: MIT OR Apache-2.0
// Balanced brackets with a stack; reports the byte offset of the first
// error.
package main

import "fmt"

func check(s string) (bool, int) {
	var stack []byte
	pair := map[byte]byte{')': '(', ']': '[', '}': '{'}
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '(', '[', '{':
			stack = append(stack, c)
		case ')', ']', '}':
			if len(stack) == 0 || stack[len(stack)-1] != pair[c] {
				return false, i
			}
			stack = stack[:len(stack)-1]
		}
	}
	if len(stack) > 0 {
		return false, len(s)
	}
	return true, -1
}

func main() {
	for _, s := range []string{"", "()", "([]{})", "(]", "((())", "{[()()]}x", "}{", "a(b[c]d)e"} {
		ok, at := check(s)
		fmt.Printf("%-12q %v %d\n", s, ok, at)
	}
}
