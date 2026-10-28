// SPDX-License-Identifier: MIT OR Apache-2.0
// fmt.Sprintf verbs, widths, flags and argument indexes.
package main

import "fmt"

type celsius float64

func (c celsius) String() string { return fmt.Sprintf("%.1fC", float64(c)) }

func main() {
	fmt.Println(fmt.Sprintf("%d|%5d|%-5d|%05d|%+d|%x|%X|%o|%O|%b", 42, 42, 42, 42, 42, 255, 255, 8, 8, 5))
	fmt.Println(fmt.Sprintf("%s|%10s|%-10s|%.3s|%q|%x|% x", "go", "go", "go", "golang", "a\"b", "hi", "hi"))
	fmt.Println(fmt.Sprintf("%c|%U|%q|%#U", 'G', 'G', 'G', '⌘'))
	fmt.Println(fmt.Sprintf("%t|%v|%p", true, nil, nil))
	fmt.Println(fmt.Sprintf("%[2]d %[1]d %d", 1, 2))
	fmt.Println(fmt.Sprintf("%d %s", "x", 5))
	fmt.Println(fmt.Sprintf("%d", 1, 2))
	fmt.Println(fmt.Sprintf("%d %d", 1))
	fmt.Println(fmt.Sprintf("%!", 1))
	fmt.Println(celsius(21.55), fmt.Sprintf("%v|%s|%.2f", celsius(-3), celsius(4), celsius(4)))
	fmt.Println(fmt.Sprintf("%6.2f%%", 99.5))
	fmt.Println(fmt.Sprintf("%*d|%-*d|", 4, 7, 4, 7))
}
