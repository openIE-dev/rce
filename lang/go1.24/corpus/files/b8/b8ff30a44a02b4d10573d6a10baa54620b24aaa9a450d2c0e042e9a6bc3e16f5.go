// SPDX-License-Identifier: MIT OR Apache-2.0
// Functions of the strings package on ASCII text.
package main

import (
	"fmt"
	"strings"
)

func main() {
	s := "  The quick brown fox  "
	fmt.Printf("[%s]\n", strings.TrimSpace(s))
	fmt.Println(strings.ToUpper(s), strings.ToLower(s))
	fmt.Printf("%q\n", strings.Fields(s))
	fmt.Printf("%q\n", strings.Split("a,b,,c", ","))
	fmt.Printf("%q\n", strings.SplitN("a,b,c,d", ",", 2))
	fmt.Println(strings.Join([]string{"x", "y", "z"}, "-"))
	fmt.Println(strings.Replace("aaaa", "a", "b", 2), strings.ReplaceAll("banana", "an", "AN"))
	fmt.Println(strings.Contains("seafood", "foo"), strings.HasPrefix("golang", "go"), strings.HasSuffix("golang", "ng"))
	fmt.Println(strings.Index("chicken", "ken"), strings.Index("chicken", "dmr"), strings.LastIndex("go gopher", "go"))
	fmt.Println(strings.Repeat("ab", 3), strings.Count("cheese", "e"), strings.Count("five", ""))
	fmt.Println(strings.Trim("xxhixx", "x"), strings.TrimLeft("xxhixx", "x"), strings.TrimPrefix("xxhixx", "x"))
	fmt.Println(strings.Title("hello world"), strings.EqualFold("Go", "GO"))
}
