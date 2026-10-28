// SPDX-License-Identifier: MIT OR Apache-2.0
// Zero values: reading a nil map, ranging over nil slices and maps, a nil
// pointer in an interface is not a nil interface.
package main

import (
	"errors"
	"fmt"
)

type myErr struct{}

func (*myErr) Error() string { return "my error" }

func mayFail(fail bool) error {
	var p *myErr
	if fail {
		p = &myErr{}
	}
	return p
}

func main() {
	var m map[string]int
	fmt.Println(m["x"], len(m), m == nil)
	for range m {
		fmt.Println("never")
	}
	var s []int
	fmt.Println(s == nil, len(s), cap(s), append(s, 1))
	var p *int
	fmt.Println(p == nil)
	err := mayFail(false)
	fmt.Println(err == nil, err != nil)
	var e error
	fmt.Println(e == nil, errors.Is(e, nil))
	var i interface{}
	fmt.Println(i == nil, i)
	var f func()
	var ch chan int
	fmt.Println(f == nil, ch == nil, len(ch))
	var arr [3]string
	fmt.Printf("%q\n", arr)
	type T struct {
		A int
		B string
		C []int
		D map[int]int
		E *int
	}
	fmt.Printf("%+v\n", T{})
}
