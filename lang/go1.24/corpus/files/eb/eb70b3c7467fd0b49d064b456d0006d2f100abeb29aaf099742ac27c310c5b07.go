// SPDX-License-Identifier: MIT OR Apache-2.0
// %v, %+v, %#v and %T on structs, slices, maps, pointers to structs and
// values with String and Error methods.
package main

import (
	"errors"
	"fmt"
)

type inner struct {
	N int
	S []string
}

type outer struct {
	Name  string
	In    inner
	Ptr   *inner
	Tags  map[string]int
	Any   interface{}
	small uint8
}

type color int

func (c color) String() string { return [...]string{"red", "green"}[c] }

func main() {
	o := outer{Name: "x", In: inner{1, []string{"a"}}, Tags: map[string]int{"k": 1, "a": 2}, Any: 2.5, small: 7}
	fmt.Printf("%v\n%+v\n", o, o)
	fmt.Printf("%#v\n", inner{2, nil})
	fmt.Printf("%#v %#v %#v %#v\n", []int{1}, map[string]bool{"t": true}, "q", 3.0)
	fmt.Printf("%T %T %T %T %T\n", o, &o, o.Tags, o.Any, color(1))
	fmt.Println(&inner{3, []string{"p", "q"}}, []*int{nil})
	fmt.Println(color(0), []color{0, 1}, map[color]int{1: 5})
	fmt.Printf("%d %s %v\n", color(1), color(1), errors.New("e"))
	fmt.Printf("%v %d\n", []byte("hi"), []byte("hi"))
	fmt.Printf("%s %q %x\n", []byte("hi"), []string{"a", "b"}, []byte("hi"))
	fmt.Printf("%5v|%-5v|%05d\n", []int{1, 2}, true, []int{3, 4})
}
