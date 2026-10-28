// SPDX-License-Identifier: MIT OR Apache-2.0
// Deferred calls run last in, first out; recover stops a panic and a
// deferred function can change named results.
package main

import (
	"errors"
	"fmt"
)

func order() {
	for i := 0; i < 3; i++ {
		defer fmt.Println("deferred", i)
	}
	fmt.Println("body")
}

func safeDiv(a, b int) (q int, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("recovered: %v", r)
			q = -1
		}
	}()
	return a / b, nil
}

func triple() (n int) {
	defer func() { n *= 3 }()
	n = 2
	return n + 1
}

func evalArgs() {
	x := 1
	defer fmt.Println("deferred sees", x)
	x = 2
	fmt.Println("x is", x)
}

func repanic() (msg string) {
	defer func() {
		r := recover()
		msg = fmt.Sprint("outer got ", r)
	}()
	defer func() {
		r := recover()
		panic(fmt.Sprint("again after ", r))
	}()
	panic("first")
}

func main() {
	order()
	fmt.Println(safeDiv(7, 2))
	fmt.Println(safeDiv(7, 0))
	fmt.Println(triple())
	evalArgs()
	fmt.Println(repanic())
	fmt.Println(recover())
	func() {
		defer func() {
			r := recover()
			err, ok := r.(error)
			fmt.Println(ok, err, errors.Is(err, errSentinel))
		}()
		panic(errSentinel)
	}()
}

var errSentinel = errors.New("sentinel")
