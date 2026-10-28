// SPDX-License-Identifier: MIT OR Apache-2.0
// Struct types, value and pointer receivers, embedding and interfaces.
package main

import (
	"fmt"
	"math"
)

type shape interface {
	area() float64
	name() string
}

type rect struct{ w, h float64 }
type circle struct{ r float64 }

func (r rect) area() float64   { return r.w * r.h }
func (r rect) name() string    { return "rect" }
func (c circle) area() float64 { return math.Pi * c.r * c.r }
func (c circle) name() string  { return "circle" }

type counter struct{ n int }

func (c *counter) inc()      { c.n++ }
func (c counter) value() int { return c.n }

type named struct {
	counter
	label string
}

func main() {
	shapes := []shape{rect{3, 4}, circle{1}, rect{w: 2, h: 0.5}}
	total := 0.0
	for _, s := range shapes {
		fmt.Printf("%s %.4f\n", s.name(), s.area())
		total += s.area()
	}
	fmt.Printf("%.4f\n", total)
	c := counter{}
	c.inc()
	c.inc()
	fmt.Println(c.value(), c)
	n := named{label: "x"}
	n.inc()
	fmt.Println(n.value(), n.label, n)
	fmt.Printf("%v|%+v\n", rect{1, 2}, rect{1, 2})
	p := &rect{5, 6}
	p.w = 7
	fmt.Println(p.area(), *p)
}
