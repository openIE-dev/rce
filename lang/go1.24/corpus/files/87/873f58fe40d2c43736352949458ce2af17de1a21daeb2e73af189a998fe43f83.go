// SPDX-License-Identifier: MIT OR Apache-2.0
// Method values and method expressions, interface embedding and type
// assertions with the comma-ok form.
package main

import (
	"fmt"
	"strings"
)

type greeter struct{ name string }

func (g greeter) hello(p string) string { return p + ", " + g.name }
func (g *greeter) rename(n string)      { g.name = n }

type stringer interface{ String() string }
type namedStringer interface {
	stringer
	Name() string
}

type thing struct{ id int }

func (t thing) String() string { return fmt.Sprintf("thing#%d", t.id) }
func (t thing) Name() string   { return "thing" }

func main() {
	g := greeter{"ann"}
	h := g.hello
	g.rename("bob")
	fmt.Println(h("hi"), g.hello("hey"))
	f := greeter.hello
	fmt.Println(f(greeter{"cid"}, "yo"))
	r := (*greeter).rename
	r(&g, "dee")
	fmt.Println(g.name)
	var s stringer = thing{7}
	if ns, ok := s.(namedStringer); ok {
		fmt.Println(ns.Name(), ns)
	}
	_, ok := s.(interface{ Missing() })
	fmt.Println(ok)
	var x interface{} = "text"
	n, ok := x.(int)
	fmt.Println(n, ok)
	up := strings.ToUpper
	fmt.Println(up("method"), strings.Map(func(r rune) rune {
		if r == 'a' {
			return -1
		}
		return r + 1
	}, "banana"))
}
