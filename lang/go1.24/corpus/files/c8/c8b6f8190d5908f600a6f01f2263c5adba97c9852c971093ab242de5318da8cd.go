// SPDX-License-Identifier: MIT OR Apache-2.0
// Errors as values: errors.New, fmt.Errorf with %w, errors.Is, errors.As,
// custom error types and errors.Join.
package main

import (
	"errors"
	"fmt"
	"strconv"
)

var errNotFound = errors.New("not found")

type parseError struct {
	Line int
	Msg  string
}

func (e *parseError) Error() string { return fmt.Sprintf("line %d: %s", e.Line, e.Msg) }

func find(key string) (int, error) {
	if key == "a" {
		return 1, nil
	}
	return 0, fmt.Errorf("find %q: %w", key, errNotFound)
}

func parse(line int, s string) error {
	if s == "" {
		return &parseError{line, "empty"}
	}
	return nil
}

func main() {
	v, err := find("a")
	fmt.Println(v, err)
	_, err = find("b")
	fmt.Println(err, errors.Is(err, errNotFound), errors.Unwrap(err) == errNotFound)
	err = fmt.Errorf("outer: %w", parse(7, ""))
	var pe *parseError
	if errors.As(err, &pe) {
		fmt.Println("as", pe.Line, pe.Msg)
	}
	fmt.Println(err)
	fmt.Println(parse(1, "ok") == nil)
	_, err = strconv.Atoi("12x")
	fmt.Println(err)
	j := errors.Join(errors.New("first"), nil, errors.New("second"))
	fmt.Println(j)
	fmt.Println(errors.Is(j, errNotFound))
}
