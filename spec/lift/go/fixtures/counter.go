package counter

import (
	"errors"
	"fmt"
)

type Counter struct {
	n     int64
	limit int64
	full  bool
	marks [4]byte
}

var ErrFull = errors.New("Full")

const Step = 1

func (c *Counter) Bump() error {
	if c.full {
		return ErrFull
	}
	c.n = c.n + Step
	return nil
}

func (c *Counter) Reset() error {
	c.n = 0
	if c.full {
		return ErrFull
	}
	return nil
}

func Describe(v interface{}) string {
	if c, ok := v.(*Counter); ok {
		return fmt.Sprintf("%d", c.n)
	}
	for i := 0; i < 3; i++ {
		switch i {
		case 0:
			fmt.Println("zero")
		}
	}
	return ""
}
