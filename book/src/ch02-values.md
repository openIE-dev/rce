# Values, types and functions

## Constants, functions and checks

```rce
{{#include ../listings/ch02/values.rce}}
```

- `const` names a value that never changes. Its type is written after `:`.
- `fn to_seconds(minutes: i64) -> i64 = ...` is a function with one
  parameter of type `i64` (a 64-bit signed integer) that returns an `i64`.
  When the body is a single expression, write it after `=`.
- A longer body goes in an indented block. `let` names a value inside the
  block. `return` gives the result.
- `check` is a test. The checker evaluates it while checking the program,
  and the program is accepted only if every check is true.

Names at the top level of a file can be used before they are declared, so
the order of declarations does not matter.

## Integer types

RCE has the integer types `i8`, `i16`, `i32`, `i64` (signed), `u8`, `u16`,
`u32`, `u64` (unsigned) and `usize` (used for lengths and positions). Each
one stands for exactly the whole numbers in its range: a `u8` is a number
from 0 to 255.

```rce
{{#include ../listings/ch02/sizes.rce}}
```

A number written in the program is an `i64` unless the place it is used
needs a different integer type. In `small_sum(100, 27)` the parameters are
`u8`, so 100 and 27 are `u8`.

Different integer types never mix silently. To add a `u8` to an `i64`, turn
it into an `i64` with `as` first, as `widen` does.

## When a result does not fit

What is `200 + 100` as a `u8`? The true answer, 300, is not a `u8`. Many
languages quietly wrap it around to 44. RCE does not pick a value. The
addition *traps*: it stops with an error. Because the check below needs that
addition, the program requires change:

```rce
{{#include ../listings/ch02/change/overflow.rce}}
```

`rce check` says:

```text
$ rce check overflow.rce
{{#include ../listings/ch02/change/overflow.out}}
```

`W-INT-1` is the name of the rule in the specification: every check and
example must hold. When you want wrapping or clamping, ask for it by name,
as `sizes.rce` does: `wrapping_add` wraps around, `saturating_add` stops at
the largest value, and `checked_add` gives an *option* that is empty when
the result does not fit. (Options come back in chapter 7.)

Mixing types without `as` requires change too, under the type rule `W-TY-1`:

```rce
{{#include ../listings/ch02/change/mixed_types.rce}}
```

```text
$ rce check mixed_types.rce
{{#include ../listings/ch02/change/mixed_types.out}}
```

## Other basic types

- `bool`: `true` and `false`, combined with `and`, `or`, `not`.
- `f64` and `f32`: floating-point numbers. A literal such as `2.5` is an
  `f64`. The specification fixes how every operation rounds, so the same
  float code gives the same bits on every target.
- `text`: a string of Unicode text, such as `"hello"`. Chapter 8 covers it.
- `()`: the unit type, the result of a function that returns nothing.

## What you learned

- Functions, constants and `check` tests.
- Integer types have exact ranges, and going outside a range traps.
- Conversions between types are always written with `as`.
