# Structs, kinds and matching

## Kinds

A `kind` is a type whose value is one of several *variants*. Each variant
can carry its own fields.

```rce
{{#include ../listings/ch07/shapes.rce}}
```

`match` looks at a value and runs the arm whose pattern fits.
`Circle(r: r)` matches a circle and names its radius `r`.

The checker requires a `match` to cover every variant. Forgetting one is
refused, so adding a variant later shows you every place that must handle
it:

```rce
{{#include ../listings/ch07/fit/missing_case.rce}}
```

```text
$ rce check missing_case.rce
{{#include ../listings/ch07/fit/missing_case.out}}
```

A `_` pattern matches anything and can serve as the last arm.

## Structs and `impl`

A `struct` groups named fields. `Point(x: 0, y: 0)` builds one; fields are
always passed by name. `impl Point` adds functions that belong to the type.
`Point.origin()` calls one that takes no `self`; `p.moved(4)` calls one
that does.

## Options

`i64?` is an *option*: either `some(value)` or `none`. RCE has no null, so
an empty result is always visible in the type. `first_even` returns `none`
when it finds no even number, and `or_zero` uses `match` to handle both
cases.

## Borrowed arguments

`area(s: &Shape)` takes a *borrow* of a shape: it can read the shape but
does not take it over. You pass a borrow with `&`, as in `area(&BOX)`.
Borrows are what chapter 9 is about. One rule shows up already: you can
only borrow a *place*, such as a variable or a constant, not a value that
was just built. That is why the checks use the named constants `BOX` and
`NOTHING`.

## What you learned

- `kind` with variants, `struct` with fields, `impl` for functions on a type.
- `match` must cover every case (rule `W-MATCH-1`).
- `T?` options with `some` and `none` instead of null.
