# Recursion and iterators

## Structural recursion

A function may call itself when it passes a *part* of a value it
received:

```rce
{{#include ../listings/ch13/tree.rce}}
```

`tree_sum` calls itself on `l` and `r`, which are fields of the `Node` it
matched. Each call gets a smaller tree, so the recursion has to stop.
`Box[Tree]` is needed because a `Tree` that contains a `Tree` directly
would have no fixed size.

Two functions that call each other are admitted in the same way, when
the calls together shrink a value. `is_even` and `is_odd` each call the
other on the part of a number it matched, so every chain of calls is
finite:

```rce
{{#include ../listings/ch13/mutual.rce}}
```

The rule the checker applies (`W-REC-1`, specification 6.7) is more
general than "one parameter gets smaller": it looks for a *measure*, a
quantity such as the number of constructor nodes in some of the
parameters, that no call makes larger and some call makes smaller, and
repeats this in layers until no cycle of calls is left.

Recursion that does not shrink a value requires change, even when it
would in fact stop:

```rce
{{#include ../listings/ch13/change/countdown.rce}}
```

```text
$ rce check countdown.rce
{{#include ../listings/ch13/change/countdown.out}}
```

Write `countdown` as a loop, or give it a `bound`, as the next section
shows.

## Recursion on numbers: `bound` and `decreases`

A number has no constructor nodes to shrink, so recursion on integers needs
something else. `fn f(...) -> T bound n` allows at most n calls of `f` to
be active at once. The next one traps with `recursion bound`, so the
recursion ends, by its value or by the trap. A cycle of calls is admitted
when it passes through a function declared with `bound`.

`decreases m` states a measure written by the program: an expression, linear in the function's
unsigned integer parameters, that every recursive call makes strictly
smaller. The checker proves this from the arguments of each call (rule
`W-DEC-1`). Several measures, `decreases m1, m2`, are compared in order,
the first one that differs deciding. A measure has no effect while the
program runs: the function still needs its `bound`, and the bound still
holds. What `decreases` adds is a checked statement of why the recursion
ends.

A `while` loop can carry a measure too, as the first line of its body. The
checker works out the measure at the end of the body from the assignments
in it, and requires it to be smaller than at the start:

```rce
{{#include ../listings/ch13/bounded.rce}}
```

A measure that a call does not make smaller requires change:

```rce
{{#include ../listings/ch13/change/wrong_measure.rce}}
```

```text
$ rce check wrong_measure.rce
{{#include ../listings/ch13/change/wrong_measure.out}}
```

A `bound` that is too small is not a hang either. The call past the limit
traps, and a check that needs it cannot be evaluated:

```rce
{{#include ../listings/ch13/change/too_deep.rce}}
```

```text
$ rce check too_deep.rce
{{#include ../listings/ch13/change/too_deep.out}}
```

These rules are how RCE keeps its promise that every step finishes.
Measures on `for` and `loop`, and measures over data rather than unsigned
integers, are reserved, not implemented.

## Iterators

A function whose return type is `iter T` produces values one at a time
with `yield`:

```rce
{{#include ../listings/ch13/iterators.rce}}
```

A `for` loop takes the values as they are produced. `collect()` gathers
them into a list. Each `yield` counts against the loop limit from chapter
3, so an iterator cannot run forever either.

## What you learned

- Recursion, including two functions that call each other, is allowed on
  parts of a matched value (rule `W-REC-1`).
- Recursion on numbers is allowed through a function declared with `bound`;
  `decreases` states a measure that the checker proves (rule `W-DEC-1`).
- `iter T` functions with `yield`, consumed by `for` or `collect()`.
