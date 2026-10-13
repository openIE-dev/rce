# Recursion and iterators

## Structural recursion

A function may call itself only on a *part* of a value it received:

```rce
{{#include ../listings/ch13/tree.rce}}
```

`tree_sum` calls itself on `l` and `r`, which are fields of the `Node` it
matched. Each call gets a smaller tree, so the recursion has to stop.
`Box[Tree]` is needed because a `Tree` that contains a `Tree` directly
would have no fixed size.

Recursion that does not shrink its input is refused, even when it would in
fact stop:

```rce
{{#include ../listings/ch13/refused/countdown.rce}}
```

```text
$ rce check countdown.rce
{{#include ../listings/ch13/refused/countdown.out}}
```

Write `countdown` as a loop instead. Two functions that call each other are
refused for the same reason. This rule is part of how RCE keeps its promise
that every step finishes.

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

- Recursion is allowed on parts of a matched value (rule `W-REC-1`).
- `iter T` functions with `yield`, consumed by `for` or `collect()`.
