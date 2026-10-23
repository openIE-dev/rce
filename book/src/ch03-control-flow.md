# Choices and loops

## `if`, `elif` and `else`

```rce
{{#include ../listings/ch03/grade.rce}}
```

`if` as a statement takes a block under each branch. There is also an `if`
*expression*, written on one line with `then` and `else`, as in `sign`. The
expression form must always have an `else`, because it has to produce a
value either way.

Comparisons cannot be chained: `0 < x < 10` is refused by the grammar.
Write `x in 0..10` instead.

## Three loops, each with a limit

```rce
{{#include ../listings/ch03/loops.rce}}
```

- `for i in 0..=n` runs over a range. `0..n` excludes `n`; `0..=n`
  includes it. A `for` loop can also run over a list, a map or a text.
- `while condition bound 10_000` repeats while the condition is true, at
  most 10,000 times. Underscores in numbers are only for reading.
- `loop bound 1_000` repeats until `break`, at most 1,000 times.

Why a limit? Because RCE promises that every program finishes. A `for` loop
always has a known length. A `while` or `loop` without `bound` gets the
default limit of 1,000,000 repetitions. A program can change that default
with `const LOOP_LIMIT: u64 = ...`.

Reaching the limit is not a hang. It is a trap with the message
`loop bound`:

```rce
{{#include ../listings/ch03/refused/spin.rce}}
```

```text
$ rce check spin.rce
{{#include ../listings/ch03/refused/spin.out}}
```

The loop condition is always `true`, so the eleventh repetition traps, the
check cannot be evaluated, and the program is refused.

## `var` and assignment

`let` names a value that does not change. `var` names one that does. You
change it with `=` or with a compound operator such as `+=`, `-=`, `*=`.
Assignment is a statement, never an expression, so `if (x = 1)` is a
syntax error.

## What you learned

- `if` statements, `if ... then ... else ...` expressions, ranges with `in`.
- `for`, `while ... bound` and `loop bound`, and why each one is limited.
- `let` for fixed names, `var` for names that change.
