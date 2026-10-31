# Choices and loops

## `if`, `elif` and `else`

```rce
{{#include ../listings/ch03/grade.rce}}
```

`if` as a statement takes a block under each branch. There is also an `if`
*expression*, written on one line with `then` and `else`, as in `sign`. The
expression form must always have an `else`, because it has to produce a
value either way.

Comparisons cannot be chained: `0 < x < 10` requires change under the grammar.
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
{{#include ../listings/ch03/change/spin.rce}}
```

```text
$ rce check spin.rce
{{#include ../listings/ch03/change/spin.out}}
```

`spin(100)` needs 100 repetitions but allows 10, so the eleventh
repetition traps, the check cannot be evaluated, and the program requires
change.

A condition that can never change is a mistake of another kind. A literal
`true` or `false` as the condition of an `if`, `elif`, `while` or guard
requires change under `W-UNUSED-2`, because one of its branches is dead
code. A loop that runs until it is stopped is written `loop bound n` with a
`break`:

```rce
{{#include ../listings/ch03/change/forever.rce}}
```

```text
$ rce check forever.rce
{{#include ../listings/ch03/change/forever.out}}
```

## `var` and assignment

`let` names a value that does not change. `var` names one that does. You
change it with `=` or with a compound operator such as `+=`, `-=`, `*=`.
Assignment is a statement, never an expression, so `if (x = 1)` is a
syntax error.

A value stored in a `var` must be read before it is replaced. An
assignment whose value is overwritten later in the same block, with no
read in between, is an unused store and requires change under
`W-ASSIGN-1`:

```rce
{{#include ../listings/ch03/change/dead_store.rce}}
```

```text
$ rce check dead_store.rce
{{#include ../listings/ch03/change/dead_store.out}}
```

## What you learned

- `if` statements, `if ... then ... else ...` expressions, ranges with `in`.
- `for`, `while ... bound` and `loop bound`, and why each one is limited.
- `let` for fixed names, `var` for names that change.
- A constant condition (`W-UNUSED-2`) and a store that is overwritten
  before it is read (`W-ASSIGN-1`) require change.
