# A ten-minute tour

This chapter runs one small program through the commands that make RCE
different from the languages you know: a program is a state machine, every
input has a declared outcome, and two programs can be compared by what they
do rather than by how they are written. Every command and every output below
is checked against `rce` each time the book is built. The chapters after this
one explain each idea slowly; here the aim is to see them all once.

## A counter that never goes below zero

```rce
{{#include ../listings/tour/counter.rce}}
```

The program has a **state** (`Counter`, one field `n`), two **events**
(`Inc` and `Dec`), one **error** (`Underflow`) and one **step** per event:
what the next state is when that event arrives. The `intent` block says what
the program is for, and its `example` lines say what it must do: after
`Inc, Inc, Dec` the counter holds 1, and a `Dec` at zero is a **kept step**,
which leaves the state as it was and names the error.

## Check it

`rce check` reads the program and either admits it or reports that it
*requires change*, naming the rule it breaks.

```text
$ rce check counter.rce
{{#include ../listings/tour/check.out}}
```

The program is admitted (`ok`). The second line is a *recommended* change,
not a requirement: the two example lines are the only obligations on the
steps, and a program that only memorised those two answers would meet them
too (rule W-INT-2). The checker says so instead of letting the examples look
like more than they are.

## Test it

`rce test` runs every `example` and `check` line:

```text
$ rce test counter.rce
{{#include ../listings/tour/test.out}}
```

## Run it

`rce run` applies a list of events and prints the state after each step:

```text
$ rce run counter.rce "Inc, Inc, Dec"
{{#include ../listings/tour/run.out}}
```

A `Dec` when the counter is at zero does not apply. The state is kept and the
error is named; nothing is half-done:

```text
$ rce run counter.rce "Inc, Dec, Dec"
{{#include ../listings/tour/kept.out}}
```

## The same program under other names

Here is the counter again, with every name changed:

```rce
{{#include ../listings/tour/tally.rce}}
```

`rce equiv` explores both programs over the same events and compares what
they do at every step, ignoring names:

```text
$ rce equiv --fuel 1000 counter.rce tally.rce
{{#include ../listings/tour/equiv_same.out}}
```

The counter can grow without end, so the exploration stops at its bound
(`--fuel 1000` states) and the verdict says "to the explored bound".
`rce fingerprint` turns a program's behaviour into one hash, so that equal
behaviour can be looked up rather than compared pair by pair:

```text
$ rce fingerprint counter.rce tally.rce
{{#include ../listings/tour/fingerprint.out}}
```

## A program that looks right and is not

This version passes its one example, but its `Dec` at zero quietly stays at
zero instead of saying that it did not apply:

```rce
{{#include ../listings/tour/sloppy.rce}}
```

`rce equiv` finds the shortest sequence of events on which the two programs
differ, and what each one does at its end:

```text
$ rce equiv --fuel 1000 counter.rce sloppy.rce
{{#include ../listings/tour/equiv_diff.out}}
```

## Where to go next

- [Getting started](ch01-getting-started.md) installs `rce` and writes a
  first program from scratch.
- [State, events and steps](ch05-state-events-steps.md) and
  [Errors and kept outcomes](ch06-errors.md) explain the counter in full.
- [The Form and the emitters](ch15-form-emitters.md) prints this program as
  Rust, Python, JavaScript or Scala with the same trace.
- [Behaviour, sameness and dimension](ch18-behaviour-dimension.md) covers
  `rce equiv`, `rce dim` and `rce change`, which classifies a change to a
  program by its exports and its behaviour.
