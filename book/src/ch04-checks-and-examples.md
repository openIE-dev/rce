# Checks and examples

Tests in RCE are part of the program, not a separate file. A program whose
tests fail is not accepted at all.

## `check`

You have already seen `check`: a `bool` expression that must be true. The
checker evaluates every `check` while it checks the program.

## `intent` and `example`

An `intent` says in words what a piece of code is for, and holds the tests
that pin it down:

```rce
{{#include ../listings/ch04/counter.rce}}
```

The parts after the `intent` (`state`, `event`, `step`) are the subject of
the next chapter. For now, read the two examples:

- `example Inc, Inc, Dec -> 1` means: start from the initial state, apply
  the events `Inc`, `Inc`, `Dec`, and the counter must then be 1.
- `example Dec -> kept: Underflow` means: applying `Dec` to a fresh
  counter must keep the state, with the error `Underflow` as the reason.

`rce test` reports how many examples and checks each file has:

```text
$ rce test counter.rce
{{#include ../listings/ch04/counter_test.out}}
```

## A failing example means the program requires change

Change the expected value and the program requires change:

```rce
{{#include ../listings/ch04/change/wrong_example.rce}}
```

```text
$ rce check wrong_example.rce
{{#include ../listings/ch04/change/wrong_example.out}}
```

This is the admission rule `W-INT-1`: every `check` and every `example`
must hold for the program to be admitted. `rce test` and `rce check` give
the same verdict; `rce test` also counts the cases.

## What an example compares

When the program has one state with one field, the value after `->` is
compared with that field. Otherwise the part after `->` must be a `bool`
expression over the final state, such as `light == Yellow` or
`items == ["jam"] and adds == 2`. Chapter 10 has an example of the second
kind.

## What you learned

- `check` and `example` are tests that live in the program.
- A program whose tests fail requires change (rule `W-INT-1`).
- `rce test` runs the same checks as `rce check` and counts them.
