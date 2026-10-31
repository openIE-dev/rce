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

## Contracts

A `check` tests one call. A *contract* states what every call of a function
needs and gives back:

```rce
{{#include ../listings/ch04/contracts.rce}}
```

- `requires E` is a *precondition*: a condition on the arguments that the
  caller must meet.
- `ensures E` is a *postcondition*: a condition on the result, which the
  clause names `result`, that the function must meet.
- `ghost let before = balance` names a value that only the clauses read.
  Code that runs may not read a ghost name (`W-GHOST-1`).

The clauses come first in the body, before the first ordinary statement. A
contract is enforced on every call: the checker turns each clause into a
test that runs at that point, `requires` on entry and `ensures` at every
`return`. There is no build in which contracts are switched off. A clause
that does not hold traps, and the trap names which clause failed, by its
kind, its position in the function (counting from 0) and the hash of the
function:

```rce
{{#include ../listings/ch04/change/broken_contract.rce}}
```

```text
$ rce check broken_contract.rce
{{#include ../listings/ch04/change/broken_contract.out}}
```

The second clause, `amount <= balance`, is index 1. A trap inside a step
keeps the state, as chapter 6 explains. The checker does not yet prove a
contract before the program runs: it enforces it when the function is
called, and the emitters of chapter 15 do not yet translate a function with
a contract.

## What you learned

- `check` and `example` are tests that live in the program.
- A program whose tests fail requires change (rule `W-INT-1`).
- `rce test` runs the same checks as `rce check` and counts them.
- `requires` and `ensures` state a function's contract, enforced on every
  call; `ghost let` names values only the clauses read.
