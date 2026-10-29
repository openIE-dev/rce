# State, events and steps

This chapter is the centre of the language. Most RCE programs are built
from four kinds of declaration:

- a `state` holds data that lasts between events;
- an `event` is something that happens, with an optional payload;
- a `step` says how a state changes when an event arrives;
- an `error` is a reason to refuse an event.

## A door

```rce
{{#include ../listings/ch05/door.rce}}
```

- `state Door` has two fields, each with a starting value.
- Four events have no payload.
- `step Door on Open if locked` is a step with a *guard*. When `Open`
  arrives and `locked` is true, this step runs and `fail Locked` refuses
  the event.
- When the guard is false, the next step for `Open` in the file is used,
  and it sets `open = true`.

For each event, the checker picks the first step, in source order, whose
event matches and whose guard is true. If no step matches, the state stays
as it was.

Every declared event must have at least one step. Otherwise the step
function would have no answer for that event, and the program is refused:

```rce
{{#include ../listings/ch05/change/unhandled_event.rce}}
```

```text
$ rce check unhandled_event.rce
{{#include ../listings/ch05/change/unhandled_event.out}}
```

Inside a step, the state's fields are variables you can read and change.
The changes become the new state only if the step finishes without being
refused.

## Running events

`rce run FILE EVENTS` applies events one by one and prints the state after
each:

```text
$ rce run door.rce "Open, Close, Lock, Unlock, Open"
{{#include ../listings/ch05/door_ok.out}}
```

A refused event leaves the state unchanged, and nothing after it is
applied:

```text
$ rce run door.rce "Lock, Open, Close"
{{#include ../listings/ch05/door_kept.out}}
```

The exit status is 1 because an event was refused.

## The counter again

In the counter from chapter 4, both steps use the short arrow form:

```text
step Counter on Inc -> n + 1
```

When a state has exactly one field, `-> expression` sets that field to the
value of the expression. `try n.checked_sub(1) else Underflow` subtracts
one if the result fits in a `u64`, and otherwise refuses the event with
`Underflow`.

```text
$ rce run counter.rce "Inc, Inc, Dec, Dec, Dec"
{{#include ../listings/ch05/counter_run.out}}
```

## Why build programs this way

A state, a set of events and a step function together form a state
machine, which the specification calls the program's *Form*. Because the
step function is a plain function of the old state and the event, a list
of events always leads to the same states. You can replay the list and get
exactly the same run. Chapter 15 shows the Form itself.

![Where state, events and steps sit: the Form P = (S, E, delta, D, lambda)](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rce-overview.svg)

*The states, events and steps of this chapter are the S, E and δ of the Form; the log of events is D. More figures: [openIE-dev/rce docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

## What you learned

- `state`, `event`, `step` and `error`, and how a step is chosen.
- Guards with `if` on a step.
- `rce run FILE "Event, Event"` to watch a state change.
