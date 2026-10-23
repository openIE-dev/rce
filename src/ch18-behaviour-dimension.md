# Behaviour, sameness and dimension

Two programs can look different and be the same. This chapter says what
"the same" means in RCE, how the checker can tell, and how to measure how
much of a program's state is doing work.

## The behaviour of a program

Run any sequence of events through a program and write down what it emits
at each step. That table, for every possible sequence, is the program's
*behaviour*. Nothing else about the program is observable: not the names of
its fields, not the order its steps are written in, not what it keeps in
state and never reads. The specification (section 6.10) defines behaviour
exactly this way, and defines two programs to be *equivalent* when their
behaviours are equal.

That is the only sense of "same" the Form needs. It is also why the
emitters can be checked: a Rust program and a Python program emitted from
one Form are correct when they have the same behaviour, and the reference
trace of chapter 15 is a prefix of that table.

## How the checker decides it

You cannot list every event sequence; there are infinitely many. The
specification gives two ways round that.

The first is a *bisimulation*: a relation between the states of two
programs such that related states emit the same thing on every event and
land in related states. If the starting states are related, the behaviours
are equal (T58). To show two programs are the same, exhibit the relation;
to show they differ, exhibit one event sequence where the outputs part.

The second is a *distance*. Two behaviours that agree on every sequence up
to length k and differ at k + 1 are at distance 2^-(k+1). Trace tests like
those of chapter 15 therefore prove closeness, not equality: `rce trace`
agreeing on 40 events shows the two programs are within 2^-41 of each
other. That is why a trace never enters a hash in RCE; a hash names one
thing, and a trace names a neighbourhood.

## Dimension: how many behaviours can a program be in?

Every RCE program has finite state, so the set of states it can reach is
finite, and the number of *different behaviours* among those states is a
number. The specification calls it the program's dimension, dim(P). It is
the count of states a minimal machine with the same behaviour would need.
State beyond that is *redundancy*: bits that no future output depends on.

`rce dim` measures it. It explores the states reachable from the start
through every event, with payloads drawn from each type's edge values, then
groups states that behave alike.

Here is a counter mod 4 that also remembers whether the last event was an
increment. Nothing reads that flag:

```
{{#include ../listings/ch18/counter_flag.rce}}
```

```text
$ rce dim counter_flag.rce
{{#include ../listings/ch18/counter_flag_dim.out}}
```

Eight states, four behaviours, one bit of redundancy, and the report says
which region carries it. The flag can go.

Redundancy is not always a spare field. This program keeps a count mod 6
but only ever emits its parity:

```
{{#include ../listings/ch18/phase.rce}}
```

```text
$ rce dim phase.rce
{{#include ../listings/ch18/phase_dim.out}}
```

Six states, two behaviours: the state has a period of six, the behaviour
has a period of two, and log2(6) − log2(2) = 1.58 bits are carrying
nothing observable.

And here is the door from chapter 10, where every reachable state answers
some future event differently:

```
{{#include ../listings/ch18/door.rce}}
```

```text
$ rce dim door.rce
{{#include ../listings/ch18/door_dim.out}}
```

Three states, not four: a door cannot be open and locked at once, because
`Lock` is refused while open and `Open` is refused while locked. Three
behaviours, no redundancy. This program's state is exactly its behaviour.

## State nobody can see

The traffic light of chapter 5 has three states and, by this measure, one
behaviour: it never emits and never refuses, so no sequence of events gets
a different answer from any state. Its examples check `light == Yellow`,
but an example reads the state from inside; the Form's behaviour is what
comes out. `rce dim` on that program reports 3 states, 1 behaviour and 1.58
bits of redundancy, and it is right: as written, the light is a machine
whose colour nothing can observe. Add `emit light` to its step and the
three states become three behaviours. When the first run of `rce dim` over
the specification's own examples was made, most of its state machines
turned out to be of this kind.

## The control view

`rce dim --control` asks a coarser question: not what the program emits,
but which steps fire. Two states are control-equivalent when every event
sequence makes the same steps fire, refuse or emit the same kinds of thing
from both. The count is never larger than the behaviour count, and its
canonical form is hashed as `shape_ctl`, a key that does not change when
you rename anything:

```text
$ rce dim --control door.rce
{{#include ../listings/ch18/door_dim_control.out}}
```

For the door the two counts agree. For the mod-4 counter they do not: every
event fires the same step with the same kind of output, so it has one
control state and four behaviours. The gap between the two numbers is
how much of the behaviour lives in data rather than in control.

## When the count is a lower bound

A program whose state can grow without limit, like the plain counter of
chapter 4, has more reachable states than `rce dim` will visit. It stops at
`--fuel` states (100,000 by default), says `bound reached: yes`, and reports
the behaviours it found as a lower bound. Exploration is deterministic:
the same program and the same fuel give the same report, byte for byte.

## What you learned

- A program's behaviour is what it emits on every event sequence; two
  programs are the same when their behaviours are equal.
- A bisimulation proves sameness; one distinguishing sequence proves
  difference; a trace proves closeness to within 2^-(k+1).
- `rce dim` counts the behaviours a program can be in and reports the
  redundancy of its state per region, in bits; `--control` counts the
  control states and hashes their canonical form as `shape_ctl`.
