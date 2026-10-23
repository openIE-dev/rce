# Tasks and channels

RCE programs can run several *tasks* that talk over *channels*.

```rce
{{#include ../listings/ch14/channels.rce}}
```

- `chan[i64](8)` makes a channel that holds up to 8 values of type `i64`.
- `spawn producer(c, 10)` starts `producer` as a new task.
- `out <- i` sends `i` on the channel. `<- c` receives. When the channel is
  closed and empty, receiving gives `none`.

```text
$ rce run channels.rce
{{#include ../listings/ch14/channels_run.out}}
```

## Why the result is always the same

Tasks in RCE are scheduled in *ticks*. In each tick, every task that can
move takes one communication step, in an order fixed by the tasks' ids.
The ids are computed from the program itself, not from a clock or a
counter. So the whole run is a function of the program and its inputs, and
running it again gives the same result. When no task can move, the run
ends with the outcome `Deadlock` instead of hanging.

The specification also defines `select` (wait on several channels) and
`actor` (a state with its own mailbox). See section 6.8 of the
specification.

## Current limits

The reference interpreter runs tasks and channels as shown here. The
Rust, Python and JavaScript emitters of chapter 15 do not translate them
yet: in emitted code, channel operations trap.

## What you learned

- `spawn`, `chan[T](n)`, send with `<-` and receive with `<- c`.
- Scheduling is fixed by the program, so concurrent runs repeat exactly.
