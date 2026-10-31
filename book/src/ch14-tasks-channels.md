# Tasks and channels

RCE programs can run several *tasks* that talk over *channels*. A task is
a function running alongside the others; a channel is a queue of values
that one task sends into and another receives from.

```rce
{{#include ../listings/ch14/channels.rce}}
```

- `chan[i64](8)` makes a channel that holds up to 8 values of type `i64`.
  `chan i64` is the type of such a channel.
- `spawn producer(c, 10)` starts `producer` as a new task. A function that
  is spawned is a *task function*: it takes a channel among its parameters.
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

A send on a full channel does not drop the value: the sender waits until
there is room, in the order the sends arrived.

## Waiting on several channels: `select`

`select` waits on several channels at once and takes the first arm that is
ready:

```rce
{{#include ../listings/ch14/select.rce}}
```

`recv some(x) from a` is ready when `a` has a value; `recv none from a` is
ready when `a` is closed and empty. When several arms are ready, `select`
does not pick at random. Each `select` statement counts its executions,
and execution k scans the arms starting from arm k (counting round the
list), taking the first ready one. So an arm that stays ready is taken
within as many executions as there are arms, and no channel is starved:

```text
$ rce run select.rce
{{#include ../listings/ch14/select_run.out}}
```

The order depends only on the program, so it is the same on every run.

A `select` can also have an `after n` arm, which runs when no other arm has
been ready for n ticks, and a `default` arm, which runs at once when none
is ready. `select nondet` may take any ready arm; the arm it took is an
input, recorded in the log, so a replay takes the same one. It is allowed
only in `fn main` and in task functions (rule `W-ND-1`).

## Tasks that end together: `scope`

`spawn f(...) -> h` names the new task's *join handle* `h`. `h.join()`
waits for the task to end and gives its result, or the error `TaskFailed`
when the task failed. A `scope` block ends only after every task spawned
inside it has ended:

```rce
{{#include ../listings/ch14/scope.rce}}
```

```text
$ rce run scope.rce
{{#include ../listings/ch14/scope_run.out}}
```

A join handle cannot leave the scope that made it. Inside a module, a task
function may spawn only inside a `scope`, and `fn main` is a scope of its
own (rule `W-SCOPE-1`), so no task outlives the code that started it.

## Deadlines

`with deadline n` runs a block and gives it n ticks. If the block has not
finished by then, it stops and the `else` block runs instead:

```rce
{{#include ../listings/ch14/deadline.rce}}
```

```text
$ rce run deadline.rce
{{#include ../listings/ch14/deadline_run.out}}
```

Nobody sends on `quiet`, so the receive never completes and the deadline
passes. A tick is a step of the schedule, not a measure of time, so the
deadline passes at the same point on every run. Deadlines are allowed only
in `fn main` and in task functions (rule `W-DL-1`).

## Budgets

`with budget n` gives a block a cost budget of n units. Every operation
charges its cost from a fixed table (specification C.6), and the operation
that would take the total past n traps with `budget`. With an `else`, the
trap becomes a value:

```rce
{{#include ../listings/ch14/budget.rce}}
```

When the budget runs out, every change the block made is undone, and then
`else` runs. `as e` names the record `Exhausted`, whose `limit` is n and
whose `used` is what the block had spent. Because the block is undone, it
may not do anything that cannot be undone: an `emit`, a channel operation,
a spawn or a `select` inside a budget with `else` requires change (rule
`W-BUD-1`). The trap falls on the same operation on every target, so
whether `else` runs is a function of the program and its inputs.

## Parallel blocks: `par` and `for ... par`

`par` runs its branches, each a named block, and gives the same result as
running them in the order they are written. `for x in e par` does the same
for the iterations of a loop:

```rce
{{#include ../listings/ch14/par.rce}}
```

The rule that makes this safe is that branches do not touch each other's
data. Two branches where one writes what the other reads or writes require
change (rule `W-PAR-1`):

```rce
{{#include ../listings/ch14/change/race.rce}}
```

```text
$ rce check race.rce
{{#include ../listings/ch14/change/race.out}}
```

A branch may not emit, use a channel or spawn (rule `W-PAR-2`). The body of
a `for ... par` may write a place only as a *reduction*, such as `s += x`
on a variable it never reads, or as slot `[i]` of a range loop over `i`
(rule `W-PAR-3`). A reduction is added up in a fixed tree, the same on
every machine, so even a floating-point sum gives the same bits
everywhere.

## Actors

An `actor` is a state with its own mailbox. Its `on` handlers are steps
(chapter 5) for the messages it receives:

```rce
{{#include ../listings/ch14/actor.rce}}
```

`Tally()` creates one, and `t <- Add(k: 5)` puts a message in its mailbox.
The actor handles at most one message per tick, in mailbox order. A reply
goes back on a channel that the message carries.

```text
$ rce run actor.rce
{{#include ../listings/ch14/actor_run.out}}
```

## Failing after an effect

Chapter 6 showed that a failed call restores its `&mut` arguments. A value
already sent on a channel cannot be taken back, so a function that can
fail after a send, a spawn, a receive, a `select` or a foreign call
requires change (rule `W-TX-1`):

```rce
{{#include ../listings/ch14/change/send_then_fail.rce}}
```

```text
$ rce check send_then_fail.rce
{{#include ../listings/ch14/change/send_then_fail.out}}
```

Return the failure as a value instead, or check before the first effect.

## Priorities

`spawn f(...) priority p` and `event E priority p` give a task or an event
a priority, an integer. When several are waiting to be admitted, the one
with the highest priority goes first.

## Emitted programs

The emitters of chapter 15 translate tasks, channels, `select`, actors and
`par`. An emitted program runs `fn main` when it is started with the
argument `--run`, and prints the same output as `rce run`, followed by the
run's outcome:

```text
$ rce emit python channels.rce > channels.py && python3 channels.py --run
{{#include ../listings/ch14/channels_python.out}}
```

## What this chapter leaves out

Appendix C of the specification also defines workgroups (`with group`
and `barrier`), session types (`protocol`), `atomic` blocks across actors,
supervisors and nodes, LVars and replicated state, and `interrupt` events.
The checker implements them, and the specification's examples in
`spec/conc06b/` show each one. This book does not teach them: the RCE
project's proposal ledger (`spec/proposals/LEDGER.md`) records an open
proposal to move actors, workgroups and supervisors out of the syntax into
library Forms built on `spawn`, `chan`, `select` and `par`, and sessions
and LVars stay proposals until their theorems are proved of the checker.

## What you learned

- `spawn`, `chan[T](n)`, send with `<-` and receive with `<- c`.
- Scheduling is fixed by the program, so concurrent runs repeat exactly.
- `select` with a fair rotation, `after` and `default`; `scope` and join
  handles; `with deadline` and `with budget`.
- `par` and `for ... par` give the source-order result, because branches
  do not share what they write.
- Actors are states with mailboxes.
- A function cannot fail after an effect it cannot undo (`W-TX-1`).
