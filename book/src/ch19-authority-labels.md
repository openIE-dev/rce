# Authority, labels and typed output

This chapter covers four rules of Appendix F of the specification that
decide what a program is allowed to do with the outside world and with
sensitive data: who may call foreign code, where a secret may go, how text
for another language is escaped, and how code that handles a key avoids
depending on it. Each one is checked before the program runs. The world,
labels, `release` and `endorse` are erased before evaluation, so a program
that uses them computes what the same program without them computes.

## The `World` capability

A *foreign function* is one written in another language and declared with
`extern`. Calling one is authority: it can read files, the network or the
clock. In RCE that authority is a value, of the built-in type `World`, that
the host gives to `fn main` and to task functions (chapter 14). Every
`extern` takes `&World` as its first parameter, so a call can only be made
by code that holds the world and passes it on:

```rce
{{#include ../listings/ch19/world.rce}}
```

No other function may hold a `&World` (rule `W-WORLD-1`), so an ordinary
function can never reach a foreign function, however deep the call:

```rce
{{#include ../listings/ch19/change/world_in_helper.rce}}
```

```text
$ rce check world_in_helper.rce
{{#include ../listings/ch19/change/world_in_helper.out}}
```

A `&World` is passed whole and by name, and is never stored, returned,
sent on a channel or captured by an inline function. Time, randomness and
files have no primitive of their own: they arrive as events or as foreign
results that the host records in the log, so a replay sees the same
values.

## Labels: where a value may go

A *label* marks a value as sensitive. `label pii` declares one, and the type
`u64 @ {pii}` is a `u64` that carries it. A labeled value may flow only to a
place with the same label or more. An `emit`, a channel and an unlabeled
state field are public, so a labeled value that reaches one requires change
(rule `W-IFC-1`). The report names where the label came from:

```rce
{{#include ../listings/ch19/change/leak.rce}}
```

```text
$ rce check leak.rce
{{#include ../listings/ch19/change/leak.out}}
```

Copying a value is not the only way to reveal it. A branch taken because of
a labeled value tells anyone who sees its effect something about that value,
so a store or an `emit` made under such a condition carries the label too:

```rce
{{#include ../listings/ch19/change/implicit.rce}}
```

```text
$ rce check implicit.rce
{{#include ../listings/ch19/change/implicit.out}}
```

For the same reason, a labeled value may not decide how many times a loop
runs (`W-IFC-2`), and an operation that can trap may not have a labeled
operand that decides whether it traps (`W-IFC-4`): a kept step is
observable too.

`release e` removes a label. It is allowed only where the program holds the
capability the label names: `label pii by Auditor` says that releasing
`pii` needs a `Cap[Auditor]`, a capability value that arrives in an event
from the host. The condition that decides the release must also be
*trusted*: event payloads are untrusted unless declared `@ {}`.

```rce
{{#include ../listings/ch19/labels.rce}}
```

Without the capability, the release requires change (rule `W-IFC-5`):

```rce
{{#include ../listings/ch19/change/no_cap.rce}}
```

```text
$ rce check no_cap.rce
{{#include ../listings/ch19/change/no_cap.out}}
```

`endorse e`, with its own capability, marks an untrusted value as trusted
in the same way. `declassify` is reserved, not implemented: `release` is the
one way a label is removed.

## Typed output

Text that goes into another language (a web page, a URL, a shell command)
has to be escaped for that language, and forgetting to do so is a common
defect. RCE gives such text its own type. `Text[Html]` is text for HTML, and
it is joined only with text of the same context: a literal part written
with `Html.lit(...)`, a value escaped with `Html.encode(...)`, or a template
`Html.t"..."`, which escapes every field. The contexts are `Html`, `Url`,
`Sql`, `Json`, `ShellArg`, `Header` and `Prompt`; `Shell.cmd` builds a
command as a program name and a list of arguments, never as one command
line. `raw()` gives the finished text.

```rce
{{#include ../listings/ch19/typed.rce}}
```

Joining context text with plain text requires change (rule `W-CTX-1`), so
an unescaped value cannot slip in:

```rce
{{#include ../listings/ch19/change/plain_join.rce}}
```

```text
$ rce check plain_join.rce
{{#include ../listings/ch19/change/plain_join.out}}
```

## Constant-time functions

A program that compares a secret key with a value can reveal the key by
how long it takes, if it stops at the first difference. A `ct fn` is a
function in which a value labeled `secret` may not decide an `if`, a
`match`, an `and` or `or`, an index, a loop bound, a shift amount, a
divisor or an argument of a call to a function that is not itself a
`ct fn` (rule `W-CTFN-1`). `ct_choose(c, a, b)` evaluates both values and
chooses one, without a branch:

```rce
{{#include ../listings/ch19/ct.rce}}
```

```rce
{{#include ../listings/ch19/change/ct_branch.rce}}
```

```text
$ rce check ct_branch.rce
{{#include ../listings/ch19/change/ct_branch.out}}
```

The rule is about the decisions the program makes. It is not a claim about
the time the compiled code of any target takes, and the specification
states no such guarantee, so no emitter prints a module with a `ct fn`
(rule `W-CTFN-2`). The reference interpreter runs it as an ordinary
function.

## What this chapter leaves out

Appendix F also defines directory capabilities for foreign functions that
take a path (`Cap[DirRead]`, `Cap[DirWrite]` and `narrow`) and foreign
functions declared `extern "model"`, which must be called inside
`with budget`. The checker implements both. The reference host has no
adapter that answers a model call yet, so this book does not teach that
construct. Template values that are not rendered at once (`Template`)
are reserved, not implemented.

## What you learned

- Foreign functions take `&World`, which only `fn main` and task functions
  hold (`W-WORLD-1`).
- `T @ {L}` labels a value; a labeled value does not reach a public place,
  directly or through a branch (`W-IFC-1`), and `release` needs the
  capability the label names (`W-IFC-5`).
- `Text[C]` for each output context, with `lit`, `encode` and `C.t"..."`
  templates that escape every field (`W-CTX-1`).
- `ct fn` keeps secret values out of every decision (`W-CTFN-1`).
