# The Form and the emitters

## The Form

Every RCE program means a *Form*: its states, events, step function, and
the log of events it was tested with. `rce emit form` prints the Form of
the counter from chapter 4 as one line of JSON:

![The Form and the emit targets](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rce-overview.svg)

*The Form and the targets `rce emit` writes. More figures: [openIE-dev/rce docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

```text
$ rce emit form counter.rce
{{#include ../listings/ch15/form.out}}
```

You can see the state `Counter` with its starting value, the events, the
error `Underflow`, the two steps, and a log for each `example`, with the
state after every event. After the logs, the same line carries the program
`source`, its parsed `terms`, and `names`, the concept record of
specification 6.6: the names as written and the intent strings, keyed by
declaration index, printed after the hashed fields and never read by a
step. Each log entry also carries `seq`, its position,
`state_hash`, the SHA-256 of the state after the event, `prev_hash`, the
`hash` of the entry before it (64 zeros for the first), and `hash`, which
covers the other four. What is hashed is not the readable `state` and
`event` you see but their *positional forms*, repeated under `hashed`:
the regions and fields as lists in declaration order, and the event by
its index. So renaming everything in a program changes no hash, which is
what theorem T7 promises. The entries form a chain: changing any past entry
changes every later `hash`. The JSON is ordered in a fixed way, so the same
program always gives the same bytes. The event `Tick` marked `builtin` is
part of every Form: it is the program's logical clock. No RCE construct
reads a wall clock.

## Emitting a target language

Every language is both an output and an input of the Form. A target is a
row of the emit registry (`spec/proposals/emit-targets.md`): one emitter
over the Form IR and one table per language, each passing the same gate,
the reference trace of every example program byte for byte. This chapter
uses the targets the reference checker emits today.

`rce emit rust`, `rce emit python` and `rce emit js` translate a checked
program into one self-contained file. Running that file prints the
program's *observations*: the result of each check, and the state after
each event of each example.

![Emit and IDE test results](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/status.svg)

*Test and differential results for the emitters and the IDE, October 2026. More figures: [openIE-dev/rce docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

```text
$ rce emit python counter.rce | python3 -
{{#include ../listings/ch15/counter_python.out}}
```

```text
$ rce emit js counter.rce > counter.mjs && node counter.mjs
{{#include ../listings/ch15/counter_js.out}}
```

The two outputs are the same line for line. The checker's own test suite
holds the emitters to this: it emits every example program of the
specification in all three languages, runs each one, and requires its
output to equal the reference interpreter's.

## What you learned

- `rce emit form` shows the Form behind a program.
- The emitters produce Rust, Python, JavaScript and Scala that print the
  same observations as the reference interpreter; further targets are
  registry rows, not new emitters.
