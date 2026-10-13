# Introduction

This book teaches the RCE programming language from the beginning. It
assumes you can use a terminal and have written a little code in some
language. It does not assume you know any particular one.

## What RCE is

RCE is a small language for programs whose meaning is fixed by a written
specification. A program is a text. Its meaning is a *Form*: a state machine
made of states, events, a step function that takes a state and an event to
a new state and some outputs, and a log of the events that happened.

Three ideas run through the whole book:

1. **The same program gives the same answer everywhere.** Arithmetic, text
   and floating point have one defined meaning. There is no undefined
   behaviour, so a result does not depend on the machine.
2. **Every program finishes.** Loops have a limit, recursion has to make the
   input smaller, and iterators are counted. A step can never run forever.
3. **Mistakes are refused before the program runs.** The checker refuses a
   program that could use memory after giving it away, that forgets a case,
   or whose own tests do not hold. Each refusal names the rule it broke.

## Status

The specification is at **draft 0.6**. That draft is published for review
and can still change. This book teaches language version 0.5, the language
of sections 2 to 10; a module opts into version 0.6 (Appendix C of the
specification) with the first-line comment `# rce-version: 0.6`. What exists today:

- the specification, with its grammar checked by machine;
- a reference checker and interpreter, `rce`, which this book uses for every
  example;
- emitters that translate a checked program to Rust, Python or JavaScript.

What does not exist yet: a production compiler and a fast runtime. The
reference interpreter defines what programs mean. It is not built for speed.

## How the examples are checked

Every program in this book is a file in the `listings/` directory of the
book's repository. The chapters include those files directly, so what you
read is what was checked. The script `scripts/check-listings.sh` runs the
real `rce` command over all of them:

- a program shown as working must pass `rce test`;
- a program shown as refused must be refused, with the rule the text names;
- a terminal transcript must match what `rce` prints today.

If you find a place where the book and the checker disagree, that is a bug
in one of them. Please open an issue.

## Conventions

Program text looks like this:

```text
fn half(x: u64) -> u64 = x / 2
```

A line that starts with `$` is a command you type. The lines below it are
what `rce` printed. The book also shows the exit status, because `rce` uses
it to say whether something was refused: 0 means success, 1 means refused
or failed, 2 means the command itself was used wrongly.
