# Introduction

This book teaches the RCE programming language from the beginning. RCE
stands for Reality Code Engine: a language and toolchain made for the AI
coding era, in which every program has the one shape of the Form, so that
programs can be written, checked, changed and run by machines with no
language model in the loop. The shape is rigid at each commit of the
specification and grows inside the same text when a program needs a form
it lacks. It
assumes you can use a terminal and have written a little code in some
language. It does not assume you know any particular one.

## What RCE is

RCE is a language whose programs take their meaning from a written
specification, and there is one language: the specification at one commit,
extended inside the same text when a program needs a form it lacks, with
no per-module version selection. A program is a text. Its meaning is a *Form*: a state machine
made of states, events, a step function that takes a state and an event to
a new state and some outputs, and a log of the events that happened.

![RCE overview: source, checker, Form, log, emit targets, host](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rce-overview.svg)

*RCE at a glance: the checker reads a program into its Form, naming the place where a text does not yet have the Form's shape; every step goes into the log; emitters project it to other languages; a host runs it. More figures: [openIE-dev/rce docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

Three ideas run through the whole book:

1. **The same program gives the same answer everywhere.** Arithmetic, text
   and floating point have one defined meaning. There is no undefined
   behaviour, so a result does not depend on the machine.
2. **Every program finishes.** Loops have a limit, recursion has to make the
   input smaller, and iterators are counted. A step can never run forever.
3. **A program's meaning is settled before it runs.** Ownership, every
   case of a match and the program's own tests are part of its Form, so an
   accepted program never uses memory after giving it away, never leaves a
   case unhandled, and its tests hold. Where a text does not yet have that
   shape, the checker names the place and the rule.

## Status

**Reality Code Engine, draft 0.9.** The specification is published for
review and can still change. There is one language: the specification at
one commit. Drafts 0.5 to 0.9 are labels of past commits of that language,
not alternatives. A `# rce-version:` line is a release label. It is
tolerated, ignored and not hashed, and it is not a per-module language
selection. This book teaches sections 2 to 10 of that one text. What exists
today:

- the specification, with its grammar checked by machine;
- a reference checker and interpreter, `rce`, which this book uses for every
  example, with the whole standard library built in (`use enc`, `use
  data.events`: nothing to install);
- emitters that translate a checked program to Rust, Python, JavaScript
  and Scala, which agree with the reference; emitting every language by table, and lifts that bring programs in
  those languages into the Form, are the design of
  `spec/proposals/rce-form.md` (one Go pattern is lifted by a probe today).

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
