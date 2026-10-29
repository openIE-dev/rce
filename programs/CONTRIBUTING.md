# Contributing to RCE examples

New programs, better explanations and fixes are all welcome.

## What makes a good example

- **It solves one recognisable problem** from its area, in a way a reader
  can follow in a few minutes.
- **It tests itself.** Add `example` lines for programs with state and
  events, and `check` lines for functions. The tests should show the
  normal case and at least one case where a step keeps the state.
- **It starts with a comment** saying what the program does and anything
  unusual about how it does it.
- **It is honest about limits.** If an area needs something RCE lacks, say
  so in the folder's README rather than working around it silently.

## Adding a program

1. Put it in the folder for its area, or propose a new area in an issue.
2. Add a line for it to the folder's README and to the table in the main
   README.
3. Run `RCE=/path/to/rce scripts/check.sh`. If you add a `.cmd` transcript,
   generate its `.out` with `scripts/check.sh --bless` and read the result
   before committing.

Programs follow the current draft of the specification. Do not use syntax
from unpublished drafts.

## License

By contributing, you agree that your contribution is licensed under
MIT OR Apache-2.0, as described in [COPYRIGHT](COPYRIGHT).
