# Contributing to the RCE Book

Corrections, clearer explanations and new examples are all welcome. If
something confused you while reading, that is worth an issue on its own.

## The rules for examples

1. **Every program is a file in `listings/`.** Chapters include it with
   `{{#include ../listings/chNN/NAME.rce}}`. Do not paste program text into
   a chapter by hand.
2. **Every program is checked.** A program meant to work must pass
   `rce test`. Give it `check` lines or `example` lines so that the test
   says something.
3. **A refused program says why.** Put it in `listings/chNN/fit/`, with
   a `NAME.expect` file holding the rule name or message the refusal must
   contain.
4. **Every terminal transcript is real.** Put the command's arguments in
   `NAME.cmd`. Generate `NAME.out` with
   `scripts/check-listings.sh --bless`, then read the diff before you
   commit it.

Run the check before opening a pull request:

```sh
RCE=/path/to/rce scripts/check-listings.sh
```

## Style

- Write for a reader who is new to RCE. Explain a term the first time it
  appears.
- Use short sentences and plain words.
- Say what the language and the tools do today. When the specification
  allows something the checker does not support yet, say so, and keep a
  listing that shows the current behaviour (chapter 12 has an example).
- Each chapter ends with a short "What you learned" list.

## License

By contributing, you agree that your contribution is licensed under
MIT OR Apache-2.0, as described in [COPYRIGHT](COPYRIGHT).
