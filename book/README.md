# The RCE Book

This is a guide to the RCE programming language, written for people who
have never used it. It starts from a single line that prints a greeting
and builds up, one idea per chapter, to states and events, errors,
ownership, modules and concurrency. Every program in the book is a real
file that is checked by the RCE reference checker, so the examples work as
printed.

In plain words:

1. **You can start with no background in RCE.** Each chapter introduces a
   few ideas and ends with a short summary of what you learned.
2. **Every example is tested.** The programs live in `listings/`, the
   chapters include them directly, and a script checks them all with the
   real `rce` command.
3. **The book shows what requires change, too.** Programs that break a rule
   are shown with the exact message the checker prints, so you learn why a
   rule exists and how to fix the program.

## At a glance

![RCE overview: source, checker, Form, log, emit targets, host](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rce-overview.svg)

*What the book teaches, in one picture: write a program, the checker accepts it or names the broken rule, the accepted program is a Form (states, events, steps, log), and `rce emit` turns it into other languages. More figures: [openIE-dev/rce docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

## Read it

Build the book with [mdBook](https://rust-lang.github.io/mdBook/):

```sh
git clone https://github.com/openIE-dev/rce
cd rce/book
mdbook serve --open
```

Or read the Markdown in `src/` directly, starting with
`src/introduction.md`.

## Check the examples

You need the `rce` command from the
[rce repository](https://github.com/openIE-dev/rce):

```sh
git clone https://github.com/openIE-dev/rce
(cd rce/spec/checker && cargo build --release)
RCE=$PWD/spec/checker/target/release/rce book/scripts/check-listings.sh
```

The script prints `N passed, 0 failed` when every listing behaves as the
book says. Chapter 15 also runs the emitted Python and JavaScript, so
`python3` and `node` must be on your `PATH` for those two checks.

## Layout

| Path | Contents |
|---|---|
| `src/` | the chapters, in mdBook format; `SUMMARY.md` is the table of contents |
| `listings/chNN/` | the programs of chapter NN |
| `listings/chNN/change/` | programs the checker must report as requiring change; `NAME.expect` holds the text the report must contain |
| `listings/**/NAME.cmd` and `NAME.out` | a command shown in the book and the exact output it must produce |
| `scripts/check-listings.sh` | runs all of the above against a real `rce` |

Two numbers are kept apart here: the specification draft (a document) and
the language version (what a module declares). The book teaches language
version 0.5, which is sections 2 to 10 of the specification. Every listing
is a version 0.5 module. The listings were last checked on 17 October 2026
against public `openIE-dev/rce` main (`e416fd0`, specification draft 0.9),
whose checker accepts language versions 0.5 to 0.9. The version 0.6 to 0.9
extensions (Appendix C and Appendix F of the specification) are not covered by the book yet.

| Chapters | Language version | Checked against |
|---|---|---|
| 1 to 15 (`listings/ch01` to `listings/ch15`) | 0.5 | `rce` main `e416fd0`, `scripts/check-listings.sh`: 54 of 54 pass (the expected output of `listings/ch15/form.out` was re-blessed: `rce emit form` now also prints a `v09` flag) |
| 16 (host tools), 17 (media calls and end-to-end encryption) | no declared version | not checked: `src/ch16-host-tools.md` and `src/ch17-media-calls-e2ee.md` have no `listings/` directory, so the script does not run them |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright (c) 2026 Open Interface Engineering, Inc.

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
you intentionally submit for inclusion in this work shall be dual licensed
as above, without any additional terms or conditions.
