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
3. **The book shows what is refused, too.** Programs that break a rule are
   shown with the exact message the checker prints, so you learn why a
   rule exists and how to fix the program.

## Read it

Build the book with [mdBook](https://rust-lang.github.io/mdBook/):

```sh
git clone https://github.com/openIE-dev/rce-book
cd rce-book
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
RCE=$PWD/rce/spec/checker/target/release/rce rce-book/scripts/check-listings.sh
```

The script prints `N passed, 0 failed` when every listing behaves as the
book says. Chapter 15 also runs the emitted Python and JavaScript, so
`python3` and `node` must be on your `PATH` for those two checks.

## Layout

| Path | Contents |
|---|---|
| `src/` | the chapters, in mdBook format; `SUMMARY.md` is the table of contents |
| `listings/chNN/` | the programs of chapter NN |
| `listings/chNN/refused/` | programs the checker must refuse; `NAME.expect` holds the text the refusal must contain |
| `listings/**/NAME.cmd` and `NAME.out` | a command shown in the book and the exact output it must produce |
| `scripts/check-listings.sh` | runs all of the above against a real `rce` |

The book follows draft 0.5 of the specification.

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
