# Every command, in one place (spec 10.19, CMD-1)

## Terms

A *command* is a path of words that `rce help` prints, such as
`rce host replay`. `COMMANDS` is the list of commands in
`spec/checker/src/cli.rs`; `rce help` prints it.

## State before

Section 10 of the specification, Appendix A of the book and the README
each listed some of the commands. Nine commands had no definition in
section 10 (`form-check`, `fingerprint`, `unform`, five `host` commands
and `help`), ten were absent from Appendix A and 23 from the README.
Nothing checked the three lists against `COMMANDS`.

## Decisions

1. Rule CMD-1: each command of `COMMANDS` is named in section 10, in
   Appendix A and in the README; a missing one requires change. Why: a
   reader of any one of the three must find every command `rce` runs.
2. The test `commands` (`spec/checker/tests/commands.rs`) reads
   `COMMANDS` and the three files. Why: a list kept by hand drifts; the
   list in the code is the one `rce help` prints.
3. The `host` commands are named in 10.19 and defined by Appendix E. Why:
   Appendix E is normative for the host contract, and a second definition
   could disagree with it.
4. `rce fingerprint` is named in 10.19 by rule BF-1 of
   `behaviour-fingerprint.md`, and `rce form-check` by FG1 to FG15 of
   `rce-form.md`. Why: both proposals hold the definitions; 10.19 does not
   restate them.

## Results

`cargo test --release --test commands`: 1 passed, over every command of
`COMMANDS`.

## Not done here

The README rows group several commands per row; Appendix A keeps one row
per command.
