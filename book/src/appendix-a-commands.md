# Appendix A: commands

The language commands:

| Command | What it does |
|---|---|
| `rce check FILE...` | parse, type check and admit; one verdict per file |
| `rce check write RECEIPT FILE...` | the same, and when every file is admitted, write the admission receipt of the program to RECEIPT |
| `rce check receipt RECEIPT FILE...` | admit the program again and say whether the receipt still holds, or whether the program, the rule set or the checker changed |
| `rce test FILE...` | the same, and count the examples and checks of each file |
| `rce run FILE "E1, E2"` | apply events and print the state after each |
| `rce run FILE` | run `fn main()` and print what it emits |
| `rce trace FILE [EVENTS]` | the reference trace of each example, or of the events, as JSON |
| `rce emit TARGET FILE` | `form`, `core`, `rust`, `python`, `js` or `scala` (chapter 15) |
| `rce fmt [check] FILE...` | write the rewrites of the altered rules in `auto` mode (W-UNUSED-1, W-DROP-1), print those in `observe` mode as diffs, then rewrite files in the canonical layout |
| `rce fmt nameless FILE` | print the program with every declared name replaced by its position |
| `rce equiv [fuel N] [json] A B` | whether two programs have the same behaviour on the explored events, with the shortest distinguishing sequence when not |
| `rce fingerprint [depth N] [fuel N] [json] FILE...` | a hash of what a program outputs on every event sequence up to N events long: programs `rce equiv` judges equivalent share it whatever their shape |
| `rce change [fuel N] OLD NEW` | classify a change: erratum, meaning change or interface change (chapter 18) |
| `rce dim [fuel N] [control] [json] FILE` | count the behaviours a program can be in and the redundancy of its state, per region (chapter 18) |
| `rce hash FILE...` | the definition hash of every named item, then the program hash (chapter 18) |
| `rce api-diff OLD NEW` | one row per change of the exports, `compatible` or `breaking` |
| `rce lib [MODULE]` | the built-in standard library, or the source of one module (chapter 12) |
| `rce manifest DIR` | write the hash-pinned manifest of a directory |
| `rce manifest-check MANIFEST` | verify a manifest (rule `W-PIN-1`) |
| `rce v09-status` | the Appendix F rules the checker implements and the ones it does not |
| `rce lsp` | language server on standard input and output |
| `rce version` | the commit `rce` was built from, the specification draft and the release label |
| `rce help` | every command with its words |

All files given to one command form one program.

The host commands (chapter 16) read the clock and the network by design and
record what they read in the log:

| Command | What it does |
|---|---|
| `rce host run [GRANT...] FILE D [EVENTS]` | admit events into the durable log D |
| `rce host serve [GRANT...] [http HOST:PORT] [presented FILE] FILE D` | run the host loop as a service; `presented FILE` reads a renderer's confirmations |
| `rce host replay [redact F,...] [frames] FILE D` | replay D with every adapter disconnected; `frames` re-derives every frame and checks `ui.rules` on them |
| `rce db D CMD [ARGS]` | RCEdb, the database whose storage is the log D |

A command line is the command, then its words, then its operands: every
word comes before the first file. A word such as `json` takes no value, a
word such as `fuel N` takes the next argument, and no argument starts with
`-`. A GRANT is one of `fs DIR`, `blob DIR`, `media DIR`, `wasm DIR`,
`net HOST:PORT`, `chan HOST:PORT`, `tls`, `peer HOST:PORT` (after `tls`,
`peer HOST:PORT KEY`), `pty` and `frames FILE` (the frame sink, HC-29). `rce help`, or `rce` with no arguments,
prints every command with its words.

Exit status: 0 success; 1 a program requires change, a test failed, a
run reached a step that kept the state, `fmt check` found a file to
change, or an admission receipt does not hold; 2 the command was used
wrongly; 3 `rce equiv` or `rce change` could not decide within its bound.

**What needs a person.** Every run of `rce check` and `rce fmt` ends with
one line on standard error that counts what it leaves for a person:

```
attention: 1 requires change, 0 recommended, 0 altered, 0 observed
```

A finding that requires change, and only such a finding, makes the status
1. A recommended finding is for a person to read; an alteration that
`rce fmt` wrote, and one it only recorded, need nobody until the record is
read.

**Modes of the altered rules.** Each rule that `rce fmt` can rewrite has a
mode in the specification's rule table, never in a program: `auto` (the
rewrite is written, and until then the text requires change), `observe`
(the rewrite is printed as a diff with its rule, nothing is written, and
the program is admitted) or `off`. W-UNUSED-1 and W-DROP-1 are `auto`; a
rule that becomes rewritable enters as `observe` and is promoted once its
diffs have been reviewed. A rule that requires change has no mode, so it
cannot be switched off.

**Admission receipts.** `rce check write counter.receipt counter.rce`
writes a receipt of why the program was admitted: its name-free hash, the
checker, every rule in force with its method and mode, each example, check
and law with its fuel and result, what the mutation gate measured, and the
law attestations. The bytes are the same on every run.
`rce check receipt counter.receipt counter.rce` admits the program again and
prints `receipt holds`, or that the program changed, that the rule set
changed (with the rules added, removed or changed, and whether the program
is still admitted under the new set), or that the checker changed. A
receipt is evidence of a past admission, never a reason to admit: the
program is always checked again.

Every language command is deterministic: the same input files give the
same output bytes. None reads the clock, the network or environment
variables. The full definitions are in section 10 of the specification.
