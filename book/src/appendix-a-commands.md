# Appendix A: commands

The language commands:

| Command | What it does |
|---|---|
| `rce check FILE...` | parse, type check and admit; one verdict per file |
| `rce test FILE...` | the same, and count the examples and checks of each file |
| `rce run FILE "E1, E2"` | apply events and print the state after each |
| `rce run FILE` | run `fn main()` and print what it emits |
| `rce trace FILE [EVENTS]` | the reference trace of each example, or of the events, as JSON |
| `rce emit TARGET FILE` | `form`, `core`, `rust`, `python`, `js` or `scala` (chapter 15) |
| `rce fmt [check] FILE...` | write the rewrites of the altered rules (W-UNUSED-1, W-DROP-1), then rewrite files in the canonical layout |
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
| `rce host serve [GRANT...] [http HOST:PORT] FILE D` | run the host loop as a service |
| `rce host replay [redact F,...] FILE D` | replay D with every adapter disconnected |
| `rce db D CMD [ARGS]` | RCEdb, the database whose storage is the log D |

A command line is the command, then its words, then its operands: every
word comes before the first file. A word such as `json` takes no value, a
word such as `fuel N` takes the next argument, and no argument starts with
`-`. A GRANT is one of `fs DIR`, `blob DIR`, `media DIR`, `wasm DIR`,
`net HOST:PORT`, `chan HOST:PORT`, `tls`, `peer HOST:PORT` (after `tls`,
`peer HOST:PORT KEY`) and `pty`. `rce help`, or `rce` with no arguments,
prints every command with its words.

Exit status: 0 success; 1 a program requires change, a test failed, a
run reached a step that kept the state, or `fmt check` found a file to
change; 2 the command was used wrongly; 3 `rce equiv` or `rce change`
could not decide within its bound.

Every language command is deterministic: the same input files give the
same output bytes. None reads the clock, the network or environment
variables. The full definitions are in section 10 of the specification.
