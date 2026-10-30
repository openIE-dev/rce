# Appendix A: commands

| Command | What it does |
|---|---|
| `rce check FILE...` | parse, type check and admit; one verdict per file |
| `rce test FILE...` | the same, and count the examples and checks of each file |
| `rce run FILE "E1, E2"` | apply events and print the state after each |
| `rce run FILE` | run `fn main()` and print what it emits |
| `rce trace --json FILE [EVENTS]` | the reference trace of each example, or of the events, as JSON |
| `rce emit TARGET FILE` | `form`, `rust`, `python`, `js` or `scala` (the emit registry lists further targets) |
| `rce fmt [--check] FILE...` | rewrite files in the canonical layout |
| `rce equiv [--fuel N] [--json] A B` | whether two programs have the same behaviour on the explored events, with the shortest distinguishing sequence when not |
| `rce fingerprint [--depth N] [--fuel N] [--json] FILE...` | a hash of what a program outputs on every event sequence up to N events long: programs `rce equiv` judges equivalent share it whatever their shape |
| `rce change [--fuel N] OLD NEW` | classify a change: erratum (every export compatible, every behaviour kept), meaning change, interface change |
| `rce dim [--fuel N] FILE` | count the behaviours a program can be in and the redundancy of its state, per region (chapter 18) |
| `rce lsp` | language server on standard input and output |
| `rce --version` | the checker version, the specification draft and the accepted language versions |

All files given to one command form one program.

Exit status: 0 success; 1 a program requires change, a test failed, a
run reached a step that kept the state, or `fmt --check` found a file to
change; 2 the command was used wrongly.

Every command is deterministic: the same input files give the same output
bytes. No command reads the clock, the network or environment variables.
The full definitions are in section 10 of the specification.
