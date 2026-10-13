# Appendix A: commands

| Command | What it does |
|---|---|
| `rce check FILE...` | parse, type check and admit; one verdict per file |
| `rce test FILE...` | the same, and count the examples and checks of each file |
| `rce run FILE "E1, E2"` | apply events and print the state after each |
| `rce run FILE` | run `fn main()` and print what it emits |
| `rce emit TARGET FILE` | `form`, `rust`, `python` or `js` |
| `rce fmt [--check] FILE...` | rewrite files in the canonical layout |
| `rce lsp` | language server on standard input and output |

All files given to one command form one program.

Exit status: 0 success; 1 a program was refused, a test failed, a run
reached a refused event, or `fmt --check` found a file to change; 2 the
command was used wrongly.

Every command is deterministic: the same input files give the same output
bytes. No command reads the clock, the network or environment variables.
The full definitions are in section 10 of the specification.
