# tree-sitter-rce

A [tree-sitter](https://tree-sitter.github.io/) grammar for the RCE
programming language. It turns RCE source into a syntax tree that editors
use for highlighting, folding and navigation. The grammar is a port of the
normative EBNF in section 3 of the RCE specification, and the RCE
reference checker uses the same grammar as one of its three parsers, so it
accepts exactly the programs the specification accepts.

In plain words:

1. **It lets editors understand RCE code.** Editors that support
   tree-sitter can colour and navigate `.rce` files with it.
2. **It follows the specification exactly.** The same grammar is checked
   against the specification's own parser on every example program and on
   thousands of generated ones.
3. **It is a normal tree-sitter package.** It builds and is tested with the
   usual tree-sitter commands, with Rust, Node and C bindings.

![RCE overview: source (parsed by this grammar), checker, Form, log, emit targets, host](https://raw.githubusercontent.com/openIE-dev/rce/main/docs/figures/rce-overview.svg)

*Where this grammar sits: it parses `.rce` source for editors and is one of the reference checker's three parsers; the checker turns accepted programs into a Form. More figures: [openIE-dev/rce docs/figures](https://github.com/openIE-dev/rce/tree/main/docs/figures).*

## Quickstart

You need the [tree-sitter CLI](https://tree-sitter.github.io/tree-sitter/creating-parsers/1-getting-started.html)
and a C compiler.

```sh
git clone https://github.com/openIE-dev/rce && cd rce/tree-sitter
cd tree-sitter-rce
tree-sitter test
tree-sitter parse examples/counter.rce
```

From Rust, add the crate and load the language:

```rust
let mut parser = tree_sitter::Parser::new();
parser.set_language(&tree_sitter_rce::LANGUAGE.into())?;
let tree = parser.parse("fn half(x: u64) -> u64 = x / 2\n", None).unwrap();
```

## What is in this repository

| Path | Contents |
|---|---|
| `grammar.js` | the grammar |
| `src/scanner.c` | the external scanner: indentation (`INDENT`, `DEDENT`, `NEWLINE`), bracket depth, reserved keywords, numbers, `<-` and comments |
| `src/parser.c` | the generated parser (`tree-sitter generate --js-runtime native`) |
| `queries/highlights.scm` | syntax highlighting captures |
| `queries/folds.scm`, `queries/indents.scm` | folding and indentation captures (nvim-treesitter names) |
| `test/corpus/` | parse tests (`tree-sitter test`) |
| `test/highlight/` | highlight query tests (`tree-sitter test`) |
| `scripts/check-queries.sh` | compiles every query, runs it over the examples, and checks the version comment gate |
| `bindings/` | Rust, Node and C bindings |
| `scripts/check-sync.sh` | checks that this grammar matches the copy in the rce repository |

## Language coverage

The grammar tracks the RCE specification at openIE-dev/rce: language
versions 0.5 to 0.8, including refined types `type R = τ where p` (`where`
is a keyword only after the type of a type declaration and stays an
ordinary name elsewhere), declared recursion bounds `fn … bound n`, and the
version 0.8 indexed types `Text[Html]`, `Text[Url]`, `Text[Sql]`,
`Text[Json]`, `Secret[τ]`, `Cap[A]` and `Guarded[τ, A]`, which parse as
ordinary type applications and are highlighted as `@type.builtin`. Type
rules such as W-CTX-1 or W-REC-1 belong to the checker, not the grammar.

## Comments

Comments (`# ...`) are `comment` nodes, emitted by the external scanner as
an extra, so they can appear between any two tokens and editors highlight
them with `(comment) @comment`. A comment never contains the line end, and
comment lines do not take part in the layout rule: the indentation that
decides `INDENT` and `DEDENT` is read from the next code line, as spec 2.6
requires. Apart from the added `comment` nodes, the trees are unchanged.

## Known limits

- The `tree-sitter test` corpus runner does not apply the
  `# rce-version: 0.6` gate (spec 2.7), so tests for draft 0.6 keywords
  live in `scripts/check-queries.sh`, which parses with the CLI.

- Node names follow the reference grammar's rule names (for example
  `step_decl`, `cmp_expr`), so the tree mirrors section 3 of the
  specification rather than a simplified editor view.

## Keeping in sync with the specification

This directory is the one copy of the grammar. The reference checker in
`../spec/checker` compiles `src/parser.c` and `src/scanner.c` from here
(feature `tree-sitter`), `rce-spec-check` regenerates the parser from
`grammar.js` and fails when the committed `src/` is stale, and
`tests/tree_sitter_sync.rs` parses every `.rce` file of the repository with
this grammar and the Rust parser and fails on any disagreement. A grammar
change is one commit: the EBNF in `../spec/book/03-syntax.md`, `grammar.js`
and the regenerated `src/`, the corpus under `test/`, and the examples.

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
