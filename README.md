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

## Quickstart

You need the [tree-sitter CLI](https://tree-sitter.github.io/tree-sitter/creating-parsers/1-getting-started.html)
and a C compiler.

```sh
git clone https://github.com/openIE-dev/tree-sitter-rce
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
| `test/corpus/` | parse tests (`tree-sitter test`) |
| `bindings/` | Rust, Node and C bindings |
| `scripts/check-sync.sh` | checks that this grammar matches the copy in the rce repository |

## Known limits

- Comments (`# ...`) are skipped by the scanner and do not appear in the
  tree, so editors cannot highlight them yet.
- Node names follow the reference grammar's rule names (for example
  `step_decl`, `cmp_expr`), so the tree mirrors section 3 of the
  specification rather than a simplified editor view.

## Keeping in sync with the specification

The source of truth is `spec/tree-sitter/` in the
[rce repository](https://github.com/openIE-dev/rce). A grammar change is
made there first, checked with the reference checker's three-parser test,
and then copied here. `scripts/check-sync.sh /path/to/rce` reports any
difference. `src/parser.c` differs from the rce copy only in the version
metadata that `tree-sitter.json` adds.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Everyone taking part follows the
[code of conduct](CODE_OF_CONDUCT.md).

## License

Copyright (c) 2026 OpenIE.

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution
you intentionally submit for inclusion in this work shall be dual licensed
as above, without any additional terms or conditions.
