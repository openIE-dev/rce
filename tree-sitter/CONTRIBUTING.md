# Contributing to tree-sitter-rce

## Queries and bindings

Improvements to `queries/` (highlighting, and new `locals.scm`,
`tags.scm` or `injections.scm`) and to the bindings can be made here
directly. Run `tree-sitter test` and check a few files with
`tree-sitter query queries/highlights.scm FILE` before opening a pull
request.

## Grammar changes

The grammar is defined by the RCE specification, and this directory is
its only tree-sitter copy, so a change is one commit in this repository:

1. Change the EBNF in `../spec/book/03-syntax.md` and `grammar.js` (and
   `src/scanner.c` when the lexer changes), with an accepted example under
   `../spec/examples/`, as described in the repository's CONTRIBUTING.md.
2. Run `tree-sitter generate --js-runtime native` here and commit `src/`;
   update `test/corpus/` with `tree-sitter test --update` and read the
   changed trees.
3. Run `cargo run --release --bin rce-spec-check` and
   `cargo test --release --test tree_sitter_sync` in `../spec/checker`.

A pull request here that changes the grammar on its own cannot be merged,
because the grammar would then disagree with the specification.

## License

By contributing, you agree that your contribution is licensed under
MIT OR Apache-2.0, as described in [COPYRIGHT](COPYRIGHT).
