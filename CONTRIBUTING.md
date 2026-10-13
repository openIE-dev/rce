# Contributing to tree-sitter-rce

## Queries and bindings

Improvements to `queries/` (highlighting, and new `locals.scm`,
`tags.scm` or `injections.scm`) and to the bindings can be made here
directly. Run `tree-sitter test` and check a few files with
`tree-sitter query queries/highlights.scm FILE` before opening a pull
request.

## Grammar changes

The grammar is defined by the RCE specification, so a change to
`grammar.js` or `src/scanner.c` starts in the
[rce repository](https://github.com/openIE-dev/rce):

1. Change the EBNF in `spec/book/03-syntax.md` and the port in
   `spec/tree-sitter/`, with examples, as described in that repository's
   CONTRIBUTING.md.
2. Once that is merged, copy `grammar.js` and `src/` here, run
   `tree-sitter generate --js-runtime native`, update `test/corpus/` with
   `tree-sitter test --update`, and read the changed trees.
3. Run `scripts/check-sync.sh /path/to/rce`.

A pull request here that changes the grammar on its own cannot be merged,
because the grammar would then disagree with the specification.

## License

By contributing, you agree that your contribution is licensed under
MIT OR Apache-2.0, as described in [COPYRIGHT](COPYRIGHT).
