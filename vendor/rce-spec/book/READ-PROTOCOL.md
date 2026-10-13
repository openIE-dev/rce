# Read protocol for agents

This file says how an AI agent shows that it has read the whole RCE spec,
rather than claiming to. The spec is split into chapter files small enough
to read whole. Each chapter ends with a footer holding the SHA-256 of its
body and has one read-receipt question whose answer appears only in that
chapter.

## What the agent must do

1. Read every file `spec/book/NN-*.md` listed in `INDEX.md`, in full, in order.
   Skimming, searching or reading a summary does not count.
2. For each chapter, copy the hash from its last line
   (`<!-- body-sha256: ... -->`) and answer its **Read receipt** question
   from the chapter's text. Answers are short: a word, a name, a rule or a
   number.
3. Report the result as JSON, one entry per chapter:

```
{"files": {
  "00-overview.md": {"sha256": "<64 hex digits>", "answer": "<answer>"},
  "01-notation.md": {"sha256": "...", "answer": "..."}
}}
```

4. Say plainly which files were not read, if any. Never fill in a hash or an
   answer for a file that was not read.

## How it is checked

`rce-book verify receipt.json` (in `spec/checker`: `cargo run --release --bin rce-book -- verify receipt.json`) recomputes each chapter's
hash from the file on disk and compares each answer with a salted SHA-256
stored in `receipts.json` (the answers themselves are not stored anywhere).
It prints PASS or FAIL per chapter and `VERIFIED` only when every chapter
passes. Answers are compared after trimming, lower-casing and collapsing
spaces.

A hash alone proves only that the agent opened the file; the answer shows it
read the part that holds it. A correct receipt is evidence of reading, not
of understanding. The salts are public, so a determined agent could guess
short answers by hashing words from a chapter; doing that still requires
having the chapter's text.

## Keeping it valid

When a chapter changes, run `rce-book build`: it rewrites the
footers, `INDEX.md` and `spec/RCE-SPEC.md`. `rce-spec-check` fails if
the book and the single file are out of sync. If a change removes the text
that answers a chapter's question, change the question and its salted answer
in `receipts.json` in the same commit.

<!-- body-sha256: b9bfae27240825038ca4b9639e11616786d9b797739d6ad4d30d3e22a214618a -->
