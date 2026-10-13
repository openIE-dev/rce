# Text: lexing and comparing

Programs that read and compare text.

| Program | What it shows |
|---|---|
| `word_count.rce` | a lexer that counts words one byte at a time, as a state machine |
| `edit_distance.rce` | Levenshtein distance by dynamic programming |

Text in RCE is UTF-8. `len()` and indexing count bytes, which is why
`edit_distance.rce` compares bytes. For text with characters outside ASCII,
compare `chars()` instead.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
