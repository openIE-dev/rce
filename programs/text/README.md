# Text: lexing and comparing

Programs that read and compare text.

| Program | What it shows |
|---|---|
| `word_count.rce` | a lexer that counts words one byte at a time, as a state machine |
| `edit_distance.rce` | Levenshtein distance by dynamic programming |
| `json_nest.rce` | a JSON parser for input of any nesting depth, with no bound written |
| `change/json_descent.rce` | recursive descent over a position, which requires change (W-REC-1) |

**Nested input without fuel.** Every loop and every recursion in RCE ends
by a rule of section 6.7. `json_nest.rce` meets the rule with loops over
its input and writes no bound: one `for` over the characters makes the
lexemes, one `for` over the lexemes builds the tree with an explicit
stack, so the number of iterations is the length of the input, whatever
its depth. The tree it builds is a recursive `kind`, and the functions that
measure it (`depth`, `count`) recurse structurally: each call takes a field
of a constructor just matched, so the depth of the calls is the depth of
the tree.

The familiar way to write a parser, recursive descent, requires change
(`change/json_descent.rce`, W-REC-1): a call that parses one element
returns the position after it, and the next call starts there, but that
position is a computed value, not a component of a matched parameter, so
no measure of 6.7 shrinks. A `decreases` clause does not help: its
fragment (F.6.4) is linear in the parameters, and the returned position is
not one, and a written measure never replaces the recursion bound. The one
rule that admits recursive descent is a declared bound (`fn value(...) ->
Parsed bound 2000`), a literal cap on the active calls, so recursive
descent with no cap on depth is not expressible.

The program writes no bound, but the reference evaluator has limits of its
own: fuel per evaluation (6.7, E1), SIZE_LIMIT (N9) and the evaluation depth.
Measured with the reference interpreter (openie-research-experiments
`rce/bench/bounded-programs`), the loop parser returns the right depth at
1, 10, 100, 1000, 2000 and 4000 and the right failure position on each
malformed input, but its time grows faster than the input: depth 1000, 2000 and 4000 took
2.6, 13.7 and 72 s, and a flat array of 1000, 4000 and 16000 numbers took
0.59, 16.4 and 327 s. The program does a fixed amount of work per lexeme;
the reference interpreter copies a value each time it is moved, so moving
a subtree (or the list of children built so far) of size k costs k. The structural recursion over a deep tree also meets a
defect of the reference checker: `rce test` of `check depth_of(nest(d)) ==
d` passes at d = 800 and aborts the process from d = 807 (SIGABRT, a
native stack overflow inside the interpreter's recursive copy of a value),
where 6.7 promises the defined `evaluation depth` trap. Under `rce run` the
same recursion returns at d = 1000 (8.5 s, 2.1 GB peak memory) and aborts
at d = 2000. That is why the depth 1000
check in `json_nest.rce` uses the depth the loop counted.

Text in RCE is UTF-8. `len()` and indexing count bytes, which is why
`edit_distance.rce` compares bytes. For text with characters outside ASCII,
compare `chars()` instead.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
