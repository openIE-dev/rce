# Science: numerical kernels

Small numerical programs.

| Program | What it shows |
|---|---|
| `numerics.rce` | dot product, mean, and a square root by Newton's method with a fixed number of iterations |
| `knapsack.rce` | the 0/1 knapsack problem by dynamic programming |
| `graph.rce` | Dijkstra's shortest paths and breadth-first search over a graph given as input |
| `dpll.rce` | a small DPLL satisfiability solver (unit propagation and splitting) |

Floating point in RCE has one defined result for every operation, so
`numerics.rce` gives the same bits on every target. The square root test
still compares within a tolerance, which is good practice for iterative
methods.

**What search over input has to state.** Every loop and every recursion
in RCE ends by a rule of section 6.7, so a search over a graph or a formula
given as input says why it ends:

- `dijkstra` in `graph.rce` states nothing: each loop runs over the
  vertices or the edges of the input.
- `bfs` states one bound, `while head < queue.len() bound n`: the queue
  grows while it is read, so the loop says that each vertex enters it at
  most once. A wrong bound traps with "loop bound"; it is never silent.
- `dpll` states one bound, `bound 64` on the recursive split. The split
  recurses on an assignment, which has no constructor measure, and a
  written `decreases` measure does not admit a recursion by itself (6.7
  and F.6.4: the recursion bound still holds), so the only rule left is a
  literal cap on the active calls. A formula needs at most n + 1 of them,
  so this solver handles at most 63 variables: 63 tautological clauses
  give `true`, 64 stop with `trap: recursion bound`. The cap cannot be an
  expression of the input. The unit-propagation loop states that each
  pass fixes one more variable (`bound a0.len()`).
- A list literal whose inner lists differ in length requires change today
  (W-TY-1), so `dpll.rce` writes the unit clause [x] as [x, x].

Measured with the reference interpreter (`rce run`, one shared Linux
container; openie-research-experiments `rce/bench/bounded-programs`):
Dijkstra on a path of 100, 200, 400 and 800 vertices took 0.07, 0.14, 0.29
and 1.0 s; DPLL on the pigeonhole formulas PHP(m+1, m), m = 2 to 5 (all
unsatisfiable), took 0.009, 0.017, 0.088 and 0.99 s.

Limits in the current draft: there are no tensor or sparse matrix types.
Dense matrices are in the standard library module `num.linalg`
(`use num.linalg`), with `num.special` and `num.stats` beside it.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
