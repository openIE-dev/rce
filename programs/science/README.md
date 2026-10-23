# Science: numerical kernels

Small numerical programs.

| Program | What it shows |
|---|---|
| `numerics.rce` | dot product, mean, and a square root by Newton's method with a fixed number of iterations |
| `knapsack.rce` | the 0/1 knapsack problem by dynamic programming |

Floating point in RCE has one defined result for every operation, so
`numerics.rce` gives the same bits on every target. The square root test
still compares within a tolerance, which is good practice for iterative
methods.

Limits in draft 0.5: there are no tensor or sparse matrix types.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
