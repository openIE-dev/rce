# Money: accounts and ledgers

Programs that move money, where an operation that does not apply must keep
the state, with nothing half done.

| Program | What it shows |
|---|---|
| `account.rce` | an account that keeps its state on overdrafts and non-positive amounts |
| `ledger.rce` | double-entry bookkeeping; each step also checks that all balances sum to zero |
| `checkout.rce` | a checkout with 40 independent options (tax regions, holds, promo codes, flags) in one record state |
| `change/decimal.rce` | shows that there is no `decimal` type |

Money is kept in integer cents. Draft 0.5 has no decimal or fixed-point
type, and a program that uses one requires change:

```text
$ rce check change/decimal.rce
requires change change/decimal.rce: line 1: W-NAME-1 unknown type `decimal`; line 1: W-TY-1 expected never, found ?0
```

(`decimal.cmd` and `decimal.out` keep this transcript checked.) Integer arithmetic in RCE traps
on overflow instead of wrapping, so a balance cannot silently become wrong.

**Many independent options.** `checkout.rce` is written by a generator
(openie-research-experiments `rce/bench/bounded-programs/tools/gen_checkout.py`):
each option is one field of the state, one or two events and their steps,
and one term of the total that `Place` emits. Forty independent options
have about 2^40 reachable states, but the program does not list them: it
grows by a fixed number of lines per option. Measured for 5, 10, 20, 40 and
80 options, the source is 56, 90, 167, 312 and 602 lines, the Form (`rce
emit form`) 12,326, 18,538, 32,612, 59,487 and 113,170 bytes (a straight
line, 1,347 bytes per option, R^2 = 0.99996), and `rce check` takes 5 to
24 ms. What does grow like 2^N is `rce dim`, which explores the reachable
states one by one: 64 states for 5 options and 4,096 for 10 (exact, in
0.01 and 2.0 s); for 20 and 40 options it stops at its default bound of
100,000 states after 236 and 1,048 s and reports a lower bound, and for 80
it did not finish in 900 s.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
