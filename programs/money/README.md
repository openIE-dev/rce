# Money: accounts and ledgers

Programs that move money, where an operation that does not apply must keep
the state, with nothing half done.

| Program | What it shows |
|---|---|
| `account.rce` | an account that keeps its state on overdrafts and non-positive amounts |
| `ledger.rce` | double-entry bookkeeping; each step also checks that all balances sum to zero |
| `change/decimal.rce` | shows that there is no `decimal` type |

Money is kept in integer cents. Draft 0.5 has no decimal or fixed-point
type, and a program that uses one requires change:

```text
$ rce check change/decimal.rce
requires change change/decimal.rce: line 1: W-NAME-1 unknown type `decimal`; line 1: W-TY-1 expected never, found ?0
```

(`decimal.cmd` and `decimal.out` keep this transcript checked.) Integer arithmetic in RCE traps
on overflow instead of wrapping, so a balance cannot silently become wrong.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
