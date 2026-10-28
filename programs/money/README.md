# Money: accounts and ledgers

Programs that move money, where a refused operation must leave nothing
half done.

| Program | What it shows |
|---|---|
| `account.rce` | an account that refuses overdrafts and non-positive amounts |
| `ledger.rce` | double-entry bookkeeping; each step also checks that all balances sum to zero |
| `fit/decimal.rce` | shows that there is no `decimal` type |

Money is kept in integer cents. Draft 0.5 has no decimal or fixed-point
type, and the checker refuses one:

```text
$ rce check fit/decimal.rce
fit fit/decimal.rce: line 1: W-NAME-1 unknown type `decimal`; line 1: W-TY-1 expected never, found ?0
```

(`decimal.cmd` and `decimal.out` keep this transcript checked.) Integer arithmetic in RCE traps
on overflow instead of wrapping, so a balance cannot silently become wrong.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
