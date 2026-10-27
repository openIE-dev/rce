# Money: accounts and ledgers

Programs that move money, where a refused operation must leave nothing
half done.

| Program | What it shows |
|---|---|
| `account.rce` | an account that refuses overdrafts and non-positive amounts |
| `ledger.rce` | double-entry bookkeeping; each step also checks that all balances sum to zero |
| `price.rce` | prices as exact decimals with `core.exact`: 0.10 + 0.20 is 0.30, and a third of 10.00 rounds to 3.33 |
| `refused/decimal.rce` | shows that `decimal` is not a built-in type name |

`account.rce` and `ledger.rce` keep money in integer cents. Exact decimals
come from the standard library: `use core.exact` gives `Dec`, an unbounded
integer coefficient with a power-of-ten exponent, with `dec_parse`,
`dec_tostr`, `dec_add`, `dec_sub`, `dec_mul`, `dec_div` and `dec_quantize`
under explicit rounding modes (`price.rce` uses them). There is no built-in
type named `decimal`, and the checker refuses that name:

```text
$ rce check refused/decimal.rce
REFUSED refused/decimal.rce: line 1: W-NAME-1 unknown type `decimal`; line 1: W-TY-1 expected never, found ?0
```

(`decimal.cmd` and `decimal.out` keep this transcript checked.) Integer arithmetic in RCE traps
on overflow instead of wrapping, so a balance cannot silently become wrong.

Check this folder with `rce test *.rce`, or the whole repository with
`scripts/check.sh`.
