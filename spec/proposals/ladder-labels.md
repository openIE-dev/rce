# Ladders and labels (D.12, LL-1 to LL-4)

**Status:** open. Text in Appendix D.12; tables and rules in
`spec/checker/src/ladder.rs`; tests in `spec/checker/tests/ladder_map.rs`.

## What it adds

The research papers each name a cost-ordered ladder (Gates: Lookup,
Formula, Solver, Model; MEI: the same four and a fifth rung; Spellcheck:
Lookup, Formula, Model; JouleOS EOC: state-construct, retrieve, refine,
check) and an evidence scheme (the JouleOS contract tags `HwShunt`,
`ModelBased`, `Estimator`; the seven MEI classes; the Latency paper's
proof, demonstrated, opinion and model output). D.12 places every rung
on a gear G0 to G5, the gate or the Form's input, and every class on a
D.11 label (`unpriced` < `est` < `reported` < `measured`), a bound, or the
cited source's own label.

## Rules

- LL-1: rungs in order (input first, gate last, gears strictly rising
  from G0).
- LL-2: no class gives a label stronger than its evidence; only a wall or
  shunt meter gives `measured`; a sum takes the weakest label.
- LL-3: `unpriced` gives `j` null, never 0; a bound gives no joule figure.
- LL-4: the printed tables are the code tables, row for row.

## What it does not change

No receipt, wire form, δ or check changes meaning. The tables are a
reading aid with a test; a paper that renames a rung or adds a class
changes `ladder.rs` and D.12 together, and `the_d12_tables_are_the_code_tables`
fails until both agree.
