# Proposal: placement of a graph term by predicted joules

Status: accepted (ledger row of 2026-11-04). Implemented in
`spec/checker/src/interp.rs` (`place_graph`, `est_j`),
`spec/checker/src/host.rs` (`CostTable`, `Host::graph_costs`, the replay
fold), `spec/checker/src/wasm_term.rs` (`probe`, `at_level`) and the word
`costs` of `spec/checker/src/cli.rs`, specified in Appendix E.6 (HC-39)
and E.5 of the specification, and tested in
`spec/checker/tests/graph_placement.rs`. Owner: David Charlot.

## 1. Terms

- *Fabric*: an executor of graph terms. Two exist: `cpu`, the reference
  evaluation (6.13), and `wasm`, the program module of 10.18 run under
  `wasmi`.
- *Work*: the number of scalar operations of a term's nodes on the
  operands of one run.
- *Cost table*: per fabric, an overhead in J and a throughput in
  operations per J, with the source of the numbers.
- *Predicted joules* (`est_j`): `overhead + work / throughput` from a
  fabric's row. A prediction is computed from the table; it is not a
  reading and D never labels it as one.
- *Probe*: one run of a term on a fabric and on the reference, on the same
  operands, each side digested (BLAKE3) node by node.

## 2. The state before

HC-12 placed lane-wise binary64 operations, sums and `for … par` bodies
from `host.Calibrated` rows and logged `host.Placed`, but graph terms
always ran on the reference, and Appendix E listed `host.Placed` without
a prediction. S7 added a second realisation of every term (`wasm`), judged
at each node's level, which made a choice possible.

## 3. Decision

1. Before an input is admitted to a program with graph terms, the host
   logs the cost table as `host.Calibrated` rows of class `graph.term`,
   each with its `source` and `unit` `J`. The table is the word
   `costs FILE` or else the declared table (`cpu` 0 J and 10^9 operations
   per J; `wasm` 10^-3 J and 2.5 x 10^8 operations per J; source
   `declared`). A source that is empty or the bare word `measured` is
   refused: a measured table names its machine and counter.
2. At each run of a term the candidates are `cpu` and every fabric with an
   agreeing probe of class `graph.KEY` in D. A fabric that would predict
   fewer joules than the current choice and has no probe is probed on
   that run's operands; the probe side digests only nodes whose result is
   at the node's level, so the two digests are equal exactly when every
   node is admitted. The least `est_j` wins; a tie goes to `cpu`.
3. The host logs `host.Placed` {class `graph.term`, est_j, fabric, op KEY,
   work} and runs the term there. An error of the chosen fabric is the
   term's error, so no second run hides a fault.
4. Probes and placements of an input are appended after δ and before the
   input's entry. Replay folds the tables and probes and re-derives every
   placement, stopping at the first that differs (T37).

## 4. Why

- One choice per input, as an entry of D (law 1 of E.2c), keeps the
  placement a function of D; the probe keeps a fabric's results out of
  state until D holds evidence that they agree (HC-13).
- The declared table places on `cpu` because `wasmi` interprets
  WebAssembly; the table was written, not read from a counter, and says
  so. The coefficients in `rce-placement` are fixtures of one machine's
  BLAS and MPS kernels, which are not these fabrics, so they are not used.
- Since a fabric is admitted only when every node is at its level, and
  the exact nodes are bit for bit, the state hashes of `dense.rce` and
  `block.rce` are the same under any table.

## 5. Not done here

- A measured cost table: it needs a reading of the package counter per
  fabric (HC-38) over a fixed set of terms; the word `costs` takes it
  when written.
- Metal, MPS, AMX and CUDA fabrics: they need their hardware.
- Placement for `fn main` programs logs no entries; their terms are
  placed by the same rule.
