# Proposal: two-regime `est_j`, the `host.Calibrated` entry, and the R² falsifier

Status: proposal. Owner: David Charlot. Derived from C.6 (budget units),
E.6 to E.7 (fabrics), 4.4 of the paper (receipts; the label lattice
unpriced < est < reported < measured) and the energy survey of 2026-10-22.

In plain words:

1. The budget table counts work; it is deterministic, in the log, and the
   same on every target. It is never joules, and it stays that way.
2. Joules per unit of work differ by machine. On a microcontroller or an
   FPGA, joules are proportional to counted operations to within a few
   percent; on a laptop or server core, power is nearly flat while the
   core is busy, so joules are proportional to *time*, and the count
   predicts almost nothing.
3. So a host measures its own joules per unit once, under a written
   protocol, and writes the result into the log with its error. Every
   estimate after that is the count times the logged calibration, and the
   calibration can be replayed and checked like anything else in D.

## 1. Three functions over one e-class

| Function | Input | Output | Where it lives | Target-blind |
|---|---|---|---|---|
| budget units (C.6) | term | integer count | D, trap-placing (T52) | yes |
| `est_j` | term, target | joules (estimate) | receipts (4.4) | no |
| representative (6.10, D) | e-class | one term | shape_eq, shape_ctl | yes |

The emitted term is chosen by `est_j` on the chosen target; the hashed
representative is chosen by the target-blind tree cost with the canonical
JSON tie-break. They may differ. Neither may differ across engines given
the same D.

## 2. The two regimes

- **Regime A (count-linear): MCU, FPGA, in-order cores.** `est_j` =
  ⟨count(term), E[target]⟩ + P_idle · t, where `count` is the vector of
  C.6 units by operation class and `E[target]` is a vector of joules per
  unit from a `host.Calibrated` entry. Published fits of this shape are
  within 3–4% on Cortex-M0 class cores (R² 0.99).
- **Regime B (time-linear): out-of-order CPUs and GPUs.** `est_j` =
  P_active · t + E_llc · misses, where t is the step's measured or
  modelled time. Replications of language-energy studies at fixed
  single-core load show power flat to ±0.5 W at ~190 W, so energy is time
  and the count enters only through time.

A fabric record (E.7) names its regime. A host that has no calibration
reports `unpriced`, never a number.

## 3. `host.Calibrated`

A host entry (E.5), never read by δ, written by `rce host calibrate`:

    host.Calibrated {fabric, regime, E: {class: {j_per_unit, se}}, p_idle,
                     p_active, protocol, n, duration_s, meter, hash}

Protocol (normative for the entry to be written): at least 30 interleaved
A/B pairs per operation class against the idle baseline; each block at
least 60 s or 10⁹ operations; warm-up discarded; temperature recorded;
gross and net reported; the meter named (`rapl`, `ina260`, `wall`), with
its own validation reference; RAPL sampled no faster than 10 ms. An entry
whose protocol fields are missing is refused by the admitter.

Receipts (4.4) gain two fields: `calibration` (the hash of the
`host.Calibrated` entry used) and `stale_fraction` (the share of the
step's operations whose class has no calibrated row, priced `unpriced`).
A total over receipts keeps `unpriced` separate, as now.

## 4. What it refuses

- Joules in C.6. The table stays units.
- Per-language constants ("Python costs 70× C"): the survey found these
  do not survive controlled replication.
- Non-additive effects in the extraction objective: DAG-cost extraction
  with a linear per-target objective is the admitted form; frequency
  scaling, cache state and operand-dependent power are regime-B time
  effects, measured, never modelled into the cost.
- Summing across labels. `measured_j`, `reported_j`, `est_j` and
  `unpriced` are reported in separate columns, always.
- Landauer's bound as a price. It is a labeled floor eight to eleven
  orders below CMOS, reported once as context, never as a cost.

## 5. The falsifier

**RCE-F5.** On the ladder subsample under the wall meter (8.3), with a
`host.Calibrated` entry for the host, the per-target R² of `reported_j`
on the C.6 count vector is below 0.8 in regime A, or `est_j` in regime B
is off by more than 25% at the median. Either defeats the claim that the
calibrated estimate is a usable proxy, and the receipt's `basis` for that
host becomes `unpriced` until a better model is calibrated.

First measurement: the KV260 at rung 3 on the VCCINT rail (regime A),
N ≥ 30 pairs, the gates paper's protocol extended with the per-class
blocks above.
