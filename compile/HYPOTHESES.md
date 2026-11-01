# RCE Compile hypotheses

**Registered 2026-11-01, before any measurement by this project.** The
hypotheses are those of the research report *Law of compilation research*
(its research programme), with the metric, the support and falsification
conditions and the experiment written out. The report's own probe of
reproducibility on its machine came before this registration; experiment
E1 is the first measurement made under it. A change after registration is
a row in the table of changes at the end, with its date and reason, and
never an edit that hides what was registered.

Terms are those of [CHARTER.md](CHARTER.md). A *declared corpus* is a set
of programs fixed, by path and hash, before the measurement that uses it.

## H1. Lowering is a function of declared inputs

**Statement.** With declared inputs (source, flags, toolchain, remapped
build path, fixed `SOURCE_DATE_EPOCH`), each compiler's output bytes are
identical across directories and across machines.

- **Metric.** For each configuration and program, the number of differing
  bytes per output file between two builds whose declared inputs are equal:
  in one directory, in two directories with the path remapped, and on two
  machines with the same toolchain version.
- **Supported if** the count is 0 for every output file in every such pair.
- **Falsified if** any byte difference survives remapping, across
  directories or across machines.
- **Experiment.** E1, [experiments/e1-determinism/](experiments/e1-determinism/PROTOCOL.md).

## H2. Gates can be classified

**Statement.** Gates differ by compiler and can be classified: some emit
nothing by default when a check fails, others emit unless configured not
to.

- **Metric.** For each compiler and each injected fault, whether an artefact
  is written, its exit status, and the flags under which this changes; for
  each gate, its stated property class, linking assumption and trusted
  computing base.
- **Supported if** every compiler falls into one class and its observed
  behaviour matches its documented behaviour.
- **Falsified if** a compiler's documented gate behaviour differs from what
  the injected-fault test observes.
- **Experiment.** E2, [experiments/e2-gate-audit/](experiments/e2-gate-audit/PROTOCOL.md).

## H3. Every pass and flag has an equivalence level

**Statement.** Every optimisation pass and flag can be assigned one of the
four equivalence levels.

- **Metric.** For each pass or flag, the level assigned before the run, and
  the largest observed output change against an exact reference under
  differential testing (bits differing; ULP distance; absolute and relative
  error).
- **Supported if** no observed change exceeds its assigned level.
- **Falsified if** a pass's observed output change exceeds its assigned
  level.
- **Experiment.** E3, [experiments/e3-equivalence-levels/](experiments/e3-equivalence-levels/PROTOCOL.md).

## H4. Erasure and description length against measured energy

**Statement.** The counted irreversible erasure ε plus a declared
description-length proxy λ predict measured energy in regime C (or a
simulation of it), but explain less variance than operation counts in
regime A and than time multiplied by power in regime B.

- **Metric.** On a declared corpus, the share of variance in measured
  joules explained by ε and λ in each regime, next to the share explained
  by the regime's own baseline (counted operations in A; time multiplied by
  power in B). The statistical threshold is declared in the E4 protocol
  before measurement.
- **Supported if** ε and λ explain variance in regime C above the declared
  threshold, and less than the baseline in regimes A and B.
- **Falsified if** ε and λ explain no variance in regime C, or explain more
  variance than the regime's own baseline in A or B, which would contradict
  the scale-gap analysis of the charter.
- **Experiment.** E4, [experiments/e4-energy/](experiments/e4-energy/PROTOCOL.md).

## H5. A name-free hash raises reuse without false hits

**Statement.** A name-free hash (bound names replaced by positions, path
removed) raises reuse hits on renamed but equal programs, without false
hits.

- **Metric.** On a declared corpus of program pairs, some equal up to
  renaming and some not, the hit count of a cache keyed by the name-free
  hash against one keyed by the compiler's own key, and the number of false
  hits (two inequivalent programs sharing a key).
- **Supported if** hits rise on the equal pairs and there is no false hit.
- **Falsified if** there is any false hit, or no gain in hits on the
  declared corpus.
- **Experiment.** Planned (E5); no protocol yet. It uses the corpus of E4.

## H6. Deterministic rungs decide at least 98%

**Statement.** The deterministic rungs G0 to G2 decide at least 98% of
accepted decisions on a declared corpus, and every stochastic proposal is
gated.

- **Metric.** Per rung, the share of accepted decisions in a unit declared
  before the run (accepted decisions, accepted rewrites, CPU seconds or
  joules, each reported on its own), and the number of stochastic outputs
  released without a recorded gate decision.
- **Supported if** the stochastic share is at most 2% in the declared unit
  and every stochastic output has a recorded gate decision.
- **Falsified if** the measured stochastic share is above 2% in the declared
  unit, or any stochastic output is released without a recorded gate
  decision.
- **Experiment.** Planned (E6); no protocol yet. It needs a compiler built
  on the gear ladder, which does not exist yet.

## H7. Partial lowering with stops admits no miscompilation inside its fragment

**Statement.** Lowering stated as partial, with a stop as an outcome, has a
measurable stop rate, and gated lowering admits no miscompilation that
EMI-style testing finds inside the checker's decidable fragment.

- **Metric.** The share of inputs on which a step stops, and the number of
  miscompilations found by EMI-style testing inside the fragment the gate
  claims to cover.
- **Supported if** the stop rate is measured and no such miscompilation is
  found.
- **Falsified if** EMI-style testing finds a miscompilation inside the
  fragment the gate claims to cover.
- **Experiment.** Planned (E7); no protocol yet. It uses the corpus of E4.

## Table of changes

| Date | Change | Reason |
|---|---|---|
| 2026-11-01 | H1 to H7 registered from the research report, with metrics, conditions and experiments | registration before E1's first run |
