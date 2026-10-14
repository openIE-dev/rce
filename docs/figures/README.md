# Figures

Figures for the RCE specification, with the scripts and data that produce
them. Every PNG has its generating script under `src/`; diagrams also ship
as SVG. Figures marked **[opinion]** encode the author's judgement rather
than a measurement or a specification rule.

## Diagrams of the specification

| Figure | What it shows | Source |
|---|---|---|
| [rce-overview](rce-overview.svg) ([PNG](rce-overview.png)) | The pipeline: source, checker, Form P = (S, E, δ, D, λ), log D, `rce emit` targets (Rust, Scala, Haskell, Julia, wasm), host and compute fabrics. | `src/infographics/make_infographics.py`; spec 0, 6.1 |
| [host-contract](host-contract.svg) ([PNG](host-contract.png)) | Appendix E: the five primitives every surveyed target supplies, adapters, the single admitter, D, and replay with adapters disconnected (HC-1, HC-3, HC-8, T37). | `src/infographics/make_infographics.py`; spec Appendix E |
| [rcedb](rcedb.svg) ([PNG](rcedb.png)) | RCEdb: D is the one primary artefact; indexes are deterministic folds written in RCE (example 55); queries read indexes with a defined order. | `src/infographics/make_infographics.py`; spec 6.1, D.5, D.6 |
| [status](status.svg) ([PNG](status.png)) | Test and differential results, October 2026: Scala emit 55/55, Haskell emit checked by GHC 451,037/451,037, Julia probe 0 mismatches in 200,000, IDE 811 tests. | `src/infographics/make_infographics.py` |
| [tensor-unification](tensor-unification.svg) ([PNG](tensor-unification.png)) | **Proposed, not normative.** Tensor unification of primitives (E.11 TU-1 to TU-6 in the engine draft): Tensor[elem, shape], static shape check, fixed-order reductions, placement by (op, elem, shape), ragged values kept separate. | `src/infographics/make_tensor.py` |
| [rcedb-graph](rcedb-graph.svg) ([PNG](rcedb-graph.png)) | RCEdb graph index (spec E.8.1): the log D is the base record; a typed property graph is folded from a log prefix in RCE, every node and edge carries the event that made it, queries are typed pattern values with canonical order. **G-1 to G-6 are proposed, not normative**; as-of, result order and termination follow from HS-8, HS-9 and T2. | `src/infographics/make_rcedb_graph.py` |

## Survey figures (inputs to Appendix E and the emit targets)

| Figure | What it shows | Source |
|---|---|---|
| [host-survey-fig1](host-survey-fig1-coverage-heatmap.png) | Host needs against surveyed targets; cells are read from sources or **[opinion]** derived judgement where no primary source was found. | `src/host-survey/gen.py`, `MATRIX.csv` |
| [host-survey-fig2](host-survey-fig2-memsafety-share.png) | Share of vulnerabilities that are memory-safety bugs, as reported by each vendor (sourced). | `src/host-survey/gen.py` |
| [host-survey-fig3](host-survey-fig3-latency.png) | Measured host boundary costs (syscall, clock, wasm call, SHA-256, fsync and others) on a Linux VM and a Mac. | `src/host-survey/gen.py`, `m.json` |
| [host-survey-fig4](host-survey-fig4-fabric.png) | Cost of strict IEEE determinism on GPU kernels and a calibrated fabric dispatch model. | `src/host-survey/gen.py`, `FABRICS.csv` |
| [nocode-fig1](nocode-fig1-walls-heatmap.png) | Categories of the most-discussed limits in no-code and low-code projects (issue tracker sample). | `src/nocode/figs.py`, `walls_by_repo.json` |
| [nocode-fig2](nocode-fig2-form-mapping.png) | No-code schema primitives mapped onto the Form P = (S, E, δ, D, λ). | `src/nocode/figs.py` |
| [lang-fig1](lang-fig1-heatmap.png) | **[opinion]** Language against RCE design dimension, author judgement from the cited sources. | `src/lang-structure/gen.py`, `MATRIX.csv` |
| [lang-fig2](lang-fig2-retrofit-timeline.png) | Years of effort for retrofits of language properties; some start dates are background knowledge. | `src/lang-structure/gen.py` |
| [fp-emit-selection](fp-emit-selection.png) | How the functional emit targets were chosen: popularity rank and two usage signals among candidates passing the hard filter. | `src/fp-emit/score.py`, `candidates.csv` |
| [replacement-matrix](replacement-matrix.svg) ([PNG](replacement-matrix.png)) | Failure classes per language family: excluded by RCE, defined by RCE, or a gap. | `src/replace/matrix.py`, `failure-classes.csv` |

Regenerate the diagrams with `python3 src/infographics/make_infographics.py`
(matplotlib) from this directory.
