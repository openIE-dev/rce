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
- **Experiment.** E5, [experiments/e5-name-free/](experiments/e5-name-free/PROTOCOL.md),
  on the corpus of E4 and E1's C and Rust programs.
- **E5's predictions** (registered 2026-11-02, before E5's first run; the
  table of changes records the registration). Two R hashes per output, a
  WARP-style function ID at the machine-code level (unlinked objects:
  x86-64 ELF, arm64 Mach-O, wasm32) and an alpha-equivalence hash at the IR
  level (linked WebAssembly modules), each over the whole output and over
  `run` alone, against two baseline keys (output bytes; source text and
  flags with the path remapped). Predicted: every R hash equal on every
  pair of builds of the same source in two directories, and on every pair
  of a source and its renamed copy that has evidence of being the same
  program; the source key unequal on every renamed pair; R hashes unequal
  on most pairs of optimisation levels (equal pairs reported, not counted
  as false hits) and on every pair of C and Rust builds of one algorithm,
  at both levels; no R hash equal on any pair of builds of two different
  kernels; the IR hash changed by every single change of an opcode within
  its class, a constant or a memory offset.
- **E5's falsifiers.** A false hit: any R hash equal on a pair of builds of
  two different kernels, or an IR hash unchanged by a mutation. No gain: no
  R hash with a higher hit rate than both baselines on the renamed pairs.

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
| 2026-11-02 | **H1, Go in GOPATH mode: verdict.** The one E1 difference that survived remapping on the first run (`go build, GOPATH mode`, 58–60 differing bytes, equal sizes, no copy of the build path) has a named cause. *Evidence*, all from the harness ([RESULTS.md](experiments/e1-determinism/RESULTS.md)): (1) every differing byte lies in the `.note.go.buildid` and `.note.gnu.build-id` sections, and code and data are identical; (2) `GODEBUG=gocachehash=1` shows the compile and link action IDs hashing `local true prefix "_DIR"`, where DIR is the build directory, because GOPATH mode gives a package named by a file outside every `GOPATH` the import path `_` followed by its absolute directory, and `-trimpath` does not remove it; once DIR is written as `{dir}`, the hashed inputs of the two directories are the same; (3) with only that input changed, the package placed in a `GOPATH` and built by its import path `e1` (`go build, GOPATH mode, import path declared`), no hashed input names the directory and the remapped builds are byte-identical across directories, while the original configuration still differs. *Verdict, by the registered conditions:* the falsifier is a difference that survives remapping when the declared inputs are equal; here the inputs were not equal, since the import path is an input that `go build` derives from the directory. The import path can be named and controlled, so it is added to H1's declared inputs, and H1 holds for Go in GOPATH mode with the import path named. A GOPATH-mode build of a file outside every `GOPATH` has the directory in its import path, so `-trimpath` alone is not a remapping for it | applies H1's registered conditions to the E1 evidence of 2026-11-02 |
| 2026-11-02 | **H1, macOS: no verdict yet.** The first macOS run of E1 (MacBook Pro M5 Max, macOS 27.2, arm64; [RESULTS.md](experiments/e1-determinism/RESULTS.md)) has differences in the same directory (C, C++ and Fortran with `-g`; Zig Debug) and differences that survive remapping (C, C++, Fortran and Rust with `-g`; Go in GOPATH mode; OCaml). No verdict is recorded for them. *Established:* (1) its remapped builds named the build directory by a path that the tools did not record: the system temporary directory `/var/folders` is under the link `/var -> /private/var`, Go's trace names the directory `_/private{dir}`, and a Linux check of gcc, clang and rustc found that a remapping naming a linked path leaves the real path in the output; the remapped C, C++, Fortran and Rust builds kept all three copies of the directory. Its remapped comparisons therefore do not test H1's remapping condition, and E1 now names the build directory by its real path ([PROTOCOL.md](experiments/e1-determinism/PROTOCOL.md), "Changes for macOS"); (2) for Go in GOPATH mode, the trace and the declared-import-path configuration give on macOS what they gave on Linux, while which bytes differ is not recorded on Mach-O. The rerun with Mach-O attribution and the debug map is the evidence for a verdict | records what the first macOS run establishes before its rerun |
| 2026-11-02 | **H2 and H3, evidence authority and digest binding.** CHARTER.md section 4.4 now requires each gate to state, besides its property class, linking assumption and trusted computing base, the authority of its evidence (pinned, replay, proved, derived or inferred, after REA's reconstruction ledgers) and the digests its verdict is bound to, and inferred evidence never makes a gate accept. *H2's metric* gains, for each gate, its stated authority and digest binding, each with a quoted source or "not stated"; E2's records are bound to the SHA-256 of the program, the compiler and every artefact, and a record whose digests do not match is void. *H3's metric* "whether a static check certified the level" now counts a level as certified only on evidence of authority proved bound to the build's digests; differential testing is observed evidence on the declared inputs, which can falsify H3 but not certify a level. The statements, the supported and falsified conditions and the experiments are unchanged. No measurement of H2 or H3 has been made | the change is made before E2's and E3's first runs, so no result is re-read under it |
| 2026-11-02 | **H5 attached to E5, before its first run.** H5's experiment, until now "Planned (E5); no protocol yet", is E5 ([PROTOCOL.md](experiments/e5-name-free/PROTOCOL.md), dated 2026-11-02), with the predictions and falsifiers written under H5. H5's metric compares the name-free key with "the compiler's own key"; E5 reads no compiler's cache key, and uses in its place two baseline keys, SHA-256 of the output bytes and SHA-256 of the source text and build command with the build directory removed, which stand for the two kinds of key the charter's part U describes. *False hit* is made concrete as an equal key on two builds of different kernels, or an unchanged IR hash under a single change of an opcode, constant or memory offset | registration of E5 before its first run, as section 8 of the charter requires |
| 2026-11-02 | **H5, E5's first results: supported on E4's corpus by the registered conditions** ([RESULTS.md](experiments/e5-name-free/RESULTS.md)). *Conditions:* no R hash was equal on any pair of builds of two different kernels, no tested mutation left the IR hash unchanged, and on the renamed pairs every R hash had a higher hit rate than both baselines. *Predictions not held*, recorded here so the support is read with them: (1) at the IR level, the linked Rust modules at `opt-level=0` (and `rust/sort` at `opt-level=1`) were unequal across directories and across renaming, in their data segments only; each holds a copy of its build directory, and after the renamed source the source's column numbers in Rust's overflow-message records also differ (PROTOCOL.md, "Changes after the first run"). The registered IR hash treats data bytes as constants and does not remove the build path that the charter's R removes; the diagnostic key that removes it makes every two-directory pair equal and leaves the column differences. (2) The source key was equal on the renamed pairs of E1's two hello programs, which bind no identifier, so the renamer left them unchanged. (3) The WARP-style ID of `run` in the WebAssembly objects was equal for the C and the Rust kernel of `fib` at five levels and of `popcount` at one; the linked modules' functions are alpha-equivalent in the same number of pairs, and the IR hashes still differ by the memory, globals and data each toolchain declares. These pairs compute one function, so they are not false hits. The support holds for this corpus, two toolchains and one IR | records E5's outcome against H5's registered conditions and E5's predictions |
| 2026-11-03 | **E4's macOS meter: macmon, without root.** The protocol read macOS energy with `powermetrics`, which needs root. It now reads `macmon pipe -i 100` (no root; [PROTOCOL.md](experiments/e4-energy/PROTOCOL.md)): one process for the whole run, one sample every 100 ms with the power of the CPU, GPU and Neural Engine from Apple's on-chip counters, each sample's watts times its measured span. The basis stays *reported* (an on-chip counter, not a meter on the supply). `powermetrics` is used only when macmon is absent and sudo is already granted; otherwise the run's energy is recorded as unmeasured. Made before E4's first macOS run; no hypothesis, metric or condition changes | the registered meter needs root; a counter read without root has the same basis |
| 2026-11-04 | **E4, first macOS runs (Mac Studio and MacBook Pro, both Apple M4 Max): no verdict.** Regime B only, energy *reported* by macmon (CPU, GPU and Neural Engine counters), 96 rows from 16 programs on each machine ([RESULTS.md](experiments/e4-energy/RESULTS.md)). By the declared analysis, on both machines ε and λ explain variance and the baseline, seconds per call times the file's mean net power, explains more, which is what H4 predicts for regime B: adjusted R² for ε and λ 0.528 [0.332, 0.783] on the Mac Studio and 0.328 [0.25, 0.642] on the MacBook Pro; for the baseline 0.873 [0.7, 0.951] and 0.898 [0.784, 0.955]; the baseline's share minus theirs 0.346 [0.0746, 0.558] and 0.570 [0.204, 0.662]. *No verdict on H4:* its support needs regime C, which no results file measures; and neither machine was quiet. The idle baseline moved from 17.40 W to 13.71 W on the Mac Studio and from 0.82 W to 1.31 W on the MacBook Pro, and the runs' operator reports other work during each run (on the Mac Studio a load average of 7 to 15; on the MacBook Pro a media-analysis process at 150 to 285% CPU), which the files do not record. Both files' sampler label reads `macmon macmon 0.10.0`, as the harness of commit 057b6739 wrote it. Runs on quiet machines are the evidence for regime B on macOS | records the first macOS runs before quiet reruns |
