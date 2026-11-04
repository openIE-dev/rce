# E4: erasure and description length against measured energy (protocol)

Status: protocol and harness. The harness runs; no energy has been read on
a metered host yet. The committed results file is an ε and λ run on a
machine with no energy counter, and it contains no joules. Tests H4 of
[HYPOTHESES.md](../../HYPOTHESES.md): the erasure count ε and a declared
description-length proxy λ ([CHARTER.md](../../CHARTER.md) section 5.3)
predict measured energy in regime C, and explain less variance than the
regime's own baseline in regimes A and B.

## Method

1. **Declare the corpus**: a set of programs, each compiled under several
   configurations that the checker of E3 accepts as equal at the exact
   level, so that energy differences come from lowering and not from a
   change of program.
2. **Count ε.** Instrument a run of each compiled program so that every
   write of a register or memory location adds the number of bits of the
   previous value that were not copied out before being overwritten; values
   that go dead without being written over are counted when their storage
   is next reused. The instrumentation (an IR interpreter, or a binary
   instrumentation tool) is chosen and pinned before the run, and its
   definition of "copied out" is stated with it.
3. **Compute λ** with the proxy declared before the run: one of IR node
   count, binary bytes, compressed bytes with a named compressor and level,
   or a two-part MDL code relative to a named model class.
4. **Measure joules** for each run, repeated enough times to state an
   interval:
   - regime A: an FPGA or in-order core with a meter on its supply, and the
     count of executed operations as the baseline;
   - regime B: an x86 host with RAPL (Intel's running average power limit
     counters) or a GPU with NVML (NVIDIA's management library), and time
     multiplied by power as the baseline;
   - regime C: a simulation of entropy production, or thermodynamic
     hardware if a measured device becomes available.
5. **Log compile-time energy** for every configuration as well, since the
   energy a compiler spends can exceed what it saves (charter section 5.2).
6. **Declare the analysis before the run.** The regression of joules on ε
   and λ, and on the baseline, is declared here before measurement: an
   ordinary least-squares model per regime, with the share of variance
   explained and a 95% bootstrap interval over programs. ε and λ explain
   variance when that interval for their share excludes zero.

Results are written by the harness as JSON with the machine, the meter and
its calibration, and the toolchain versions, and the write-up is generated
from them, as in E1. Energy figures without a meter reading are labelled
modelled.

## Declared before the first measurement (2026-11-02)

The method above leaves these choices open. Each is fixed here, with its
reason, before any energy is read; a change after the first metered run is
a dated row in the table of changes of [HYPOTHESES.md](../../HYPOTHESES.md).

### Files

| File | Contents |
|---|---|
| [matrix.json](matrix.json) | the corpus, the configurations, the build commands, the inputs and every constant below |
| [programs/](programs/) | the kernels in C and Rust, and the C driver |
| [reference.py](reference.py) | the expected value of each kernel, written from the kernel's comment |
| [wasm_eps.py](wasm_eps.py) | the pinned instrumentation for ε |
| [meters.py](meters.py) | the RAPL, macmon and powermetrics readers |
| [stats.py](stats.py) | the declared analysis |
| [e4.py](e4.py) | the harness, which writes `results/*.json` |
| [report.py](report.py) | writes [RESULTS.md](RESULTS.md) from the results files and checks it |
| [test_e4.py](test_e4.py) | unit tests, with the ε of tiny modules worked out by hand |

### Corpus and configurations

- **Programs.** Eight kernels (`sumsq`, `collatz`, `sieve`, `sort`,
  `crc32`, `matmul`, `fib`, `popcount`), each written in C and in Rust:
  16 programs. Each is a function `run(n)` on 32-bit unsigned integers that
  uses no library, plus a native driver that calls it R times and prints
  `run(n)` and a checksum of the R calls. *Reason:* ε needs an instrumented
  run, and a kernel without a library or system call can be run by the
  same interpreter on every host; the eight cover arithmetic, branches,
  memory, calls and bit operations. E1's programs print through the C and
  Rust libraries, so they are not reused.
- **Configurations.** Six optimisation levels per toolchain: `clang -O0,
  -O1, -O2, -O3, -Os, -Oz` and `rustc -C opt-level=0, 1, 2, 3, s, z`
  (Rust edition 2021). 96 rows of program and configuration. *Reason:* both
  compilers lower through LLVM to native code and to WebAssembly, so ε can
  be counted for every row; GCC has no WebAssembly target and is left out.
- **Input.** Each program's `eps_input` (matrix.json) is the n whose call is
  counted for ε and timed and metered for energy, chosen so that one call
  executes about 10⁵ WebAssembly instructions at `-O2`.

### Equivalence evidence (weaker than E3)

The method asks for configurations that E3's checker accepts as equal at
the exact level. That checker does not exist yet. The evidence used instead
is **output equality on declared inputs**: for every row, the native program
and the WebAssembly module give `reference.py`'s value for each input in the
program's `inputs` (0, 1, 2, 10 and `eps_input`), and every energy run's
printed value equals it too. This is weaker than E3's exact level, which
decides equality for every input; a configuration that differs only on an
input not declared passes this evidence. A row whose evidence fails is kept
in the results file and left out of the energy runs and of the analysis.

### The ε count

- **Instrumentation.** [wasm_eps.py](wasm_eps.py), an interpreter of
  WebAssembly written for E4 in Python with no dependency outside the
  standard library. Each row's kernel is built a second time, for
  `wasm32` (`clang --target=wasm32 -nostdlib`, `rustc --target
  wasm32-unknown-unknown`), at the row's optimisation level, and the
  interpreter runs `run(eps_input)` on a fresh instance. *Reason:* Valgrind
  and QEMU user mode run on Linux only, and Unicorn needs a loader and a
  system-call layer for real executables; an interpreter of one small,
  specified IR runs identically on macOS arm64 and Linux x86, and its
  definition of "copied out" is in its own code. The instrumentation is
  pinned by the SHA-256 of `wasm_eps.py`, which every results file records.
- **What is counted.** ε is counted on the WebAssembly lowering of each
  kernel at the configuration's level, not on the native machine code
  whose energy is read. The two share the source and LLVM's optimisation
  passes and differ in instruction selection and register allocation, so ε
  is a count for the configuration, not for the native binary. The same
  module gives the same ε on every host, so ε measured on a Linux machine
  applies to a Mac with the same compiler versions.
- **Definition.** A *location* is an operand-stack slot, a local of a live
  call frame, a global, or a byte of linear memory. Every value is held as
  bytes, and every byte carries an *instance*, the identity of the byte
  value the instruction that made it produced. An instruction that moves
  or copies a value (local and global access, loads, stores, `select`,
  `i32.wrap_i64`, `i64.extend_i32_u`, `memory.copy`, `memory.fill`, call
  arguments and results) gives its destination the same instances; any
  other instruction gives its result fresh instances. A byte instance is
  *copied out* while another location still holds it. When a location is
  overwritten or released and held the last copy of an instance, the
  instance's 8 bits are erased.
- **Permanent bytes**, never counted: constants, the zero bytes of zero
  extension and of comparison results, the initial memory (data segments
  and zeros), initial globals, zero-initialised locals, and the kernel's
  argument, which the caller keeps. The returned value is copied out.
- **When.** A location's last copy is counted when the location is
  overwritten or released: a stack slot popped by an instruction that does
  not move it, the locals and leftover stack slots of a returning frame,
  the values a branch discards. The method counts dead values "when their
  storage is next reused"; in a stack machine a released slot is the next
  one reused, so the two differ only for storage still live when the run
  ends (memory and globals), which neither counts.
- **Bound.** Bytes made by sign extension are fresh, and a computing
  instruction's result is fresh even when it repeats an input, so ε is an
  upper bound, at byte granularity, on the information a run discards.
- **Coverage.** The integer instructions of WebAssembly 1.0, the
  sign-extension operators, multi-value block types, `memory.copy` and
  `memory.fill`. A module that uses floating point, SIMD, imports or table
  instructions other than `call_indirect` gives an error naming the
  instruction, and the row records ε as not counted.
- **Landauer bound.** ε·k_B·T·ln 2 with k_B = 1.380649×10⁻²³ J/K and
  T = 300 K, labelled *modelled*: no ambient temperature is read.
- **Tests.** `test_e4.py` checks the count on eleven tiny modules whose ε
  is worked out by hand in the test's comments, and checks that three C
  kernels give the reference value and the same ε twice.

### λ

All proxies are computed for every row; one is primary.

| Proxy | Definition |
|---|---|
| **`ir_nodes` (primary)** | WebAssembly instructions in all function bodies of the kernel's module, without each body's closing `end` |
| `wasm_bytes` | bytes of the module without its custom sections (names, debug information) |
| `wasm_zlib_bytes` | `wasm_bytes` compressed by zlib at level 9 (Python's `zlib`, whose version the results record) |
| `native_bytes`, `native_zlib_bytes` | the native executable's bytes, and compressed by zlib at level 9 |

*Reason for the primary:* it is computed from the same IR as ε, and the
native executable's size is dominated by the C runtime and, for Rust, the
standard library, which do not change with the kernel's lowering.

### Joules

| Host | Counter | Kind | Regime |
|---|---|---|---|
| Linux, x86 | RAPL through `/sys/class/powercap`: the sum of the top-level zones `intel-rapl:N` (sub-zones not read, `psys` skipped), counter `energy_uj`, one wraparound past `max_energy_range_uj` added back per read, wraps counted per zone | reported | B |
| macOS | `macmon pipe -i 100` (no root; macmon version recorded): `cpu_power + gpu_power + ane_power` (W, the average over each sample) times the span between measured sample timestamps, the cpu + gpu + ane boundary. Without macmon, `powermetrics` is used only when sudo is already cached (Apple silicon `processor.cpu_power` times `elapsed_ns`, the CPU clusters) | reported | B |
| any other | none | unmeasured | - |

- **Kind.** Both counters are *reported*: an on-chip energy model or
  counter read through the operating system, not a meter on the supply.
  *Measured* is used only for a wall or shunt meter, and *modelled* for
  figures computed from a model or a declared constant. Every results file
  records the counter's provenance: sampler, counter, interval, idle
  baseline, machine, boundary, start, end and wraps. The powermetrics key
  names are from its documentation and other tools' use of it [K]; the
  first metered Mac run records the keys of its first sample, so a
  different name shows in the results.
- **macmon windows.** One `macmon pipe` process runs for the whole run.
  Each sample covers the time from the previous sample's measured
  timestamp to its own; a window takes the share of every sample that
  overlaps it. (The powermetrics fallback stamps each sample when it
  arrives and takes it to cover the `elapsed_ns` before.) At 100 ms samples and 2 s runs, the window edges
  move each run's figure by up to about 5%.
- **Idle baseline.** The counter is read over 10 s of sleep before the
  builds and again after the energy runs. *Net* joules are gross joules
  minus the mean of the two idle powers times the run's seconds; compile
  energy subtracts the first idle power.
- **Runs.** For each row whose equivalence evidence holds, R is set so that
  one native run of R calls of `run(eps_input)` takes about 2 s (the first
  run of at least 0.2 s, scaled). Each row is run 5 times, in 5 rounds;
  each round runs every row once, in an order shuffled by the seed
  4 + round. The figure per row is the mean net joules per call with its
  95% Student t interval over the runs whose output equals the reference.
- **Compile-time energy.** Each native build is repeated 3 times under the
  counter, and the results record its seconds and gross and net joules;
  RESULTS.md sums the mean net joules per configuration.
- **No counter.** When the host has none, the harness still builds, checks
  the evidence and records ε and λ, writes every energy field as
  *unmeasured* with the reason, and exits normally. When a counter exists
  but cannot be read by the user, the harness prints the one line that
  grants access and exits with status 2 without writing.

### Declared analysis

- **Rows.** One per program and configuration with ε counted, λ computed
  and net joules per call; the response y is net joules per call.
- **Models.** Ordinary least squares with an intercept: y on ε and λ
  (`ir_nodes`); and y on the baseline. In regime B the baseline is seconds
  per call times the file's mean net power (total net joules over total
  seconds). With one constant power per file, this baseline explains the
  same share as seconds alone.
- **Share of variance explained.** *R²* is one minus the residual sum of
  squares over the total sum of squares. The share used is *adjusted R²*,
  1 − (1 − R²)(n − 1)/(n − k − 1) for n rows and k predictors, because it
  can fall below zero and so makes "the interval excludes zero" a test
  that can fail.
- **Interval.** 2000 resamples of whole programs with replacement (all six
  configurations of a drawn program together), seed 4; the 95% interval is
  the 2.5th and 97.5th percentiles. A draw whose design is singular is
  skipped and counted.
- **Rules.** ε and λ explain variance when the interval of their adjusted
  R² lies above zero. The baseline explains more than ε and λ when the
  interval of (baseline's adjusted R² minus theirs) lies above zero, and
  less when it lies below. For H4 in regime B, support needs the second
  interval above zero; a falsifier is that interval below zero.
- **Per file.** Each results file is analysed alone: the counters of a Mac
  and an x86 host have different boundaries, so their joules are not
  pooled.
- **Regimes A and C.** No host of this harness has a counter of either;
  RESULTS.md says so for every file.

## Running it

On a Linux x86 host with RAPL, after `sudo apt-get install clang lld` and
`rustup target add wasm32-unknown-unknown`:

```sh
cd compile/experiments/e4-energy
python3 e4.py run results/NAME.json
python3 report.py write
python3 report.py check
```

RAPL's `energy_uj` is readable only by root since Linux 5.10. When it is
not readable, `e4.py` prints the one line to run first,
`sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`, which lasts
until the next boot.

On a Mac (Apple silicon), after `brew install llvm lld` (Apple's clang has
no WebAssembly target; the harness uses Homebrew's clang for both builds
when it finds it) and `rustup target add wasm32-unknown-unknown`:

```sh
cd compile/experiments/e4-energy
python3 e4.py run results/NAME.json
python3 report.py write
python3 report.py check
```

with `brew install macmon` done first. macmon reads the counters without
root, so no `sudo` is needed. Without macmon (and without cached sudo for
the powermetrics fallback), energy is recorded as unmeasured.
Without Homebrew's clang, the C rows are built natively and their ε is
recorded as not counted, with the reason.

`python3 test_e4.py` runs the unit tests. `report.py check` fails when
`RESULTS.md` is not what `report.py write` prints from `results/*.json`.
A run takes about 2 minutes without a counter and about 25 minutes with
one (96 rows, 5 runs of 2 s each, idle windows and repeated builds).

The workflow
[`.github/workflows/compile-e4.yml`](../../../.github/workflows/compile-e4.yml)
runs the unit tests, checks `RESULTS.md`, and runs the harness on a
GitHub-hosted Ubuntu runner. Those runners expose no RAPL counter, so the
CI results hold ε and λ and no energy; they are uploaded as the artifact
`compile-e4-results`.

## Metrics

- ε (bits) and ε·k_B·T·ln 2 (joules, at the declared temperature) per call.
- λ in its declared unit per compiled program.
- Reported joules per call and per compilation, with intervals.
- Per regime: the share of variance in joules explained by ε and λ, and by
  the baseline, each with its interval.

## What it needs

- For regime B: an x86 host with RAPL counters, or a Mac with
  macmon. The machine of the committed results has no
  `/sys/class/powercap` interface and no wall meter, so its results file
  holds ε and λ only.
- An FPGA or in-order core with a supply meter for regime A, and a GPU with
  NVML for the GPU part of regime B; neither is in this harness.
- A simulator of entropy production for regime C.

## Limits

- ε is counted on the WebAssembly lowering, not on the native code whose
  energy is read.
- The equivalence evidence is output equality on declared inputs, weaker
  than E3's exact level.
- RAPL, macmon and powermetrics are reported counters; their accuracy against a
  supply meter is not established here.
- The corpus is small: 16 programs, so the bootstrap over programs has 16
  units.
