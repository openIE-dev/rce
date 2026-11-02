# E4: erasure and description length against measured energy (protocol)

Status: protocol only; nothing has been run. Tests H4 of
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

## Metrics

- ε (bits) and ε·k_B·T·ln 2 (joules, at the measured temperature) per run.
- λ in its declared unit per compiled program.
- Measured joules per run and per compilation, with intervals.
- Per regime: the share of variance in measured joules explained by ε and λ,
  and by the baseline, each with its interval.

## What it needs

- A host with RAPL energy counters or an external wall meter. The machine
  of E1's committed results has no `/sys/class/powercap` interface, so it
  cannot read RAPL, and no wall meter; E4 cannot run there.
- An FPGA or in-order core with a supply meter for regime A, and a GPU with
  NVML for the GPU part of regime B.
- A simulator of entropy production for regime C.
