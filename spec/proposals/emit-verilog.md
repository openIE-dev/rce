# Proposal: the Verilog target, first slice

Status: open; the slice below is implemented (`rce emit verilog`, the
ledger row of 2026-11-01). Derived from 10.5 (`rce emit`), 6.3 (the
numeric contract), 6.4 (δ), 6.7 (bounded loops), 6.1 and 10.9 (the log D
and its hashes), the Tier 3 row of `emit-targets.md` ("one step per
clock; bounded state only; the Esterel size ceiling applies"), XF-12 of
`one-library-every-fabric.md` and the hardware rows of ONE-26 in
`one-emitter.md`.

In plain words: `rce emit verilog FILE` prints the step function δ of a
program as one synthesizable Verilog module (IEEE 1364-2005, "Verilog"
below). The module keeps the state in registers and takes one step per
clock. A program outside the subset of EV-1 is not printed; the answer
names the reason and the line. Simulators run the module on the
program's examples, and the trace computed from the register dumps is
compared with the reference trace line for line.

## Terms

- **Module**: a Verilog `module`, the unit a simulator runs and a
  synthesis tool turns into gates.
- **Register**: a value the module holds from one clock edge to the next.
- **Clock edge**: the rising edge of the input `clk`; one step is taken at
  each edge where the input `ev_valid` is high.
- **Event bus**: the inputs `ev_valid`, `ev_tag` (the index of the event
  among the module's declared events, as the positional form of 6.1 numbers
  them) and `ev_payload` (the event's fields).
- **Test bench**: a Verilog module that drives the module under test and
  prints what it does; it is not part of the printed module.
- **Simulator**: a program that runs Verilog. Two are used: Icarus Verilog
  (`iverilog`, an interpreter of the compiled design) and Verilator (which
  compiles the design to C++).
- **Synthesis**: turning the module into a netlist of logic cells. Yosys's
  generic `synth` is used, which maps to Yosys's own cell library, not to a
  device.
- **Unroll**: replace a loop by copies of its body, one per iteration.
- **Inline**: replace a call by the body of the function called.

## EV-1 The subset

A program is printed when all of the following hold; otherwise the answer
is `requires change for target verilog: REASON, line N`, the status is 1,
and nothing is printed on standard output.

1. It passes `rce check`.
2. It declares at least one `state` region, and no `fn main`, `actor`,
   `extern` function or event, `impl`, `trait`, `protocol`, `node`,
   `supervisor`, `use`, `replicated state`, `interrupt event` or step
   `deadline`.
3. Every state field and every event field has one of these types: a
   fixed-width integer (`i8` to `i64`, `u8` to `u64`, `usize`), `bool`, a
   `kind` whose variants carry no fields, or a fixed-size array `[T; n]`
   of these with a constant n up to 4,096. Arrays do not nest.
4. The steps use only: `let`, `var`, assignment and compound assignment
   to a local, a field or an array element; `if`, `match` on integers,
   bools and kinds (literal, variant, range, alternative, wildcard and
   name patterns, with guards); `while … bound n` and `loop bound n` with
   a constant n; `for` over a range with constant ends or over an array;
   `break` and `continue`; `emit` of a scalar; `fail E` and `fail E(…)`;
   `try x.checked_add|sub|mul|div(y) else E`; calls of the program's
   functions; the integer operators of 6.3 N2, comparisons, `and`, `or`,
   `not`, `if` expressions, casts between integer types and from `bool`,
   indexing, list literals and `[x; n]`; and the methods `abs`,
   `wrapping_add|sub|mul` and `saturating_add|sub|mul`.
5. Functions called from a step are not recursive, not generic, take no
   `&` or `&mut` parameter, and do not emit. They are inlined, so a call
   costs no clock.
6. There is no text, list, set, map, float, option value (other than the
   operand of `try … else` above), tuple, struct, closure, channel or task.

Kept outcomes are outputs (EV-5), not a reason to refuse.

## EV-2 Static integer types

The reference evaluator gives every integer value the name of its type,
and an untyped literal the name `lit`, which is checked as `i64` (6.3).
An operation checks its result against the type of its left operand
(the right one when the left is `lit`). The emitter follows these names
along every path of the step. Where two paths join (after an `if`, a
`match`, a loop iteration or a skipped statement) a local must carry the
same name on both paths; otherwise the type of the local depends on the
run, and the program requires change with the local's name. The same
holds for the two arms of an `if` expression.

## EV-3 Outputs and emitted values

The outputs are registered: they hold the outcome of the step taken at the
last clock edge where `ev_valid` was high.

| Port | Width | Meaning |
|---|---|---|
| `out_valid` | 1 | a step was taken at the last edge |
| `out_applied` | 1 | 1 when the event applied, 0 when it was kept |
| `out_error` | ⌈log2(e + 1)⌉ | 0, or 1 + the index of the declared error a `fail` kept the state with |
| `out_trap` | ⌈log2(t + 1)⌉ | 0, or 1 + the index of the trap in the interface's trap table |
| `out_trap_value` | at least 2 | the value the trap names, signed: V of "overflow: V is outside T", K of "index K out of range" |
| `out_emit_count` | | the number of values the step emitted (0 when kept) |
| `out_emit_type`, `out_emit_value` | slots × width | per slot, the kind of the value (int, bool or a kind) and its bits |
| `state` | sum of the field widths | the state registers, first field at bit 0 |

A step emits at most 64 values. Emitted values are scalars.

## EV-4 Bounded unrolling and the size ceiling

Every loop is unrolled to its bound: `while c bound n` becomes n copies of
the body, each taken while the condition holds, and a check that traps with
"loop bound" when the condition still holds after them, as 6.7 says. A
`for` over a constant range or an array becomes one copy per item. A
bound or range end that is not a constant requires change. The logic of a
module's steps may unroll to at most 20,000 Verilog statements; a program
above that requires change. This is the size ceiling of the Tier 3 row: a
circuit is finite, so the program's step must have a finite unrolling.

## EV-5 The module

`rce emit verilog FILE` prints:

- a first line naming the program hash and the checker commit (CC-4), as
  the other emitters do;
- comment lines describing the interface, and one line
  `// rce-hdl/1 JSON`: the bus and state layout (field names, types,
  offsets and widths), the error table, the trap table, the kinds of
  emitted values, the kinds' variants, and the program's example events
  (each event's index and field values);
- one module named `rce_` followed by the region names, with the inputs
  `clk`, `rst` (synchronous, active high) and the event bus, and the
  outputs of EV-3.

Inside, the state is one register per field, loaded with the initial state
s0 at reset. s0 is computed by the reference evaluator when the module is
printed; the module holds it as constants. One combinational block computes
δ on the current state and the event on the bus: regions in declaration
order, each region's steps in source order, the first whose pattern
matches and whose guard holds taken, on the pre-state (6.4). Every integer
operation is computed at a width that holds its exact result (for
example, the sum of two `u64` values at 66 bits and their product at 130
bits) and is then checked against the range of its type, so an overflow,
a division by zero, an index out of range, a shift out of range, a match
with no arm and a loop bound keep the state exactly where the reference
evaluator keeps it, with the same reason. A `fail` keeps it with the
declared error. Each field is then checked against its declared type, as
6.4 step 3 says. At the clock edge, the state registers take the next
state only when the step applied.

## EV-6 The test bench and the harness

`spec/checker/hdl/rce_tb.v` is a Verilog test bench that reads an event
stream (rce-hdl-stream/1) from a file: a record starting with the byte 0
resets the module and dumps its state registers; a record starting with
the byte 1 holds one event's tag and payload, least significant byte
first, which the bench drives on the bus for one clock before dumping
every output and the state registers in hexadecimal. The bench computes
nothing.

`spec/checker/hdl/bench.py` writes the program's example events as one
such stream (a reset before each example), runs a simulator on the
bench and the module, and prints the trace of every example from the
dumps: the line `example LINE`, then the lines that an emitted program
prints for that example under `trace fuel-trace` (10.5,
`emit::trace_example`). It computes each state hash outside the hardware:
the positional form of the dumped state (6.1) as canonical JSON (6.1.1),
then SHA-256, then each entry of D and its hash, chained from the
genesis. One line of the block is not printed: the last, `fuel N`, which
counts the reference evaluator's steps (6.7) and has no counterpart in a
circuit, where a step is one clock. The words are positional:
`bench.py iverilog|verilator MODULE.v WORKDIR`, `bench.py replay MODULE.v
DUMPFILE` (the trace from a dump written earlier) and `bench.py yosys
MODULE.v WORKDIR` (synthesis and cell counts).

## EV-7 Tests

`spec/checker/tests/emit_verilog.rs` holds them.

- Run everywhere: two emits of each program give the same bytes; each
  module of six programs equals its golden text in
  `spec/checker/tests/verilog/golden/` (all but the first line); the
  programs of `spec/examples/` that the subset admits are the six listed
  in the test; ten programs outside the subset each get their reason and
  line, status 1 and no output; and the harness, given a dump that holds
  the reference evaluator's states and outcomes in the module's layout,
  prints the reference's trace for every program. The last checks the
  layout, the tables and the harness's hashing, not the circuit.
- Run by `.github/workflows/verilog.yml`, which installs Icarus Verilog,
  Verilator and Yosys with the runner's package manager and prints their
  versions: every program (the counter of `01_counter.rce`, the door of
  `30_state_machine_guarded.rce`, the flag game `tests/law/flag_walls.rce`,
  the key handler `tests/law/keys.rce`, `tests/verilog/overflow.rce`,
  whose examples keep a step on an overflow and on a declared error,
  `tests/verilog/regfile.rce`, whose examples keep steps on an index out
  of range, a division by zero, a shift out of range and an `i8`
  overflow, and the other admitted examples) is simulated by each
  simulator, and the harness's trace must equal the reference's line for
  line; then Yosys synthesizes each module and the test prints its cell
  counts. A simulator or synthesis tool that does not start fails the
  test.

## EV-8 What this slice does not claim

- A passing run is a simulation result and a synthesis result. No module
  has run on a board, and no energy figure is measured or modelled here
  (XF-12, `energy-calibration.md`).
- State whose size is not static (lists, maps, text) is not held in
  addressed memory yet; XF-12 and ONE-26 describe that next step. Until
  then such a program requires change.
- The Yosys cell counts are for Yosys's generic cell library, not for a
  device, and nothing is optimised beyond what `synth` does.
- The module is printed by its own emitter in `spec/checker/src/emit/`,
  not by the one emitter of ONE-1.
