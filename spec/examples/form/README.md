# Fixtures of the gate on `rce-form/1`

The modules in this directory exercise the form module of the checker
(`spec/checker/src/form/`, `rce form-check FORM [VECTORS]`) against the
canonical rules C1 to C11 and the gate rules FG1 to FG15 of
`spec/proposals/rce-form.md` (§10, §11).

- `make.py` writes every fixture and `expect.json`. It builds each module from
  the rules as the proposal states them, with an encoder of its own: canonical
  JSON (6.1.1), group and module keys, Merkle digests (C2), and the orders that
  C3, C5, C7 and C9 fix. A fixture that requires change breaks one rule and is canonical in
  every other respect. `python3 spec/examples/form/make.py --check` fails when
  a file here differs from what `make.py` writes.
- `expect.json` gives each fixture's outcome, written by hand in `make.py`
  from the rules, never from the checker: `applied`, or `requires change` with the rule
  the gate names first and, for a canonical rule, the rule its reason names.
  Its `rules` table names, for each of C1 to C11 and FG1 to FG15, a fixture
  the rule applies and one that requires change under it.
- `spec/checker/tests/form_gate.rs` runs the gate on every fixture and checks
  the table, and checks that decoding then encoding every applied fixture
  gives its own bytes (FG3).

| Fixture | What it holds | Outcome |
|---|---|---|
| `counter` | the counter of §10's worked example: `Inc` and `Dec` with their overflow guards inline, and the guard `n > 0` | applied |
| `counter.vectors` | admission vectors: `Inc, Inc, Dec` ends in state `[[1]]`; `Dec` at 0 is applied with the state unchanged (6.4) | applied |
| `types` | a declared record and kind (C8), an option field, a payload, an emitted float | applied |
| `group` | `even` and `odd` as one recursive group with declared bounds (C9, FG8) | applied |
| `loops` | Iterate with a measure and with a count bound, `list.get` under DF5, `sum.get` under DF4, a memo cell (FG6, FG8) | applied |
| `counter-c1`, `-c3` … `-c7`, `-c10`, `-c11`, `types-c8`, `group-c9` | each breaks one canonical rule | requires change under FG1 (C1, C10) or FG3 (the others) |
| `counter-fg1` … `-fg13`, `types-fg12`, `types-fg14`, `group-fg8`, `counter.wrong-vectors` | each breaks one gate rule | requires change under that rule |
| `cycle` | a region whose node reads its own output | requires change under FG4 (below) |

## Rules no module can break on its own

- **C2** is the Merkle digest that C3, C4, C7 and C9 order and merge by. It
  is a definition, not a check, so no module requires change under C2 alone.
- **FG9** (acyclic regions) follows from FG4, which the gate applies first:
  when every source precedes its consumer in the numbering, the region is
  acyclic. `cycle` is a region with a cycle, and it requires change under FG4.
- **C10** (no unobservable names) follows from C1: no field of the schema
  holds a name, and an unknown field fails decoding. `counter-c10` adds a
  `names` field and requires change under FG1, whose reason names both C1 and C10.

## How the module reads what the proposal leaves open

Where the proposal does not fix a rule exactly, the module reads it as
follows; each is a choice the proposal's text should settle.

- **Catalogue key (FG1).** §4 fixes no catalogue key yet; a module names the
  SHA-256 of the bytes of `spec/data/form-catalogue.json`.
- **Digests (C2).** A digest inside a digested array is written as its
  lower-case hexadecimal text.
- **Order of FG3 and FG4.** A region with a source that names no node, or
  with a cycle, has no canonical form (C5 numbers nodes topologically); FG3
  then leaves the module to FG4.
- **Naming the canonical rule (FG3).** The canonical form is reached by one
  pass per rule (C8, C11, C6, C7, C4, C3, C5, C9), repeated until no pass
  changes the module; the first pass that changes it names the rule. Bytes
  that decode to a canonical module but are not its canonical JSON name 6.1.1.
- **C11.** "A Halt with the runtime's trap text" names no definition and no
  text. The module inlines a call when the callee is not recursive and its
  body is: Bind nodes, one Apply of a domain test that is an entry, a Branch
  on that test whose arm 0 is exactly one Apply of a partial entry the test
  guards, on the same operands, and whose arm 1 is Bind and Apply nodes ending
  in a Halt; the body's outputs are the Branch's. DF3's `divisor-not-zero`,
  which is not an entry, is never inlined.
- **Declared types.** A type argument of kind `sum` or `tup` may be a declared
  kind or record, which stands for its declaration one level unfolded; type
  equality is otherwise by spelling.
- **Body representation of frame types (§3).** `res` is
  `sum(τ, tup(fields of each error)…)` in the order the type lists its errors;
  `chan` and `actor` have none, so a step over one requires change under FG10 or FG11.
- **FG5 and FG8 on a Branch selector.** FG5 requires `bool` (two arms) or an
  integer type; FG8 requires an integer selector's range to lie in 0 to n − 1.
- **FG6 on memo cells.** "Takes full" is a fact of a run. A cell counts as
  taken after `memo.get` except in arm 0 of a Branch on `sum.tag` of the value
  `memo.get` gave, where it was taken empty.
- **FG8 tests.** The tests the catalogue names in words are read as: an index,
  start or length not negative by its type, or by a Branch on `int.le(0, x)`
  (arm 1) or `int.lt(x, 0)` (arm 0); a length above zero by `int.lt(0, len)`
  (arm 1) or `int.eq(len, 0)` (arm 0); a divisor not zero by `int.eq(b, 0)`
  (arm 0). A measure's "i raised by exactly one" is `int.add` of i and the
  literal 1, through Branch arms that halt. A component of a parameter is an
  output of `sum.get`, `tup.get` or `tup.open` of it. A layered-measure
  certificate `["layers", L]` gives in `L[k][f]` the positions of member f
  for layer k, and every member of the cycle carries the same certificate.
- **FG11.** Names are not in the encoding, so "main names region Main and its
  `Start` event" is read as: the module has one state region, and main's
  inputs are its fields, the payload of 4.2 (`List[text]`) or of a lifted
  command (§9), and ω. The task events of 6.8 have no names in the proposal,
  so `Tick` is the only built-in event a step may name.
- **FG13.** `rce-cap/1` is named and not defined, so no capability has a
  signature and every module that names one requires change.
- **FG15.** `rce-vectors/1` does not fix an input's form, a case's outcome or
  how several inputs' outputs combine. An input is `{"type": i | "Tick" |
  "Start", "payload": [VALUE…]}`, the positional event of 6.1; each input
  starts with a fresh ω (§5, on a host); the outputs are the λ of the applied
  inputs in order; a kept input ends the case with outcome `kept:X`, X its
  trap text or its error's name, and otherwise the outcome is `done`; `state`
  is the SHA-256 of the positional state. `observations` must be empty, since
  the reference trace's names come from the concept record, which the gate
  does not read.
- **Values of the frame (FG10).** §3 and 6.1 state no positional form of a
  result, a channel or an actor handle, so a field of one of those types
  requires change. A step's `deadline` is decoded and not evaluated.
