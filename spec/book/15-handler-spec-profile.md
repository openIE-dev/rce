## Appendix D. The engine profile: handler specs and fits (normative)

This appendix defines the second input language of the engine: the
*handler spec*, a line-oriented `.rce` file that `rce check`, `rce
compile`, `rce run`, `rce auto` and `rce form` read (`src/spec/parser.rs`),
and the JSON *fit* that says what each node of the compiled graph does
(`src/form/fit.rs`). It is a profile tied to the language spec, not a
second programming language: a handler spec declares a graph and its
contracts; the behaviour of each node is a fit entry, and the compiled
result is a Form P = (S, E, δ, D, λ) as in 6.1, with S, E and δ given by
D.6. A file is a handler spec when its first non-blank, non-comment line
starts with `spec`; otherwise it is a module of sections 2 to 10. Example:

```
spec "orders-form" v0.1.0
reason "Demo for the Form compiler" by "david"
currency max_age=24h
```

**Implementation status.** This appendix describes the engine profile. The
code it names (`src/spec/`, `src/form/`) belongs to the engine crate, which
is not in this repository. The reference checker in `spec/checker` has no
`src/spec` or `src/form`, `rce-spec-check` does not read handler specs, and
no file under `spec/checker/src`, `spec/checker/tests`,
`spec/checker/examples` or `spec/examples` mentions a W-HS rule. So nothing
in this appendix is checked by the tools published here; the rules below are
a specification of the engine, and its tests are not published.

### D.1 Lexical rules

- The file is UTF-8 text read line by line. A line that is blank after
  trimming, or whose first non-space characters are `#` or `//`, is
  skipped.
- A *top-level line* starts in column 0. A *body line* starts with a space
  or a tab and belongs to the nearest preceding top-level block. A blank
  line ends a body.
- Tokens are separated by white space. A token that starts with `"` runs
  to the next `"` (no escapes); its value is the text between the quotes.
- Durations: hours are `N` or `Nh`; milliseconds are `Nms`, `Ns`
  (seconds) or `N`. N is a decimal integer.

### D.2 Grammar

```
File        = SpecLine { TopLine } .
SpecLine    = "spec" STRING VERSION .                VERSION = [ "v" ] MAJOR "." MINOR "." PATCH
TopLine     = ReasonLine | CurrencyLine | Objective | Constraint | Acceptance
            | Deliverable | Role | Dependency | Escalation | Flow | ChangeControl .
ReasonLine  = "reason" STRING [ "by" STRING ] .
CurrencyLine= "currency" { "max_age=" HOURS | "last_checked=" RFC3339
            | "toolchain." NAME "=" VALUE | "dep." NAME "=" VALUE } .
Objective   = "objective" ID STRING Body( "metric" NAME "target" VALUE | ReasonLine ) .
Constraint  = "constraint" ID ( "substrate" | "performance" | "regulatory"
            | "resource" | "compatibility" ) STRING Body( ReasonLine ) .
Acceptance  = "acceptance" ID "validates" ID
              Body( "property" STRING | "error_budget" NUMBER | ReasonLine ) .
Deliverable = "deliverable" ID "fulfills" ID
              Body( "schema" STRING | "description" STRING | "blast_radius" NAME | ReasonLine ) .
Role        = "role" ID STRING ( "execution_node" | "coordinator" | "ai_foreman"
            | "order_giver" ) Body( "blast_radius" NAME | ReasonLine ) .
Dependency  = "dependency" ID ( "node" | "crate" | "service" | "data" ) STRING
              Body( "input_schema" JSON | ReasonLine ) .
Escalation  = "escalation" ID { "from" ENDPOINT | "to" ENDPOINT | "timeout" MS }
              Body( ReasonLine ) .
Flow        = "flow" ID "from" ENDPOINT "to" ENDPOINT Body( ReasonLine ) .
ChangeControl = "change_control" { "approvers" LIST | "review_period" HOURS | "frozen" } .
```

`Body(X)` is zero or more body lines each of the form X. `error_budget`
must lie strictly between 0 and 1. `input_schema` is JSON that may span
body lines until its brackets balance. An unknown top-level keyword, an
unknown constraint, role or dependency kind, a flow without `from` and
`to`, a missing `spec` line, an `error_budget` out of range, a
`last_checked` that is not RFC 3339 and invalid `input_schema` JSON are
refused at parse time. The reference parser ignores unknown body keywords
and unknown `change_control` words, and reads an unparsable duration as
its default (24 h, 30 000 ms).

### D.3 Meaning of each declaration

- `spec NAME VERSION`: the name and version of the spec. The version is
  recorded in the compiled graph's metadata (`spec_version`) and in the
  Form IR (`FormIr.version`). The version is not yet checked.
- `reason WHY by AUTHOR`: why an artifact exists. A top-level reason
  belongs to the spec; a body reason belongs to its block. A reason is
  recorded on the node or edge it explains and enters the log with every
  mutation that creates it. Reasons are data: they never change δ.
- `currency`: when the spec's toolchain and dependencies were last
  verified (`last_checked`), the versions seen then, and the maximum age
  `max_age` in hours. Parsing is not a check: without `last_checked` the
  record is *unchecked*. `rce currency FILE --mark` records a check by
  writing `last_checked` back into the file. A record is stale when it was
  never checked or is older than `max_age`; the promotion gate records a
  stale spec, and blocks on it when the engine's `enforce_currency`
  policy is on (off by default).
- `objective`, `constraint`, `acceptance`: requirements. Acceptance
  properties and their error budget are not inputs of δ: no step reads
  them. They decide whether a candidate passes the promotion gate, and
  so which δ is installed next (see the versioning of δ below).
- `deliverable D` compiles to the node `del_D` (an executor and a sink),
  `role R` to `role_R` (kind executor, decision, escalation or source),
  `dependency P` to `dep_P` (a source). `flow F from A to B` compiles to
  the edge `flow_F`; `escalation E from A to B timeout t` to `esc_E`. An
  endpoint names a node id, a node name or a declared id; an unknown flow
  endpoint is refused by the compiler, an unknown escalation endpoint is
  a diagnostic.
- `change_control`: who may approve changes to the spec, the minimum
  review period, and whether the spec is frozen. These bind the engine's
  promotion path (spec mutations are logged and need an approver). They
  are not inputs of δ; they govern which δ may be installed, and by
  whom.

δ is versioned. A promotion that passes the gate is committed as a
RESOLVE entry of D (`PromoteEscalation` with a `resolve` stamp) that
records the log height h of the entry, the version of δ at its node
(raised by one at each RESOLVE for that node), and the sha256 digests of
the installed HandlerSpec and of the fit evidence the gate accepted. The
new handler governs every entry after h, and every entry before h was
computed under the previous one. A tick is one step and no input can be
committed inside it, so a step in flight when a RESOLVE is requested
finishes under the δ it started with. The state after n entries is the
fold of D under the δ in force at each height: s(i+1) = δ_v(i)(s(i),
D(i)), where v(i) is the version set by the last RESOLVE before i. A
fold applies each RESOLVE at the height it records and refuses one whose
height, version or spec digest differs from the fold's; the fit digest
is recorded and not refolded, since the fit evidence is not in D. The
handler installed is the one the entry carries; only an opaque handler
(no spec) runs as the live object, and every result it returns is in D
(`ExternalResult`).

### D.4 Fits

A fit is a JSON object:

```
{ "handlers": { NODE: HandlerSpec | {"kind": "term", "term": TERM}
                       | {"kind": "recorded", "contract": C, "table": {SHA256: RESULT}} },
  "types":    { SOURCE_NODE: TYPE },
  "keys":     { SINK_NODE: [COLUMN, ...] },
  "reads":    { SINK_NODE: {"key": [FIELD, ...], "into": FIELD} },
  "joins":    { SINK_NODE: [ {"relation": REL, "key": [FIELD, ...], "into": FIELD,
                              "scan": BOOL}, ... ] } }
```

An object with any of the five fields above is a full fit, even when it
holds only `reads` or `joins`; any other object is a bare map of handlers
and is read as `handlers` alone. NODE is a
compiled node id (`role_R1`, `dep_DEP1`, `del_D1`). `types` gives the
input contract of a source; `keys` the key of a sink's relation (default:
the input number `seq` and the index `n` within the step). A recorded
entry is an oracle: a recorded input replays its result and an unseen one
escalates as "oracle miss".

### D.5 Reads, joins and scans

Each is applied before the body of every rule that writes the sink, to
the state at the event's entry (pre-state, as in 6.4), in this order:
the read, then the joins in list order.

- `reads[S] = {key, into}`: the row of S's own relation whose key is the
  payload fields `key` (in key-column order) is put into the input as
  field `into`, or null when there is none. The relation must exist and
  its key must have as many columns as `key` names.
- `joins[S]`, an entry without `scan`: the row of relation `relation`
  keyed by the payload fields `key`, or null, into field `into`. The
  relation must exist and its key arity must match.
- an entry with `"scan": true`: every row of `relation` as an array, in
  canonical key order (code point order of the canonical JSON key);
  `key` is ignored.
- A payload that is not a record is passed unchanged. Joined rows and
  scans are typed as JSON. Each read or join is covered by the rule's
  SHA-256 id, and a rule with a read or a join is not monotone (CALM,
  `form/laws.rs`). A `joins` entry for a sink no rule writes is refused.
- Every target (reference δ, Rust, Python, JavaScript for XState and
  SCXML) applies them identically.

**CALM tags (normative).** Each rule's `monotone` tag is derived by
`form::laws::calm_reason`, never declared, and `laws::check` refuses an
IR whose tag differs. A rule is monotone only when it has no resolve, no
read, no join and no scan. The reasons, by derivation:

| Rule has | Tag | Why |
|---|---|---|
| none of the below | monotone | its writes only add keys and grow counters |
| resolve | not monotone | it deletes escalation rows |
| read | not monotone | read-modify-write of its own sink |
| keyed join | not monotone | the joined row may be null now and present later, or overwritten |
| scan | not monotone | it observes the whole relation, so also what is absent |

The tag is conservative (sound, not complete): a join against an
insert-only relation would be monotone in principle, but the profile
does not prove insert-only, so it is not claimed. A Form is
coordination-free (`laws::coordination_free`) iff every rule is
monotone. Because every read, join and scan sees the entry state, the
post-state of an input event does not depend on the order in which
rules for different sinks run; only the output order does, and that is
fixed by content hash (law 2).

### D.6 The compiled Form

Σc has one region per node, `Ready | Escalated`. Σd has the logical clock,
an input count and the typed relations of the sinks plus `escalation` and
`error`. E is `IN_<source>`, `TICK` and `RESOLVE_<node>`. An input event
normalizes the payload (6.3 N1), checks the source's contract (a mismatch
is a `reject` output and an `error` row, not a refusal of the event) and
runs the rules in content-hash order. A `timeout` is converted to ticks of
a nominal 100 ms, rounded up; the clock is logical (6.3 N11). D is the
canonical log of 6.1: entries {seq, event, prev_hash, state_hash, hash}
chained by SHA-256. Every event is applied (no `outcome` field is
needed), since a contract mismatch is an output, not a refusal. The numeric
edges are those of 6.3, with one recorded difference: in this profile an
integer above i64::MAX is "integer out of range" (D.10).

### D.7 Relation to the language

A handler spec and a module of sections 2 to 10 both denote a Form, but
the profile has no expressions of its own: node behaviour comes only
from fits. A fit `term` is a term of the Form IR (`form/ir.rs`), not an
RCE expression. A function of a module is used as a fit through its
lowering (`rce index`, `spec/checker/src/brick_lower.rs`): a function
f(p1: T1, ..., pn: Tn) -> R lowers to the term t with
t({"p1": v1, ..., "pn": vn}) = f(v1, ..., vn), values in the Form's JSON
embedding (lists and tuples as arrays, structs as objects). A trap of f is
an error of t at the node `module.f`. A function that calls others lowers
to a Form IR format 2 program (`spec/proposals/form-ir-2.md`): each callee
is a definition keyed by the SHA-256 of its term's canonical JSON, and a
call is a `ref` to that key; a function that calls nothing lowers to a bare
format 1 term.

Statements lower store-passing: each is a term from an environment record
(the variables in scope and the control fields `$done`, `$ret`, `$brk`,
`$cnt`) to the next one, skipped once the function has returned or the
iteration has been left. `let`, `var`, assignment (also compound and to a
list item), `push` and `pop` statements, `if`, `while c bound n` (the
iteration count traps past n, after the condition holds, as in 6.7),
`loop`, `for x in e` (a range or a list), `break`, `continue`, `return`
anywhere and `check` lower. Expressions: literals, constants (their
values, by the reference evaluator), arithmetic (`+` as `plus`: integers
add, texts and lists concatenate), bit operations (`band`, `bor`, `bxor`,
`shl`, `shr`, `bnot`; a shift amount outside 0..64 is an error, as it
traps), comparisons, `and` and `or` (as `cond`), `if`, list and tuple
literals, `[v; n]`, indexing (`at`: a text is read as its UTF-8 bytes, as
in 6.3; a negative index traps), fields, struct construction, the methods
`len` (`size`, bytes of a text), `chars`, `tostr`, `toint`, `upper`,
`lower`, `contains`, `startswith`, `endswith`, `split`, `join`, casts to
i64 and usize, shared borrows, `copy`, and calls of other lowerable
functions (by `ref`). A function with `&mut` parameters lowers by
state-passing: its term returns `[result, final values of the &mut
parameters in parameter order]` (the result is null for a function
without one), and a call of it, as a statement, a `let` or `var`
initializer or the right side of `=`, writes those values back to the
borrowed places; such a call inside a larger expression is refused.
`match` lowers arm by arm (variants by their `$v` tag, `some`/`none`,
literals, ranges, tuples, alternatives, guards; no arm traps), and a
variant value is its fields with `$v`. A function with result `T ! E`
lowers with its errors as values: `fail e` returns `{"$err": e}`, `try x
[else e]` (as a `let`/`var` initializer, an assignment's right side, a
statement or a `return` value) tests `x` with `has(x, "$err")` (`none`
for an optional) and returns the error, and `x catch |e| h` tests the
same and evaluates `h`; a call of such a function anywhere else is
refused. A recursive group (a cycle of the call graph) lowers to one
`fix` term of its members' bodies, a call inside it being `rec` of the
callee's index; `bound n` is a counter of active calls passed by every
`rec`, trapping past n as in 6.7. The evaluator bounds `rec` nesting at
20000 (the reference evaluator's nesting limit) and calls per evaluation
at 50000000 (its fuel); past either is an error, and an input on which
the reference evaluator meets its own limits gives no vector. Integers
are i64 or usize (a usize variable that would become negative traps);
floats, narrower integers and closures are refused. A generic function
lowers to one term for every instance, since the IR is untyped over JSON.
A term over 1 MiB of JSON, not counting its definitions, or a program
nested over 1000 levels is refused (catalog readers parse past JSON
libraries' usual 128 levels, with a stated bound). Every refusal is recorded in the
catalog with its line and construct, never approximated.

`rce index --vectors N` records the reference evaluator's results on N
generated inputs per brick; `rce-brick-verify` (`src/form/brick.rs`)
evaluates each term on them and reports agreement. Agreement is tested on
vectors (`spec/checker/tests/brick.rs`), not proved.

### D.8 Lift guarantees (normative)

A brownfield lift (`rce scan DIR --form`, `brownfield/lift.rs`) reads
existing Rust, Python, TypeScript or Go into a Form schema. A lift gives
exactly these guarantees, and no other is claimed:

- L1 Labeled: every relation, region, event and transition carries a
  label that starts with "unproven"; a lifted element is never presented
  as checked.
- L2 Fail closed: a guard the lift cannot read is opaque, and
  `Lifted::step` refuses to step through it rather than guess.
- L3 Generated code agrees with the lift: code emitted from a lifted
  machine (`--emit-machine`) steps as `Lifted::step`.
- L4 Evidence, not proof: agreement of the lift with the original
  program is tested on seeded runs of the original; its level is
  tested (8). A lift is an
  input to a gate, never an authority: what it proposes is admitted only
  through the checker and the gate, like any other proposal.

### D.9 Rules (normative)

These are checked by `crate::spec::rules` (`src/spec/rules.rs`) in the engine
crate (see the implementation status above), run by the engine's `rce check`,
and each diagnostic names its rule.

- W-HS-VER-1: the `spec` version is not `MAJOR.MINOR.PATCH` of decimal
  integers (an optional leading `v` is allowed).
- W-HS-VER-2 (compatibility, between an old and a new revision of one
  spec, `rce diff`): the version must not decrease; a revision that
  removes or modifies a declaration must increase MAJOR, or MINOR when
  MAJOR is 0; a revision that only adds declarations must increase at
  least MINOR; a revision with no change may keep the version. Flows are
  declarations here: a flow whose endpoints change is modified.
- W-HS-CUR-1: a `max_age` that is not a positive integer number of hours
  (`N` or `Nh`, N ≥ 1).
- W-HS-CUR-2 (staleness): at check time `now`, the currency record is
  stale when it was never checked or `now - last_checked` exceeds
  `max_age` hours. A stale spec is refused by `rce check --strict` and by
  the promotion gate when the engine's `enforce_currency` policy is on;
  otherwise it is reported as a warning naming W-HS-CUR-2. This is the one
  rule of the appendix whose verdict depends on a time given as an input
  (see D.10).
- W-HS-REASON-1: the spec, or a declaration that compiles to a node or an
  edge (deliverable, role, dependency, flow, escalation), has no `reason`
  line, or a reason whose text is empty or `unspecified`.
- W-HS-DUR-1: a duration that is not a decimal integer with an allowed
  suffix (D.1).
- W-HS-KW-1: an unknown body keyword or `change_control` word.

### D.10 Recorded differences and open items

- Integers in [2^63, 2^64) are u64 values, as in the language (6.3 N2):
  the u64 operators compute on them exactly, comparisons, `sort`,
  `lmin`, `lmax`, `min` and `max` order them exactly, and the f64 and
  f32 operators convert them to f64. The signed operators refuse them
  ("integer out of range"), which no well-typed program reaches, since a
  signed value is never at or above 2^63. A JSON integer beyond u64 is
  read as a float (`form::num`); no integer type of the language holds
  one.
- W-HS-CUR-2 takes `now`. 10.1 item 1 says the subcommands of section 10
  read no clock; the staleness check is outside them, but a verdict that
  depends on `now` is a function of its recorded inputs only if `now` is
  recorded with the verdict. Whether it must be is not decided.
- The cost model of extraction (est_j) and the gate's certificate are
  outside this appendix, except the break-even receipt of D.11.
- `InstallHandler` and `RetireHandler` also change δ at the height of
  their entry and carry the same RESOLVE stamp as a promotion: height,
  the next δ version at the node (versions count every install,
  promotion and retirement there), the digest of the installed spec
  (for a retirement, the digest of the stub, distinct from an opaque
  install) and a fit digest. A fold refuses one whose stamp does not
  match. Entries written before the stamp existed carry none and fold
  as before.

### D.11 Promotion break-even (receipt)

Each promotion record (promoted or refused) carries a break-even
receipt `break_even` with three terms and n\* =
J_build / (J_model_path - J_handler), the number of inputs after which
the promoted handler has paid back its build:

- J_build: the est_j of the gear resolution that wrote the handler plus
  the coverage shadow run of the promotion path. Static checks (proof,
  kernel) are not priced and not included, so J_build and n\* are lower
  bounds.
- J_model_path: joules per input on the model path, the est_j of one
  gear resolution (the price the retirement rule uses for savings).
- J_handler: joules per input of the handler, the shadow run's priced
  operations per sample plus one dispatch.

Each term is {`j`, `basis`} with basis one of `est` (a model of the
work, including a model provider's own figure), `reported` (an OS or
on-chip energy counter: RAPL, NVML, IOReport), `measured` (a wall or
shunt meter on a named rail) or `unpriced` (no figure; `j` is null,
never 0). The terms the engine fills today are est or unpriced: a gear
receipt's `est_j` is an estimate by schema and is never relabeled from
its free text. A figure
without a basis is refused when a record is read, and the type that
holds a term has no constructor from a bare number. `n_star` is
{`kind`: `unpriced`} when any term is unpriced (it is never computed
from the other terms or from a default), {`kind`: `never`} when
J_model_path - J_handler <= 0, and otherwise {`kind`: `value`, `n`,
`inputs` = ⌈n⌉}; its basis is the weakest of the three
(unpriced < est < reported < measured). The promotion path's last step
records the same figures as text. Implemented in
`src/runtime/break_even.rs`, tested there and in the engine's promotion
tests.

**Readings: provenance, boundary and amortization (normative; P4, P5,
P6 of the receipt field proposals of 13 Oct 2026).** These rules apply to every
term labeled `reported` or `measured`, in every receipt that holds one,
and are checked where a term is built and where a record is read:

1. A reading carries `provenance` = {`sampler`, `counter`, `interval_s`,
   `idle_baseline_w`, `machine`, `boundary`}: the four names non-empty,
   `interval_s` finite and > 0, `idle_baseline_w` finite and >= 0 (0 when
   no baseline was subtracted). `boundary` says what the joules include
   (`package`, `package+dram`, `rail:VDD_GPU`). A reading without a valid
   provenance is refused; unknown fields in it are refused. Derivation:
   the label `reported` or `measured` claims that something was read; a
   reading that cannot be reproduced or compared (which counter, over what
   interval, against which baseline, on which machine, inside which
   boundary) does not support that claim, so a term that lacks it is not
   a reading, and labeling it one would be unsound.
2. A term labeled `est` or `unpriced` carries no `provenance` (nothing was
   read); one that does is refused.
3. A reading carries `amortization`, one of {`kind`: `per_call`},
   {`kind`: `batch_mean`, `n` >= 1}, {`kind`: `per_solved_task`}. An
   `est` term may carry one; the engine's own estimates omit it.
4. Figures of different kinds are never added or subtracted: a sum of two
   terms whose `amortization` differs (absent counts as its own kind) is
   refused, and so is a break-even whose J_model_path and J_handler differ
   in kind (n\* divides by their difference) or whose J_build is a mean
   (`batch_mean` or `per_solved_task`; J_build is a one-off total).
   Derivation: a batch mean added to a per-call figure, or a per-task
   figure subtracted from a per-input one, is arithmetic on different
   units; no result of it is a figure of either kind.
5. A sum of two readings keeps the weaker label only when both come from
   the same sampler, counter, machine and boundary (the intervals add);
   otherwise it is refused. A sum with an `est` term is `est` with no
   provenance (unpriced < est < reported < measured, as for n\*).
6. A stored break-even receipt is re-derived from its terms when read; a
   stored n\* that disagrees with them is refused.

The `est` wire form is unchanged (`{"j": 2.5, "basis": "est"}`), so no
stored receipt changes meaning; no receipt in the repository holds a
reading today. Implemented by `Term`, `Provenance` and `Amortization` in
`src/runtime/break_even.rs` (no constructor makes a reading without
provenance and amortization), tested there (`p4_*`, `p6_*`).

The receipt is a measurement record. It is not an input to δ, to any
check, or to the decision to promote, so it does not change the meaning
of any program or handler spec in any language version (C.15:
receipts are version-independent).

**Read receipt.** Which compiled node id does `deliverable D1` become? (for example role_R1)

<!-- body-sha256: 375c3aa0d3316d41bda6c6f4b3b948717cd752f06a5a96302f4e15d45b00038f -->
