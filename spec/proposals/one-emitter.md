# Proposal: one emitter, and each language as one table

Status: proposal. Owner: David Charlot. Derived from T9 (`spec/RCE-SPEC.md:2402-2412`), 10.5 (`:3066-3200`), 6.1 to 6.8, the numeric contract of 6.3, C.6, C.8 and C.15 of the specification; from `emit-targets.md` sections 0 to 4, `form-ir-2.md` and `one-library-every-fabric.md` (XF-3, XF-5, XF-6, XF-11, XF-12, XF-13); and from three read-only inventories of `openIE-dev/rce` at `d91a9fc8` (25 October 2026): the four emitters and their runtimes, the Form IR (the intermediate representation of `rce emit form`), and prior systems that print many languages from one specification. Nothing in this proposal has been built. Every figure about the code is from that commit. A statement about another system marked [K] is taken from knowledge and was not checked in this work; one marked (search) comes from a search summary whose page was not opened.

In plain words:

1. RCE emits every language: languages of the past, of the present and of the future. A language whose integers wrap, whose numbers are decimal, whose floats are 32 bits wide, or which has no maps, no recursion or no byte strings is a full target. What it lacks is supplied in that language, by code printed in that language.
2. Adding a language is adding one data file. Nothing in the design is a flag, a macro, a directive or a choice of version.
3. At `d91a9fc8` four languages are emitted, each by a hand-written emitter path and a hand-written runtime of about 2,000 lines. Rust, Python and JavaScript agree with the reference; Scala does not. No language yet meets the gate below, because the one emitter it depends on has not been started.
4. The proposal: the checker lowers every program to one small typed form; one printer prints that form through a table of text templates for the language; every construct the language has no template for is printed from a definition in terms of constructs it has, down to a floor that any language able to write a byte, branch, repeat, call or jump, and keep small integers in an array meets.
5. The runtime (display, canonical JSON (JavaScript Object Notation), SHA-256 hashing (the Secure Hash Standard, FIPS 180-4), the log chain, maps, float text, the scheduler) is written once in RCE and printed through the same table.
6. A language counts when its printed programs print the reference trace of every example, byte for byte, on that language's own compilers or interpreters, pinned by hash and run in continuous integration. Historical languages run on maintained implementations or on period implementations under emulation. A public counter reports the number of such languages against the catalogue of the online History of Programming Languages.

## 0. Where the goal stands at `d91a9fc8`

| Item | State | Evidence |
|---|---|---|
| Languages emitted | Rust, Python, JavaScript, Scala: `rce emit rust\|python\|js\|scala FILE` writes one file, a runtime prelude followed by the program's functions | 10.5 (`RCE-SPEC.md:3066-3076`) |
| What the emitters read | the abstract syntax tree (AST) after `anf::lower`, not the Form IR | `spec/checker/src/emit/mod.rs:81` |
| Generator code | 5,790 lines in `emit/{base,expr,stmt,prog,tasks,conc06,probe,mod}.rs`, with 498 per-target branches (390 match arms and 108 conditionals); about 40 syntax helpers in `base.rs:272-519` already act as an informal print table | emitter inventory, `count.py`; the registry's figure, 445 branches in 5,717 lines (`emit-targets.md:15-16`), is out of date |
| Runtime code | `rt.rs` 2,109, `rt.py` 2,126, `rt.js` 2,015, `rt.scala` 1,919 lines (8,169). Each rebuilds the value model, checked integers, float text, canonical JSON, SHA-256 and the log chain, fuel and budgets, pattern matching and a tick scheduler of about 570 lines | `wc -l spec/checker/src/emit/rt.*` |
| Agreement | Rust, Python and JavaScript: 1,459 pass, 0 fail, 520 not applicable each. Scala: 52 pass, 1,189 fail and 591 marked unsupported, of 1,832 results | `conformance/README.md`, status block |
| The Scala figures | stale: they predate the conversion of 371 programs and were not re-run. Every harvested `trace` case passes (51 of 51); 1,166 of 1,167 generated and targeted cases fail, 1,186 failures read "run error: exit 2", and the cause was cut from the record (the adapter keeps 500 characters of standard error, the harness 300) | `conformance/README.md`, run of 18 October; `emit_adapter.py:75`; `run.py:99` |
| The Form IR as a print source | `rce emit form` carries the program bodies as a serialised AST (`terms`) that no code reads. It has no static types: `+` at u64, `+` on text and `+=` on f64 are one node, `Bin("+")`, and the operator types the checker computes (`check.rs:76`) are never read | Form IR inventory, section 1.2 |
| Tables | none; `emit/targets/` does not exist | the registry's prerequisite (`emit-targets.md:17-19`, `:351-353`) has not started |
| Registry text | names about 265 languages. Defines an "observation-only" class (`emit-targets.md:340-343`, `:187-188`; `RCE-SPEC.md:3087-3088`), a Tier 4 "registered for completeness" (`:169-176`) and hardware rows with "bounded state only" (`:152`) | each contradicts the direction |

Two findings of the inventories bear directly on the gate.

- **Text display is not pinned.** The reference shows a text value with Rust's `{:?}` (`interp.rs:156`), and so does the emitted Rust runtime (`rt.rs:213`). The bytes of every trace line that shows a text therefore depend on the Unicode tables of the Rust toolchain, which the Unicode pin of 6.3 N7 does not cover. Python, JavaScript and Scala approximate the escaped set with three ranges (`rt.py:178-191`, `rt.js:124-134`, `rt.scala:191-200`). On six probe strings the emitted Python and JavaScript differ from the reference on five (U+200B, U+0483, U+FEFF, U+2028, U+E000). No corpus program contains such a string, so the existing gate stays green.
- **Each runtime restates 6.3 in its own words**, and the known divergences have been of that kind: the C.8 tree against the left fold, `-0` printed as `0`, and a different out-of-range value in an overflow message (`conformance/README.md`, run of 17 October).

Under the gate this proposal defines, the number of languages that pass at this commit is 0. Rust, Python and JavaScript pass today's emitter tests and enter the count when they move onto the one emitter.

## 1. Terms

- **Form**: a program P = (S, E, δ, D, λ) of 6.1: its state, events, step, log and outputs.
- **Reference interpreter**: `spec/checker/src/interp.rs`. It fixes the meaning of every program.
- **Trace**: the bytes the reference prints for a program: its observations, the fuel each uses, the head of each example's log chain and the run of `fn main`. T9 asks every target to print them.
- **ANF** (A-normal form): every operand of an operation is a variable or a literal, so the order of evaluation is the order of the statements.
- **Core** (`rce-core/1`): the typed form defined below, the only input of the printer.
- **Table**: the file `lang/<L>/print.toml` that describes one language L.
- **Row**: an entry of a table that maps one construct of the core to text.
- **Hole**: a named place in a row, such as `{0}`, `{x}` or `{body}`, which the printer fills.
- **Definition**: core code, written once, that expresses one construct by other constructs.
- **Floor**: the constructs without a definition. A table supplies one family of rows for each line of the floor.
- **Carrier**: the way a type without a row is held, as values of types that have rows.
- **Word**: the bound below which a language's own integers are exact.
- **Oracle**: a pinned compiler or interpreter of L that runs printed text in the gate.
- **Gate**: the checks a table passes before its language is counted.
- **Toolchain image**: an OCI (Open Container Initiative) image that holds an oracle, named by the SHA-256 digest of its content.
- **Catalogue**: the public list of languages against which the counter is reported: the online History of Programming Languages, HOPL (hopl.info).
- **CI** (continuous integration): the workflows of `.github/workflows/`.

## 2. Requirements, and what they imply

- **R1 (T9).** Every target prints the trace of every example program byte for byte, on every edge of 6.3.
- **R2.** Every language is a full target. What a language lacks is supplied in that language. No language is held in a lesser class.
- **R3.** Adding a language is adding data. No code is written for a language.
- **R4.** No flags, macros, directives or version selection anywhere in the design.

They imply seven properties of the printed form.

- **D1. One meaning per construct.** If one construct had several meanings (`+` on u8, on f64 and on text), a template would need the target to choose among them at run time. That is what makes each of today's runtimes 2,000 lines. So the form is typed and monomorphic: `int.add` at u8, `f64.add` and `bytes.cat` are three constructs.
- **D2. No evaluation order in a template.** C leaves the order of arguments unspecified; other languages fix it differently. So the form is in ANF, and the order of 6.2 (`RCE-SPEC.md:1555-1558`) is the order of statements. T34 already does this for write-back calls; the form extends it to every expression.
- **D3. No trap rule in a template.** Rows that each restate 6.3 are where today's divergences come from. So every primitive is total on a stated domain, a partial operation p is preceded by its domain test `bad.p` and an explicit halt, and 6.3 is stated once, in the lowering and the library.
- **D4. Every construct beyond a floor has a definition.** Some languages lack exceptions, labelled exits, recursion, mutation, records or wide integers. A table maps what its language has; the printer prints the definitions of the rest. This is how R2 holds without breaking R3.
- **D5. Display, hash and JSON bytes are library code**, written once, so that they are the same on every target.
- **D6. A template has holes and nothing else**: no condition, no loop, no option. The presence or absence of a row is the only choice a table makes (R4).
- **D7. One table describes one language definition.** C.15 states that "a release label names exactly one spec commit … no label selects among alternatives inside a text" (`RCE-SPEC.md:4381-4382`); a table holds no alternatives either.

## 3. The core

### 3.1 One emitter

**ONE-1 One emitter.** `rce emit L FILE` lowers FILE once, to the core, and prints the core through the table of L. FILE is an RCE module or a core module (`"format": "rce-core/1"`). No other code path prints a language. `rce emit core FILE` prints the core as JSON, and `rce emit form FILE` is unchanged. `rce run-core FILE` prints what the core evaluator writes for the core of FILE: one implementation under test, compared with the reference like any other, never the expected bytes. L names a table; the checker carries every table, as it carries the library (XF-3), so adding a file is the whole of adding a language.

### 3.2 The constructs

**ONE-2 The core.** The core has exactly the constructs listed in this section. A new construct enters the core by the rule that changes the language (CONTRIBUTING rule 3), and a change to the core runs the gate of every table.

**Types (9).**

| Type | Values | Kind |
|---|---|---|
| `bool` | true, false | plain |
| `int[lo, hi]` | the integers from lo to hi, −2^63 ≤ lo ≤ hi ≤ 2^64 − 1 | plain |
| `f64` | finite binary64 values; −0.0 is a value distinct from 0.0 (6.3 F4, F6) | plain |
| `f32` | binary32 values, held as binary64 (6.3, the f32 rule) | plain |
| `bytes` | immutable byte strings; text is UTF-8 bytes (6.3 N7) | plain |
| `arr τ` | arrays of fixed length | owned |
| `list τ` | growable sequences | owned |
| `tup(τ1, …, τn)` | products | owned |
| `sum(τ1, …, τn)` | tagged unions; the tag is `int[0, n−1]` | owned |

The RCE types map onto these as the Form IR inventory sets out: integers to `int[lo, hi]`, text to `bytes`, records, events, errors, regions and the state to `tup` in declaration order (the positional form of 6.1, `RCE-SPEC.md:1383-1395`), `Set` and `Map` to lists in canonical order without duplicates, options, kinds and `τ ! ε` to sums, closures to sums of their captures (defunctionalised), and channel, actor and task identities to `bytes` (the content hash of 6.6). One more type is fixed: `Halt = sum(bytes, tup(bytes, bytes))`, either a trap text from the closed list of 6.2 (`RCE-SPEC.md:1538-1550`) or a declared error as its name and display. It does not depend on the module, so library code can halt.

**Atoms (3).** A local `vN`, a global `gN`, a literal `k:τ` of a plain type.

**Statements (13).**

| Form | Meaning |
|---|---|
| `vN = a` | bind atom a; an owned a is moved |
| `vN… = p[ι](a…)` | apply primitive p at type argument ι; p is total on its domain and binds zero or more results |
| `vN = call F(a…)` | call F; if F halts, this function halts with the same Halt |
| `vN = catch F(a…)` | call F; vN : `sum(ret_F, Halt)`, 0 when F returned and 1 when it halted. The only way to observe a halt |
| `var vN: τ = a` | a mutable local |
| `set vN = a`, `set gN = a` | assignment |
| `if a {S} else {S}` | a : bool |
| `block {S}` | run S; a `break` may leave it |
| `loop {S}` | run S repeatedly; left only by `break`, `ret` or `halt` |
| `break d` | leave the d-th enclosing `block` or `loop`, 0 being the innermost |
| `ret a` | return a |
| `halt a` | a : Halt; end this function in a halt |
| `out a` | a : bytes; append to standard output, the only effect |

**Declarations (3).** `fn (params) -> τ {S}`; `group [fn…]`, a set of mutually recursive functions that call each other as `rec i`; `global gN: τ = literal`, where a global of an owned type has no literal and is set by the module's initialisation before it is read.

**Primitives (100).** Each name has one meaning at its type argument. A domain test `bad.p` is true exactly where p is undefined.

| Family (count) | Total operations | Domain tests and domains |
|---|---|---|
| `bool` (4) | `not and or eq` (strict; short-circuit evaluation is written with `if`) | none |
| `int` (33) | `add sub mul neg abs quo rem` (quotient toward zero, remainder with the dividend's sign); `eq lt le min max`; `wadd wsub wmul` (wrapping) and `sadd ssub smul` (saturating) of N2; `and or xor not shl shr` on the two's-complement form; `cast` | `bad.add bad.sub bad.mul bad.neg bad.abs`; `bad.quo` (the case MIN / −1, for `quo` and `rem`; the zero divisor is tested with `eq`); `bad.shift`; `bad.cast` |
| `f64` (25) | `add sub mul div sqrt` (correctly rounded, F1); `neg abs floor ceil round` (ties away from zero, F8); `min max eq lt le` (IEEE order); `of_int` (round to nearest even, N4); `trunc` (toward zero); `to_f32` | `bad.add bad.sub bad.mul bad.div bad.sqrt` (the IEEE result is not finite, F4); `bad.trunc`; `bad.to_f32` |
| `f32` (9) | `add sub mul div` (the f32 rule of 6.3); `to_f64` | `bad.add bad.sub bad.mul bad.div` |
| `bytes` (8) | `len get slice cat eq lt` (bytewise order, which on UTF-8 is code-point order); `of_list to_list` | `get` needs i < len; `slice` needs i ≤ j ≤ len |
| `arr` (4) | `new len get set` | i < len |
| `list` (6) | `new len get set push pop` | i < len; `pop` needs len > 0 |
| `tup` (3) | `mk get open` | none |
| `sum` (3) | `mk tag get` | `get[k]` needs tag = k |
| `io` (1) | `byte`: write one byte, `int[0,255]`, to standard output | none |
| `mem` (2) | `get set` on one global array of integers, for languages without arrays of their own; indices from 0 to the array's size | none |
| generic (2) | `copy[τ]`: a value equal to its operand that shares nothing with it; `drop[τ]`: consume the operand | none |

Everything else is library code, written once and described below: text methods, maps and sets, canonical order and equality, display, JSON, SHA-256, decimal and float text, trap messages, the C.8 tree, fuel and cost, and the scheduler.

### 3.3 Well-formedness

**ONE-3 Well-formedness**, checked whenever a core module is read.

- **W1 Types.** Every atom and application is well typed, and every `ret` matches the function's result type.
- **W2 Paths.** Every path of a function ends in `ret` or `halt`, and every `break d` names an enclosing `block` or `loop`.
- **W3 Ownership.** Every owned local is consumed exactly once on every path. The consuming operations are `call`, `catch`, `ret`, `halt`, `set`, a binding, the `mk` operations, `tup.open`, `sum.get`, `drop`, and `set`, `push` and `pop` on `arr` and `list`. `len`, `get`, `tag`, `eq`, `lt` and `copy` read without consuming, and a `get` of an owned element returns a copy. An owned global is moved out only by a binding, and every path from that binding sets the global again before the function returns or calls. So every consuming operation may print as an update in place, every copy is visible as `copy`, and every value's end is visible as `drop`; a table that holds values in `mem` releases them at `drop`, through the carrier definitions.
- **W4 Domains.** Every application of a partial primitive is dominated by the domain test the lowering placed before it, or by one fact from a fixed list (for example, a loop counter below its bound).
- **W5 Keys.** A group's key is the SHA-256 of its canonical JSON (`form-ir-2.md` section 1, rule (a)). Types are written inline, so a key does not depend on the module that uses it, and `rec i` occurs only inside its own group, so the graph of keys has no cycle.
- **W6 Arguments.** The operands of `call` and `catch` are locals or literals, never globals, and `set` names a `var` local or a global, never a parameter. A language that passes arguments by reference (FORTRAN, for one) then prints the core without a copy.

### 3.4 The JSON form

```
module = {"format":"rce-core/1","globals":[[τ,LIT|null]…],"defs":{KEY:{"fns":[fn…]}},
          "entry":[KEY,i],"names": the concept record of 6.6, never hashed}
fn     = {"params":[τ…],"ret":τ,"body":[stmt…]}
τ      = ["bool"]|["int","lo","hi"]|["f64"]|["f32"]|["bytes"]|["arr",τ]|["list",τ]|["tup",τ…]|["sum",τ…]
atom   = ["v",i]|["g",i]|["k",τ,LIT]    LIT: true|false; an integer as decimal text;
                                        f64 and f32 as the 16 hex digits of the binary64 bits; bytes as hex
stmt   = ["let",atom] | ["prim",NAME,[ι…],atom…] | ["call",F,atom…] | ["catch",F,atom…]
       | ["var",τ,atom] | ["set",["v",i]|["g",i],atom] | ["if",atom,[stmt…],[stmt…]]
       | ["block",[stmt…]] | ["loop",[stmt…]] | ["break",d] | ["ret",atom] | ["halt",atom] | ["out",atom]
F      = [KEY,i] | ["rec",i]
```

Parameters are locals 0 to n − 1. In textual order, each `let`, `prim`, `call`, `catch` and `var` binds the next index or indices. The concept record gives each function and each local the name of the RCE item it came from, which the printer writes as a comment.

### 3.5 Lowering

**ONE-4 Lowering.** The checker records the static type, the binding and the resolved method of every node; today it records none of them (`check.rs:76`). From that typed tree the lowering writes the core:

| Surface | Core |
|---|---|
| any expression | ANF bindings in the order of 6.2 (T34, extended to every expression) |
| `a ⊕ b` at an integer type T | `int.bad.⊕[T]`, then `if … { halt msg }`, then `int.⊕[T]`; for `/` and `%`, first `int.eq(b, 0)` halting "division by zero", then `bad.quo`. The trap falls at the operation, as the reference's does |
| `+` on text or on a list | `bytes.cat`, `lib.list.cat` |
| float operators, `as` | one guarded primitive per source operator (F2) |
| `a[i]`, `a[i:j]` | a guard on `len`, then `get` or `slice`; text slices also call `lib.utf8.boundary` (N7, N8) |
| `and`, `or`, `if` expressions | `if` statements |
| constants | their comptime values, as literals or globals |
| methods (5.15) | primitives or library calls chosen from the static receiver type |
| constructors | `tup.mk`, `sum.mk` |
| `match` | a chain of `if`s on `sum.tag`, literal tests and range tests in arm order, then `halt` "no match arm" |
| `fail e` | in a `τ ! ε` function, `ret sum.mk[1](e)`; in a step or `main`, `halt (name, display of e)` |
| `try e`, `try e else X`, `e catch \|x\| h` | a tag test, then an early `ret` or the handler. These are values, not halts, so a trap is never caught by them (6.2) |
| `var`, `=`, `op=`, `a.f[i] = v` | `var` and `set`; a path update is `tup.open`, `list.set` and `tup.mk`, all consuming |
| `&mut` | state passing (`form-ir-2.md` section 2); the caller passes `copy(a)` when the callee can halt and `a` is still live (the write-back rule of 6.2 and F.4.4) |
| `while c bound n`, `loop bound n`, `for` | `loop` with a counter, `if k = 0 { halt "loop bound" }; k −= 1`, the shape proved in `spec/lean/RCE/Emit.lean:142-145`; `for` walks ranges, lists, maps and sets in canonical order, and text by index |
| generics, traits | monomorphised; trait calls resolved statically |
| lambdas, iterator functions (5.14) | defunctionalised: a sum of captures and an `apply` function |
| recursion, `fn … bound n` | a `group` with `rec i`; `bound n` becomes a counter per function |
| the steps of a region | one region function: the steps in source order, each a tag test, the payload binding, the guard and the body |
| `emit v` | `list.push` onto the step's output list |
| `par` | the branches in source order (C.4, T16), after the charge of C.6.3 |
| `for x in e par` | each item from the identity of the reduction operator (`conc.rs:494-506`), the contributions collected in a list and reduced by `lib.tree.<op>` (C.8) |
| `with budget n … else` | push a budget frame; `catch` of the body as a function of copies of the locals it writes; pop; take the results, or bind `Exhausted{limit, used}` and run the else branch, or halt again |
| tasks, channels, `select`, actors, deadlines | each task is a state machine, `resume(frame, input) -> (frame′, action)`, and the scheduler of 6.8 is library code; no language needs threads |
| fuel (6.7, 10.5) | `call lib.tick(n)` before each group of statements, n the ticks the reference makes in the group; a group ends before every statement that can halt or be observed (a domain test, a guard, a call, an `out`, the head of a loop or a branch), so the trap observation and the totals equal the reference's |
| cost (C.6) | `call lib.charge(n)` over a global list of budget frames, all or nothing (C.6.2); left out of code that no `with budget` can enclose |

The constructs today's emitters do not translate are held as follows; none is marked unemittable in the core.

| Construct | Today | Held in the core as |
|---|---|---|
| `ct fn` (F.5.4) | W-CTFN-2: "ct fn is not emitted for this target" (`v09.rs:1668`) | a function like any other, printed with its meaning; `ct_choose` as an arithmetic selection. Constant time is a property of timing, which T9 does not state, so the registry reports for each table whether a timing measurement exists. A timing gate is left open |
| `Shell.cmd`, `ShellArg` text, `Cap.narrow`, `ct_choose`, contract clauses (Appendix F) | W-RES-1, "run by the reference evaluator only" (`v09.rs:1666-1704`) | core code with the meaning the reference gives each (Appendix F), checked differentially against the reference like every other construct of the lowering; the host's run of a command is a foreign call answered from D (XF-5, 5.11) |
| a write-back call in a condition | `unsupported` (`expr.rs:161-168`, `anf.rs:16-20`) | ANF of every expression |
| a generic call whose type argument is unresolved (`r5-generic-call`) | not emitted today | monomorphised from the checker's types |
| the restore of `&mut` on failure (F.4.4) | the module is not emitted (`v09.rs`, `needs_restore`) | state passing with `copy` |
| a closure, iterator or LVar sent to another task in JavaScript | a trap (`rt.js:1292`) | tasks are state machines in one process; nothing is serialised |

### 3.6 The fixed output

**ONE-5 One output, no arguments.** A printed program reads no argument, no environment variable and no input. It prints the trace: the lines that `--trace` and `--fuel-trace` print together today, for every observation in source order, then the run of `fn main` when the module has one. For `spec/examples/01_counter.rce` that is `example 2`, four state lines, `trace 889e6559…`, `fuel 11`, then `example 3`, two lines, `trace 07ad897c…`, `fuel 6` (the emitted Python at `d91a9fc8` run with both arguments). The arguments of today's emitted programs (`--trace`, `--fuel-trace`, `--run`, `--fuel=N`, `--fuel-batch=`, and the arguments of the `--run` path such as `--event=`, `--link=` and `--restart=`, `rt.py:24-56`) are retired:

- the parts they select are all printed, every time;
- the fuel limit is a global literal of the core module, so a test that needs another limit changes that literal in the core JSON, which is data, and the printer is unchanged;
- the inputs a conformance test passes today as arguments are written into the test's program text, converted once and checked against the reference before any table uses them.

The reference's trace (ONE-5, `one::trace::reference`) is the expected output of every gate. No gate takes its expected bytes from an implementation under test, the core evaluator included.

## 4. The table

### 4.1 One file per language

**ONE-6 A language is one file.** A table is `lang/<L>/print.toml`, in TOML (Tom's Obvious, Minimal Language). Its key is the SHA-256 of the canonical JSON (6.1.1) of the file's content without its `oracle` records. The first line of every printed program names the program hash (10.8), the table's key and the checker commit, so that printed text is tied to its source, its table and its printer. `lang/<L>/grammar` and `lang/<L>/semantics.toml` of `emit-targets.md` section 0 stay the lift's artefacts; they are not needed to emit.

### 4.2 Rows, holes and patterns

**ONE-7 Rows are templates.** A row is one of:

- a string: one line, or an expression;
- an array of lines, each `[depth, text]` or `[depth, text, label]`, where depth is added to the nesting level and label is `"lh"`, `"lx"`, `"n"` or a decimal label;
- an inline table `{ text = …, helper = [names] }`, whose helpers are printed once in the file when the row is used.

A row holds text and holes and nothing else. `{{` and `}}` print braces. A line that consists only of a list hole (`{then}`, `{else}`, `{body}`, `{decls}`, `{common}`, `{ginit}`, `{helpers}`, `{defs}`) is replaced by the lines of that list, each indented by the depth written on the hole's line. A row whose text is empty prints nothing and is still present. A statement list that prints no line prints the layout's `empty`. A primitive row prints as a statement when its text has a result hole, when it is an array of lines, or when its primitive binds no result; otherwise it is an expression, placed in the `bind` row.

The holes are:

| Hole | Filled with |
|---|---|
| `{0}` … `{9}` | the operands, printed as atoms |
| `{args}`, `{params}` | the operands, the parameters, joined by `sep` |
| `{x}`, `{x0}`, `{x1}`, `{xs}` | the result, each result, all results joined by `sep` |
| `{e}` | in `bind`: the atom or the expression row's text |
| `{f}` | the callee in `call`, `rec` and `catch`; the function being printed in `fn` and `ret` |
| `{t}` | the `type` row of the declared name's type in `decl`, of the result type in `fn` |
| `{lo}`, `{hi}`, `{bits}` | the bounds of the integer type argument as literals, and ⌈log2(hi − lo + 1)⌉ |
| `{k}` | the component index of `tup.get` and the variant index of `sum.mk` and `sum.get` |
| `{then}`, `{else}`, `{body}`, `{decls}` | statement lists and declarations |
| `{lh}`, `{lx}` | the head and exit labels of a `loop` or `block`; in `break`, the exit label of the target |
| `{n}` | the number of a name or a label |
| `{text}` | a comment's text |
| `{dec}`, `{hex}`, `{json}` | a literal's decimal magnitude, a byte as two hex digits, a float in its canonical JSON form (6.1.1) |
| `{helpers}`, `{globals}`, `{defs}`, `{entry}`, `{common}`, `{ginit}`, `{ng}`, `{memory}` | in `host` rows: the helpers, the initial values of the globals joined by `sep`, the functions, the entry's name, the `common` lines, the `ginit` lines, the number of globals, the `memory` field |
| `{src}`, `{exe}` | in an oracle record: the printed file and the built program |

A hole that a row's construct does not define is an error when the table is read.

A row key is a construct name, optionally followed by a pattern in brackets: `"int.add[int@word]"`, `"tup.mk[tup()]"`, `"list.get[owned]"`, `"break[0]"`. A pattern is a base followed by qualifiers. The base is a type name (`bool`, `int`, `f64`, `f32`, `bytes`, `arr`, `list`, `tup`, `sum`), `plain`, `owned`, `*` or one closed core type in text form (`int[0,255]`, `tup()`). The qualifiers are `signed` (lo < 0), `unsigned` (lo = 0) and `@word`, written on the base, which holds when every operand and the exact result of the operation lie strictly between −word and word (for a `type` row: every value does). For `break` the bracket holds the depth. The pattern is matched against the primitive's type argument ι: the integer type for `int`, the element type for `arr` and `list`, the component type for `tup.get`, the tuple type for `tup.mk` and `tup.open`, the payload type for `sum`, the integer type for `f64.of_int` and `f64.trunc`, the operand's type for `copy` and `drop`, and the target type for `cast` (where `@word` also covers the source). The most specific matching row is used: a closed type, then a base with more qualifiers, then a base with fewer, then `plain`, `owned` or `*`, then a key without brackets. Two equally specific matches are an error when the table is read.

### 4.3 Every field

**ONE-8 The fields.** A table has the sections and fields below and no others. Every row section may omit any row; absence is meaningful, as the rule on completeness below states.

| Section | Field | Value | Meaning |
|---|---|---|---|
| `lang` | `name` | text | the language definition the table prints, by the name its defining document gives it |
| | `defined_by` | text | the defining document: a standard, a report, or an implementation's reference at a release or commit |
| | `catalogue` | text | the entry of the catalogue snapshot this table counts for, as the counter below states |
| | `ext` | text | the file name extension of a printed program |
| | `word` | text | `"unbounded"`, or the decimal integer w such that the language computes every integer v with \|v\| < w exactly |
| | `depth` | integer | how deeply calls between members of a group may nest through the language's own call; 0 when there is no `rec` row |
| | `memory` | integer | words of the single array `mem`; 0 when the table has `arr` rows |
| `layout` | `indent` | text | added before a line once per nesting level |
| | `width` | integer | the last column a line may use; 0 when lines are unbounded |
| | `lead` | text | printed before every unlabelled line |
| | `cont` | text | begins a continuation line (used only when `width` > 0) |
| | `cont_end` | text | ends a line that is continued |
| | `label` | template `{n}` | replaces `lead` on a labelled line, padded with spaces to the length of `lead` |
| | `empty` | text | the statement printed for an empty statement list |
| | `comment` | template `{text}` | a comment line, printed without `lead`; one precedes each function and names its origin from the concept record |
| `lex` | `local`, `global`, `fn` | template `{n}` | names of locals, globals and functions; `global` may carry type patterns |
| | `reserved` | list of text | words no name template may produce; checked when the table is read |
| | `name_max` | integer | the longest name; 0 when unbounded |
| | `sep` | text | the separator of `{args}`, `{params}`, `{xs}` and `{globals}` |
| | `true`, `false` | text | the bool literals |
| | `unset` | text | the initial text of a global of an owned type |
| | `int`, `neg` | template `{dec}` | an integer literal of magnitude below the word, non-negative and negative |
| | `f64` | template `{json}` | a binary64 literal; absent when f64 has no type row |
| | `bytes_open`, `bytes_close`, `bytes_plain`, `bytes_esc` | text; list of hex ranges; template `{hex}` | a byte-string literal: its delimiters, the bytes printed as themselves, the form of every other byte; absent when bytes has no type row |
| `type` | a pattern | text | the language's spelling of a type (`{t}`). Its presence makes the type native, carried by the language itself |
| `stmt` | `bind`, `var`, `set`, `call`, `rec`, `catch`, `if`, `block`, `loop`, `break`, `break[0]`, `ret`, `halt`, `out`, `fn`, `decl`, `label`, `goto`, `ifgoto` | row | the statement forms, declarations and the jump family (`label` with `{n}`, `goto` with `{n}`, `ifgoto` with `{0}` and `{n}`) |
| `prim` | a primitive with a pattern | row | the 100 primitives |
| `lib` | a library function with its key, `name@KEY` | row | a native replacement of a library function, under the rule for library rows below |
| `host` | `file` | lines | the skeleton of a printed program |
| | `common` | lines | declarations every function repeats |
| | `ginit` | row | one line per global's initial value (`{x}`, `{0}`) |
| `helper` | a name | lines | text printed once when a row names it |
| `oracle` | `name`, `kind`, `image`, `build`, `run`, `device`, `suite`, `limit` | records | the toolchains of the gate, described with the gate below |

No field is a boolean, and no field selects among behaviours. `word`, `depth` and `memory` are numbers that describe the language; the printer reads them by the fixed rules stated with completeness below and with the carriers and control definitions of the next section.

### 4.4 Definitions, completeness and the floor

**ONE-9 Definitions and completeness.** Every construct outside the floor has one or more definitions in the core, in a fixed order; the definitions are listed in the next section. The printer prints a construct with the table's row when one applies, and otherwise with the first definition all of whose constructs it can print. A table is complete when every construct the program reaches can be printed; that is a least fixpoint and is decidable. When a table is not complete, the printer names the floor rows it lacks, and nothing else: that list is all a language, including one designed after this text, has to supply. A missing row never changes a meaning; it changes only which text expresses it.

**ONE-10 The floor.** A complete table has rows for one family in each line:

1. **Output**: the `out` statement, or the primitive `io.byte`.
2. **Control**: `if`, `loop`, and `break` or at least `break[0]`; or `label`, `goto` and `ifgoto`.
3. **Locals**: `var` and `set`, or `rec` with `depth` > 0, so that loops become recursion that passes state.
4. **Functions**: `fn`, `call` and `ret`, or the jump family with `mem`, so that calls become jumps with a stack of return points.
5. **Integers**: `int.add`, `int.sub`, `int.mul`, `int.quo`, `int.rem`, `int.lt` and `int.eq` at `int@word`, with a word of at least 256.
6. **Aggregates**: `arr.new`, `arr.len`, `arr.get` and `arr.set`, or `mem.get` and `mem.set` with `memory` > 0.

A language that can write a byte, branch, repeat, call or jump, and keep integers below 256 in an array meets the floor, whatever its age. Halts, recursion, booleans, every integer type, binary64, binary32, byte strings, lists, records and sums all have definitions, listed in the next section. A language that describes data and does not run, such as JSON Schema or a Protocol Buffers schema, has no output or control of its own; it is held by the parts of the Form it can state, the types of S and E and their values, and its gate is the round trip of the corpus's values through it. That is the whole of what such a language says, and its table prints all of it.

### 4.5 What a definition leaves open, and versions

**ONE-11 Choices the language leaves to its implementations.** Where a language's definition leaves a property to the implementation (the width of FORTRAN 77's INTEGER, the call depth of CPython, the character code of a byte above 127), the table states the value its text relies on, in `word`, `depth` or `memory` or in a row, and every toolchain the table lists must provide it; the row vectors of the gate check it. A toolchain that provides less is listed only after the value is lowered for every toolchain of the table. No value is chosen per toolchain.

**ONE-12 One language definition per table.** A table prints one text for every toolchain it lists, and that text must pass on all of them. A newer release of an implementation is added to the oracle list of the same table, never given a table of its own, as long as one text passes on all. Two definitions for which no single text passes on both (FORTRAN 66 and FORTRAN 77, which differ in the meaning of a zero-trip `DO` loop [K]; ECMAScript 3 and ECMAScript 2020) are two languages, with two tables and two catalogue entries. There is no third form: no table holds alternatives, the printer takes no option, and the Form never carries a target annotation. The prior systems surveyed hold dialects by a flag, a directive or an annotation (gfortran `-std`, GnuCOBOL `-std` and its configuration files, Free Pascal `{$mode}`, Nim `--jsbigint64`, SymPy's `standard=`, Haxe `#if`, protobuf's `jstype` and Editions); in this design each of those is either one table's text or two tables.

## 5. The runtime, written once

### 5.1 Where printed code comes from

**ONE-13 Four origins, none per language beyond rows.**

| Origin | Written | Printed by | Contents |
|---|---|---|---|
| rows | per language, in its table | that table | the primitives and statements the language has |
| definitions | once, in core text, `lang/core/defs/` | every table that lacks the row | carriers, and control, calls and halts (below) |
| the runtime library | once, in RCE, `lang/rt/*.rce`, lowered like any program | every table | display, JSON, hashing, the log chain, collections, float text and the scheduler (below) |
| per-module code | by the lowering, from the module's types | every table | the δ driver, the observation driver, `show[τ]`, `cjson[τ]` and `cmp[τ]` for each type the module uses |

The definitions are written in core text rather than in RCE because they are what RCE's own checked integers, floats and aggregates are made of; the runtime library sits above them and is written in RCE. So a table supplies at least the rows of the floor (ONE-10) and at most rows for the 100 primitives and the 19 statement rows; everything else in a printed program is emitted RCE code (the library) or emitted core code (definitions and per-module code), the same for every language.

### 5.2 Carriers

**ONE-14 Carriers.** A type without a `type` row is held as values of types that have rows, by these definitions, applied in order until every type that remains has a row:

| Type without a row | Carrier | Definitions |
|---|---|---|
| `int[lo, hi]` outside the word | a sign and ⌈bits / b⌉ digits in base B = 2^b, b = ⌊log2(word) / 2⌋ (FORTRAN 77: B = 2^15, so a u64 is five digits; a word of 10^18, as COBOL-85's 18 decimal digits give, yields B = 2^29) | `lang/core/defs/int.core`: schoolbook add, subtract and multiply, long division (Knuth's Algorithm D [K]) for `quo` and `rem`, comparisons; every result is exact, so every trap and every trap text follows from it |
| bit operations, wrapping and saturating operations at an int type that has arithmetic rows but no row for them | the same type | `quo` and `rem` by 2 digit by digit; the exact result reduced by `rem` or clamped by `min` and `max` |
| `bool` | `int[0,1]` | `bool.core` |
| `f64` | its 64 bits as `int[0, 2^64 − 1]`, itself carried by digits where it does not fit the word | `f64.core`: IEEE 754-2019 binary64 add, subtract, multiply, divide and square root, correctly rounded to nearest even, with subnormals (F1, F3); exact `floor`, `ceil`, `round`, `of_int` and `trunc`; the finiteness test of F4 on the exponent field. Berkeley SoftFloat is the published model of such code [K] |
| `f32` | the binary32 bits | `f64.to_f32` rounding in `f64.core`; the operations by the f32 rule of 6.3 |
| `bytes` | `list int[0,255]` | `bytes.core` |
| `list τ` | an `arr τ` with a length and a capacity that doubles | `list.core` |
| `arr`, `tup`, `sum` | records in `mem`: a header word and the fields; owned fields are addresses | `mem.core`: a free-list allocator over `mem`, `copy` as a deep copy, `drop` as a release |

Values of plain types that a carrier puts in `mem` become owned under the carrier: the definition inserts `copy` at each use but the last and `drop` after the last, by liveness, so the allocator's live memory is the program's live data. A prior system that supplies wide integers in the source language, as Haxe builds `Int64` from two `Int32` words (`std/haxe/Int64.hx`, `Int32.hx`, Haxe at 931ad718), is the same method applied to one type.

### 5.3 Control, calls and halts

**ONE-15 Control, call and halt definitions.**

| Construct without a row | Definition |
|---|---|
| `block {S}` | `loop { S; break 0 }` |
| `break d`, d > 0, with only `break[0]` | an exit counter: `set e = d; break 0`, and after each construct left this way, `if e > 0 { set e = e − 1; break 0 }` |
| `if`, `loop`, `block`, `break`, with only the jump family | labels and jumps, the lowering Ratfor made from structured control to FORTRAN 66 (Kernighan 1975) |
| `label`, `goto`, `ifgoto`, needed by the next row, with only structured rows | a loop over a program counter |
| `rec` beyond `depth`, or with no `rec` row | the group as one function with an explicit stack of frames (in a list, or in `mem`) and a dispatch loop over its call sites; with `depth` > 0, calls nest through the language's own call up to `depth` and continue on the explicit stack beyond it |
| `call`, `ret`, with no `fn` row | jumps with a stack of return points |
| `halt`, `catch`, with no rows | status returns: `halt a` sets the globals "halted" and "Halt" and returns a default value; after each call of a function that can halt, a test returns at once; `catch` tests and clears the status. This is the `?` shape of `spec/lean/RCE/Emit.lean` |
| `var`, `set`, with no rows | state passing: a loop becomes a recursive function whose parameters are the loop's variables |
| `out` with only `io.byte` | a loop over `bytes.get` |

Closures, iterator functions, tasks, channels, `select` and actors have no rows at all: the lowering has already turned them into sums, `apply` functions and state machines (ONE-4) for every language.

### 5.4 The runtime library in RCE

**ONE-16 The runtime library is RCE.** The library is RCE modules in `lang/rt/`, checked by `rce check`, lowered to the core like any program, and printed through every table. Its code is not charged fuel: the lowering places no tick in it, and the program's charge points carry the cost of each library call as C.6.4 and 10.5 state ("a temporary introduced by the emitter's own lowering is not charged").

| Module | Contents | Source |
|---|---|---|
| `rt/fuel` | `tick`, `charge`, budget frames | `interp.rs:436-442`, C.6.2; new, a few lines |
| `rt/msg` | the trap texts of 6.2, with exact values | `interp.rs:638-648`; new |
| `rt/show` | display of every plain type; text escaped by the pinned escape set (below) | `emit/mod.rs:86-210`, `interp::show`; new |
| `rt/cjson`, `rt/cmp` | canonical JSON (6.1.1) and canonical order and equality (6.1) | new; with the per-module `cjson[τ]` and `cmp[τ]` |
| `rt/float` | shortest round-trip digits, in the text form of N3 and the JSON form of 6.1.1, from the bits | new; an exact algorithm in the line of Steele and White (1990) or Ryū (Adams 2018) [K]; the bits by the exact arithmetic of `stdlib/lib/num/special.rce:27`, since RCE has no bit cast |
| `rt/sha256` | SHA-256 over bytes (FIPS 180-4) | port of `stdlib/lib/crypto.rce:21` from `List[i64]` to bytes and u32 |
| `rt/dlog` | the log chain of 6.1 | new |
| `rt/utf8` | boundaries, decoding, code points | `stdlib/lib/text/utf8.rce` |
| `rt/text`, `rt/list`, `rt/map`, `rt/set` | the methods of 5.15 on text, lists, maps and sets, maps and sets as sorted lists | new |
| `rt/tree` | the reduction tree of C.8 | `host::tree_split`, `RCE.Par.split`; a few lines |
| `rt/sched`, `rt/chan`, `rt/actor`, `rt/lvar` | the scheduler of 6.8, channels, the rotation of `select`, actors, deadlines, restarts, LVars | 6.8, C.11; the largest piece |

The library replaces what the emitter inventory classes as algorithm in today's runtimes, which is almost all of their 8,169 lines; the host needs that remain are standard output and, for speed only, threads.

### 5.5 Library rows

**ONE-17 A library row replaces one function, by key.** A table may give a library function a native row in `[lib]`, keyed `name@KEY` with the function's key (W5), for speed. The row applies only while the function's key is that key: when the library changes, the row stops applying and the function prints from its RCE source. A row is admitted only after the function's library vectors in the gate pass on every oracle of the table, and it must charge what the RCE source charges, which XF-6 already requires of native library code. No table needs a library row; the worked tables have none.

### 5.6 Pinned data

**ONE-18 Data the library reads is pinned in the text.** The 25 white-space code points of N7 are already enumerated (`interp::WHITE_SPACE`). The set of code points that display escapes, and the rule for a grapheme extender at the start of a text, are enumerated in the same way, as data, from the tables of the pinned Rust toolchain (`spec/checker/rust-toolchain.toml`, channel 1.98.1), so that no trace changes, and both `interp::show` and `rt/show` read that data. The Unicode version the enumeration came from is recorded beside the N7 pin; moving it is a change of the specification. The trap texts are the closed list of 6.2.

## 6. Two worked tables

### 6.1 The table of Python 3

```toml
# lang/python3/print.toml: the table of Python 3 (ONE-6 to ONE-8)

[lang]
name = "Python 3"
defined_by = "The Python Language Reference, release 3.13, with the library modules copy, math, struct and sys"
catalogue = "Python 3"
ext = "py"
word = "unbounded"
depth = 400
memory = 0

[layout]
indent = "    "
width = 0
lead = ""
cont = ""
cont_end = ""
label = ""
empty = "pass"
comment = "# {text}"

[lex]
local = "v{n}"
global = "G[{n}]"
fn = "f{n}"
reserved = ["False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield", "G", "Halt", "copy", "math", "struct", "sys"]
name_max = 0
sep = ", "
true = "True"
false = "False"
unset = "None"
int = "{dec}"
neg = "(-{dec})"
f64 = "{json}"
bytes_open = "b'"
bytes_close = "'"
bytes_plain = ["20-26", "28-5b", "5d-7e"]
bytes_esc = "\\x{hex}"

[type]
bool = "bool"
int = "int"
f64 = "float"
f32 = "float"
bytes = "bytes"
arr = "list"
list = "list"
tup = "tuple"
sum = "tuple"

[stmt]
bind = "{x} = {e}"
var = "{x} = {0}"
set = "{x} = {0}"
call = "{x} = {f}({args})"
rec = "{x} = {f}({args})"
catch = [[0, "try:"], [1, "{x} = (0, {f}({args}))"], [0, "except Halt as e:"], [1, "{x} = (1, e.args[0])"]]
if = [[0, "if {0}:"], [1, "{then}"], [0, "else:"], [1, "{else}"]]
loop = [[0, "while True:"], [1, "{body}"]]
"break[0]" = "break"
ret = "return {0}"
halt = "raise Halt({0})"
out = "sys.stdout.buffer.write({0})"
fn = [[0, "def {f}({params}):"], [1, "{body}"]]

[prim]
"bool.not" = "(not {0})"
"bool.and" = "({0} and {1})"
"bool.or" = "({0} or {1})"
"bool.eq" = "({0} == {1})"
"int.add[int]" = "({0} + {1})"
"int.sub[int]" = "({0} - {1})"
"int.mul[int]" = "({0} * {1})"
"int.neg[int]" = "(-{0})"
"int.abs[int]" = "abs({0})"
"int.quo[int]" = { text = "_quo({0}, {1})", helper = ["quo"] }
"int.rem[int]" = { text = "_rem({0}, {1})", helper = ["quo", "rem"] }
"int.eq[int]" = "({0} == {1})"
"int.lt[int]" = "({0} < {1})"
"int.le[int]" = "({0} <= {1})"
"int.min[int]" = "min({0}, {1})"
"int.max[int]" = "max({0}, {1})"
"int.wadd[int]" = { text = "_wrap({0} + {1}, {lo}, {hi})", helper = ["wrap"] }
"int.wsub[int]" = { text = "_wrap({0} - {1}, {lo}, {hi})", helper = ["wrap"] }
"int.wmul[int]" = { text = "_wrap({0} * {1}, {lo}, {hi})", helper = ["wrap"] }
"int.sadd[int]" = "min(max({0} + {1}, {lo}), {hi})"
"int.ssub[int]" = "min(max({0} - {1}, {lo}), {hi})"
"int.smul[int]" = "min(max({0} * {1}, {lo}), {hi})"
"int.and[int]" = "({0} & {1})"
"int.or[int]" = "({0} | {1})"
"int.xor[int]" = "({0} ^ {1})"
"int.not[int signed]" = "(~{0})"
"int.not[int unsigned]" = "({hi} - {0})"
"int.shl[int]" = { text = "_wrap({0} << {1}, {lo}, {hi})", helper = ["wrap"] }
"int.shr[int]" = "({0} >> {1})"
"int.cast[int]" = "{0}"
"int.bad.add[int]" = "not ({lo} <= {0} + {1} <= {hi})"
"int.bad.sub[int]" = "not ({lo} <= {0} - {1} <= {hi})"
"int.bad.mul[int]" = "not ({lo} <= {0} * {1} <= {hi})"
"int.bad.neg[int]" = "not ({lo} <= -{0} <= {hi})"
"int.bad.abs[int]" = "not ({lo} <= abs({0}) <= {hi})"
"int.bad.quo[int]" = "({0} == {lo} and {1} == -1)"
"int.bad.shift[int]" = "not (0 <= {1} < {bits})"
"int.bad.cast[int]" = "not ({lo} <= {0} <= {hi})"
"f64.add" = "({0} + {1})"
"f64.sub" = "({0} - {1})"
"f64.mul" = "({0} * {1})"
"f64.div" = "({0} / {1})"
"f64.sqrt" = "math.sqrt({0})"
"f64.neg" = "(-{0})"
"f64.abs" = "abs({0})"
"f64.floor" = { text = "_floor({0})", helper = ["floor"] }
"f64.ceil" = { text = "_ceil({0})", helper = ["ceil"] }
"f64.round" = { text = "_round({0})", helper = ["round"] }
"f64.min" = "min({0}, {1})"
"f64.max" = "max({0}, {1})"
"f64.eq" = "({0} == {1})"
"f64.lt" = "({0} < {1})"
"f64.le" = "({0} <= {1})"
"f64.of_int[int]" = "float({0})"
"f64.trunc[int]" = "int({0})"
"f64.to_f32" = { text = "_f32({0})", helper = ["f32"] }
"f64.bad.add" = "not math.isfinite({0} + {1})"
"f64.bad.sub" = "not math.isfinite({0} - {1})"
"f64.bad.mul" = "not math.isfinite({0} * {1})"
"f64.bad.div" = "({1} == 0.0 or not math.isfinite({0} / {1}))"
"f64.bad.sqrt" = "({0} < 0.0)"
"f64.bad.trunc[int]" = "not ({lo} <= int({0}) <= {hi})"
"f64.bad.to_f32" = { text = "_bad_f32({0})", helper = ["bad_f32"] }
"f32.to_f64" = "{0}"
"bytes.len" = "len({0})"
"bytes.get" = "{0}[{1}]"
"bytes.slice" = "{0}[{1}:{2}]"
"bytes.cat" = "({0} + {1})"
"bytes.eq" = "({0} == {1})"
"bytes.lt" = "({0} < {1})"
"bytes.of_list" = "bytes({0})"
"bytes.to_list" = "list({0})"
"arr.new[plain]" = "[{1}] * {0}"
"arr.new[owned]" = "[copy.deepcopy({1}) for _ in range({0})]"
"arr.len" = "len({0})"
"arr.get[plain]" = "{0}[{1}]"
"arr.get[owned]" = "copy.deepcopy({0}[{1}])"
"arr.set" = [[0, "{0}[{1}] = {2}"], [0, "{x} = {0}"]]
"list.new" = "[]"
"list.len" = "len({0})"
"list.get[plain]" = "{0}[{1}]"
"list.get[owned]" = "copy.deepcopy({0}[{1}])"
"list.set" = [[0, "{0}[{1}] = {2}"], [0, "{x} = {0}"]]
"list.push" = [[0, "{0}.append({1})"], [0, "{x} = {0}"]]
"list.pop" = [[0, "{x1} = {0}.pop()"], [0, "{x0} = {0}"]]
"tup.mk[tup()]" = "()"
"tup.mk" = "({args},)"
"tup.get[plain]" = "{0}[{k}]"
"tup.get[owned]" = "copy.deepcopy({0}[{k}])"
"tup.open[tup()]" = ""
"tup.open" = "{xs}, = {0}"
"sum.mk" = "({k}, {0})"
"sum.tag" = "{0}[0]"
"sum.get" = "{0}[1]"
"copy[plain]" = "{0}"
"copy[owned]" = "copy.deepcopy({0})"
"drop" = ""
"io.byte" = "sys.stdout.buffer.write(bytes(({0},)))"

[helper]
quo = [[0, "def _quo(a, b):"], [1, "q = abs(a) // abs(b)"], [1, "return q if (a < 0) == (b < 0) else -q"]]
rem = [[0, "def _rem(a, b):"], [1, "return a - b * _quo(a, b)"]]
wrap = [[0, "def _wrap(v, lo, hi):"], [1, "return (v - lo) % (hi - lo + 1) + lo"]]
floor = [[0, "def _floor(x):"], [1, "return math.copysign(float(math.floor(x)), x)"]]
ceil = [[0, "def _ceil(x):"], [1, "return math.copysign(float(math.ceil(x)), x)"]]
round = [[0, "def _round(x):"], [1, "t = math.trunc(x)"], [1, "if abs(x - t) >= 0.5:"],
         [2, "t += 1 if x > 0 else -1"], [1, "return math.copysign(float(t), x)"]]
f32 = [[0, "def _f32(x):"], [1, "return struct.unpack('<f', struct.pack('<f', x))[0]"]]
bad_f32 = [[0, "def _bad_f32(x):"], [1, "try:"], [2, "struct.pack('<f', x)"], [2, "return False"],
           [1, "except OverflowError:"], [2, "return True"]]

[host]
file = [[0, "import copy, math, struct, sys"], [0, "{helpers}"], [0, "class Halt(Exception):"],
        [1, "pass"], [0, "G = [{globals}]"], [0, "{defs}"], [0, "{entry}()"]]

[[oracle]]
name = "CPython 3.13.5"
kind = "maintained"
image = "sha256:<written by the first gate run>"
build = ""
run = "python3 -I {src}"
device = "octets"
suite = ""
limit = 600

[[oracle]]
name = "CPython 3.14.5"
kind = "maintained"
image = "sha256:<written by the first gate run>"
build = ""
run = "python3 -I {src}"
device = "octets"
suite = ""
limit = 600
```

The table has 98 primitive rows covering 90 of the 100 primitives, and 13 statement rows. What it leaves to definitions, and why:

- `block` and `break` of depth above 0: Python has neither, so `block` prints as `loop { S; break 0 }` and a deeper break as an exit counter (ONE-15).
- The four f32 operations and their four domain tests: each prints as `f64.to_f32` of the binary64 operation, which is exact by Figueroa's theorem (6.3, the f32 rule).
- `mem.get` and `mem.set`: not used, because the table has `arr` rows.
- Recursion deeper than 400 calls: the explicit stack of ONE-15, because CPython's own limit is near 1,000 frames by default [K].
- Every library function (section 5).

Each row encodes a fact about Python that the row vectors of the gate check: `//` floors, so truncating division is a helper; `math.floor` returns an integer, so the helpers restore the sign of zero (`floor(-0.0)` is −0.0, `ceil(-0.5)` is −0.0, F6, F8); `round` in Python rounds ties to even, so `_round` rounds ties away from zero (F8); `1.0 / 0.0` raises an exception rather than giving infinity, so `bad.div` tests the divisor before dividing; integers are unbounded, so every `bad.*` test computes the exact result and compares it with the bounds. The helpers were checked on the edge values −0.0, ±0.5, ±2.5, 0.49999999999999994, 2.5000000000000004, −5e−324, 1e300 and 2^52 + 1 under CPython 3.13.16 in this work; that is a hand check, not the gate.

### 6.2 The table of FORTRAN 77

```toml
# lang/fortran77/print.toml: the table of FORTRAN 77 (ONE-6 to ONE-8)

[lang]
name = "FORTRAN 77"
defined_by = "ANSI X3.9-1978, Programming Language FORTRAN"
catalogue = "FORTRAN 77"
ext = "f"
word = "2147483648"
depth = 0
memory = 16777216

[layout]
indent = "  "
width = 72
lead = "      "
cont = "     &"
cont_end = ""
label = "{n}"
empty = "CONTINUE"
comment = "C     {text}"

[lex]
local = "V{n}"
global = "GV({n})"
"global[bool]" = "GL({n})"
fn = "F{n}"
reserved = ["CHAR", "GL", "GV", "IABS", "M", "MAX", "MIN", "MOD", "RCE", "RCEBD", "RCEOUT"]
name_max = 6
sep = ", "
true = ".TRUE."
false = ".FALSE."
unset = "0"
int = "{dec}"
neg = "(-{dec})"

[type]
bool = "LOGICAL"
"int@word" = "INTEGER"

[stmt]
bind = "{x} = {e}"
var = "{x} = {0}"
set = "{x} = {0}"
call = "{x} = {f}({args})"
if = [[0, "IF ({0}) THEN"], [1, "{then}"], [0, "ELSE"], [1, "{else}"], [0, "END IF"]]
block = [[1, "{body}"], [0, "CONTINUE", "lx"]]
loop = [[0, "CONTINUE", "lh"], [1, "{body}"], [0, "GO TO {lh}"], [0, "CONTINUE", "lx"]]
break = "GO TO {lx}"
ret = [[0, "{f} = {0}"], [0, "RETURN"]]
fn = [[0, "{t} FUNCTION {f}({params})"], [1, "{decls}"], [1, "{body}"], [0, "END"]]
decl = "{t} {x}"

[prim]
"bool.not" = "(.NOT. {0})"
"bool.and" = "({0} .AND. {1})"
"bool.or" = "({0} .OR. {1})"
"bool.eq" = "({0} .EQV. {1})"
"int.add[int@word]" = "({0} + {1})"
"int.sub[int@word]" = "({0} - {1})"
"int.mul[int@word]" = "({0} * {1})"
"int.neg[int@word]" = "(-{0})"
"int.abs[int@word]" = "IABS({0})"
"int.quo[int@word]" = "({0} / {1})"
"int.rem[int@word]" = "MOD({0}, {1})"
"int.eq[int@word]" = "({0} .EQ. {1})"
"int.lt[int@word]" = "({0} .LT. {1})"
"int.le[int@word]" = "({0} .LE. {1})"
"int.min[int@word]" = "MIN({0}, {1})"
"int.max[int@word]" = "MAX({0}, {1})"
"int.cast[int@word]" = "{0}"
"int.bad.add[int@word]" = "(({0} + {1}) .LT. {lo} .OR. ({0} + {1}) .GT. {hi})"
"int.bad.sub[int@word]" = "(({0} - {1}) .LT. {lo} .OR. ({0} - {1}) .GT. {hi})"
"int.bad.mul[int@word]" = "(({0} * {1}) .LT. {lo} .OR. ({0} * {1}) .GT. {hi})"
"int.bad.neg[int@word]" = "((-{0}) .LT. {lo} .OR. (-{0}) .GT. {hi})"
"int.bad.abs[int@word]" = "(IABS({0}) .GT. {hi})"
"int.bad.quo[int@word]" = "({0} .EQ. {lo} .AND. {1} .EQ. -1)"
"int.bad.cast[int@word]" = "({0} .LT. {lo} .OR. {0} .GT. {hi})"
"mem.get" = "M({0})"
"mem.set" = "M({0}) = {1}"
"copy" = "{0}"
"drop" = ""
"io.byte" = { text = "CALL RCEOUT({0})", helper = ["out"] }

[helper]
out = [[0, "SUBROUTINE RCEOUT(B)"], [1, "INTEGER B, N"], [1, "CHARACTER*1048576 L"],
       [1, "SAVE N, L"], [1, "DATA N /0/"], [1, "IF (B .EQ. 10) THEN"], [2, "IF (N .EQ. 0) THEN"],
       [3, "WRITE (*, 9000)"], [2, "ELSE"], [3, "WRITE (*, 9001) L(1:N)"], [2, "END IF"],
       [2, "N = 0"], [1, "ELSE"], [2, "IF (N .EQ. 1048576) STOP 3"], [2, "N = N + 1"],
       [2, "L(N:N) = CHAR(B)"], [1, "END IF"], [0, "FORMAT ()", "9000"], [0, "FORMAT (A)", "9001"],
       [0, "END"]]

[host]
file = [[0, "PROGRAM RCE"], [1, "{common}"], [1, "INTEGER {entry}"], [1, "INTEGER V0"],
        [1, "V0 = {entry}()"], [0, "END"], [0, "{defs}"], [0, "BLOCK DATA RCEBD"], [1, "{common}"],
        [1, "{ginit}"], [0, "END"], [0, "{helpers}"]]
common = [[0, "INTEGER M(0:{memory})"], [0, "INTEGER GV(0:{ng})"], [0, "LOGICAL GL(0:{ng})"],
          [0, "COMMON /RCEM/ M"], [0, "COMMON /RCEG/ GV, GL"]]
ginit = "DATA {x} /{0}/"

[[oracle]]
name = "GNU Fortran 13.3"
kind = "maintained"
image = "sha256:<written by the first gate run>"
build = "gfortran {src} -o {exe}"
run = "{exe}"
device = "octets"
suite = ""
limit = 600

[[oracle]]
name = "LLVM Flang 20"
kind = "maintained"
image = "sha256:<written by the first gate run>"
build = "flang {src} -o {exe}"
run = "{exe}"
device = "octets"
suite = ""
limit = 600
```

The table has 29 primitive rows and 11 statement rows. It meets the floor through `io.byte`; `if`, `loop` and `break` (FORTRAN 77 has the block `IF` but no `WHILE` and no labelled exit, so a loop is a labelled `CONTINUE` with a `GO TO`); `var` and `set`; `fn`, `call` and `ret`; the integer rows at the word; and `mem` with 16,777,216 words. FORTRAN 77 has none of the five forms the direction names as ones a language may lack (checked integers, maps, recursion, binary64 and byte strings), and each is supplied in FORTRAN 77 by the definitions of section 5, printed through these same rows:

| FORTRAN 77 lacks | Printed instead |
|---|---|
| checked integers (overflow is undefined), and any integer type beyond the INTEGER the table relies on | `bad.*` tests on the exact result where it fits the word; elsewhere a sign and base-2^15 digits in `mem` (a u64 is five digits), with every operation and trap text computed from the exact value |
| bit operations (IAND and IOR are extensions, MIL-STD-1753 [K], not FORTRAN 77) | digit-wise `quo` and `rem` by 2 |
| binary64 (DOUBLE PRECISION is processor-dependent, and the standard permits any mathematically equivalent evaluation that respects parentheses [K]) | software binary64 over the 64 bits held as digits, correctly rounded to nearest even (F1) |
| byte strings (CHARACTER is fixed in length) | lists of bytes in `mem`; `CHAR` appears only in the output helper |
| maps, lists, records, unions (no dynamic storage) | records in `mem` with a free-list allocator; `drop` releases a record |
| recursion | an explicit stack of frames in `mem` and a dispatch loop |
| exceptions | status returns: two globals and a test after each call that can halt |

`word` is 2^31 because every listed toolchain's default INTEGER holds |v| < 2^31; the standard does not fix the width. Listing a toolchain with a 16-bit INTEGER would lower `word` to 32,768, with 7-bit digits, for every toolchain of the table (ONE-11). The output helper writes each line as one formatted record and depends on `CHAR(B)` for B up to 255, which FORTRAN 77 leaves to the processor; the row vectors check all 256 values on each toolchain. In this work the helper was compiled and run on GNU Fortran 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1): all 255 values other than the line end, an empty line and a 300-byte line came back byte for byte, with no diagnostic other than the obsolescence of `CHARACTER*n` under `-std=f95 -pedantic`. The record's length bounds a trace line at 1,048,576 bytes; a longer line stops the program with `STOP 3`, an outcome the gate records as a failure, never as a pass.

### 6.3 One fragment through both tables

The fragment below is the core of a step `step Counter on Inc -> n + 1` whose field is `n: u8`, simplified to show printing: a step function returns its outputs as well (section 3.5). `lib.tick` is the fuel function; `lib.msg.add[u8]` writes "overflow: V is outside u8" with the exact V (`interp.rs:638-648`).

```
fn (v0: tup(int[0,255])) -> tup(int[0,255])
  v1 = tup.get[0](v0)                       ; n
  drop(v0)
  v2 = call lib.tick(3)                     ; the ticks of `+`, `n` and `1`
  v3 = int.bad.add[int[0,255]](v1, 1)
  if v3 { v4 = call lib.msg.add[u8](v1, 1) ; v5 = sum.mk[0](v4) ; halt v5 } else {}
  v6 = int.add[int[0,255]](v1, 1)
  v7 = tup.mk(v6)
  ret v7
```

Python 3 prints it as it is. `drop` prints nothing, `tup.get` and `tup.mk` are rows, and the halt is an exception. The function numbers (`f17` for the fragment, `f3` for `lib.tick`, `f9` for `lib.msg.add[u8]`) are those of this illustration:

```python
# step Counter on Inc, with n: u8 (fragment)
def f17(v0):
    v1 = v0[0]
    v2 = f3(3)
    v3 = not (0 <= v1 + 1 <= 255)
    if v3:
        v4 = f9(v1, 1)
        v5 = (0, v4)
        raise Halt(v5)
    else:
        pass
    v6 = (v1 + 1)
    v7 = (v6,)
    return v7
```

The FORTRAN 77 table has no rows for `tup`, `sum` or `halt`, so before printing, the definitions of ONE-14 and ONE-15 rewrite the fragment into core that uses only `int@word`, `bool`, `mem`, `call` and globals: the tuple becomes an address in `mem`, `drop` a call of the allocator's release (`f202`), `sum.mk` and `tup.mk` calls of record constructors (`f205`, `f206`), and the halt the status globals `g1` (halted) and `g2` (the Halt). Locals are renumbered in textual order. The same rows then print:

```fortran
C     step Counter on Inc, with n: u8 (fragment)
      INTEGER FUNCTION F17(V0)
        INTEGER V0
        INTEGER V1
        INTEGER V2
        INTEGER V3
        INTEGER V4
        LOGICAL V5
        LOGICAL V6
        INTEGER V7
        INTEGER V8
        INTEGER V9
        INTEGER V10
        INTEGER F202
        INTEGER F3
        INTEGER F9
        INTEGER F205
        INTEGER F206
        INTEGER M(0:16777216)
        INTEGER GV(0:3)
        LOGICAL GL(0:3)
        COMMON /RCEM/ M
        COMMON /RCEG/ GV, GL
        V1 = (V0 + 1)
        V2 = M(V1)
        V3 = F202(V0)
        V4 = F3(3)
        V5 = GL(1)
        IF (V5) THEN
          F17 = 0
          RETURN
        ELSE
          CONTINUE
        END IF
        V6 = ((V2 + 1) .LT. 0 .OR. (V2 + 1) .GT. 255)
        IF (V6) THEN
          V7 = F9(V2, 1)
          V8 = F205(0, V7)
          GV(2) = V8
          GL(1) = .TRUE.
          F17 = 0
          RETURN
        ELSE
          CONTINUE
        END IF
        V9 = (V2 + 1)
        V10 = F206(V9)
        F17 = V10
        RETURN
      END
```

The status test after `F3(3)` exists because `lib.tick` can halt ("comptime fuel exhausted"); there is none after `F9`, which cannot. With `n: u64` instead of `n: u8`, the Python text changes in one place, the bound `18446744073709551615` that `{hi}` prints. In FORTRAN 77, u64 is outside the word, so `int.bad.add` and `int.add` at u64 have no row: n is carried as a record of a sign and five digits, and those two lines become calls of the printed digit definitions, with the literal 1 built as a record first. The text of the rows is the same; only the core before printing differs.

Both texts were produced in this work by a provisional printer, written for this check, that implements only the constructs of this fragment and of one loop, reading the two tables above verbatim. The Python ran under CPython 3.13.16 with stand-ins for `f3` and `f9`, and returned `(255,)` for n = 254 and raised the Halt `(0, b'overflow: 256 is outside u8')` for n = 255. The FORTRAN compiled and ran under GNU Fortran 13.3.0 with stand-ins for the five called functions and gave the same two outcomes through the status globals. The loop printed through the `loop`, `if` and `break` rows with its labels in columns 1 to 5, and a line longer than 72 columns was continued with `&` in column 6. That checks the format and these rows on one fragment; it is not the gate.

## 7. The gate

**ONE-19 The gate's stages.** A table's gate runs, on every toolchain of its oracle list:

| Stage | Checks | Expected bytes from |
|---|---|---|
| G1 rows | every row the table has, on vectors at the edges of its type argument: lo, hi, lo + 1, hi − 1, 0, ±1; inputs on both sides of every `bad.*` boundary; for f64, ±0.0, 5e−324, 2.2250738585072014e−308, 1.7976931348623157e308, sums and products that overflow, the ties ±0.5, ±2.5 and 0.49999999999999994; for bytes, all 256 values and lengths 0, 1 and N9; for lists, sizes 0 and N9; every lexical row on its extremes | the reference, stored as data in `lang/vectors/` |
| G2 library | the 77 canonical JSON vectors (`spec/data/canonical-json-vectors.json`), SHA-256 vectors, float text vectors (`spec/examples/57_float_vectors.rce`, `tests/traces.rs`), display vectors including the six probe strings of section 0 | the reference |
| G3 examples | every program of `spec/examples` that `rce check` accepts (60 files; the 62-program corpus of `tests/emit.rs` with its probes), the edge program of `tests/emit_edge.rce`, and `spec/traces/` | the reference's trace |
| G4 conformance | every test of `conformance/tests` whose expectation is an observation, a trace or a run of `fn main` | the reference's trace |
| G5 fuel | the sweep of `tests/fuel.rs`: ten programs at every limit from 0 to 300 and at 400 limits above, each limit a change of one global literal of the core | the reference's trace at the same limit |
| G6 lens | where a lift λ_L exists, hash(λ_L(π_L(F))) = hash(F) on the corpus | encapsulation (`encapsulation.md`); not part of passing |

A case passes when the oracle exits with status 0 and its standard output, read through the oracle's device code, equals the expected bytes exactly. A table passes when G1 to G5 pass on every toolchain of its oracle list and its failure list is empty. Rows are gated before programs, so a native operator is used only after its edge vectors pass. Kaitai Struct's issue #746, open since May 2020, has eight of twelve targets printing a truncating `/` against a specification and a test that expect floor division; under G1 such a row fails before any program is printed with it.

**ONE-20 The oracle record.** Each `[[oracle]]` entry of a table has:

| Field | Meaning |
|---|---|
| `name` | the implementation and its release |
| `kind` | `maintained`, `emulated` or `reconstructed`; the kinds follow |
| `image` | the SHA-256 digest of the toolchain image; for `emulated`, the image holds the emulator built from a stated source commit, the disk images of the period system with their SHA-256, and the job template that submits `{src}` and collects the output |
| `build`, `run` | the commands, with `{src}` and `{exe}`. A command may name the language definition the implementation is to accept (such as `-std=c99`) and may isolate the run from its environment (such as `python3 -I`); it carries no option that changes the meaning of the text: no contraction, overflow, fast-mathematics or optimisation option. Where a language's meaning needs such an option, the row it would serve is absent and the definition prints |
| `device` | `octets` when the implementation writes bytes; otherwise the name of a code, stated in the record, that maps the device's output (a printer file with a carriage-control column, a numeric printer, a storage buffer) back to bytes. A code is injective and fixed, and the comparison is on the decoded bytes |
| `suite` | the language's own validation suite and the implementation's result on it, when one exists |
| `limit` | seconds per case |

Each case runs twice and must give the same bytes both times, so that the oracle itself is deterministic.

**ONE-21 Oracle kinds, and how historical languages run.**

- **Maintained**: an implementation of the language definition maintained today. GNU Fortran and LLVM Flang for FORTRAN 77; GnuCOBOL for COBOL-85, whose baseline on CCVS85, the COBOL Compiler Validation System of the National Institute of Standards and Technology (NIST), is 9,610 of 9,748 tests passing and none failing (`tests/cobol85/summary.txt`, GnuCOBOL at b3959a62); MARST for ALGOL 60, which translates the Modified Report's language to C and carries Brian Wichmann's 130 validation tests (search); cim for Simula; GNU APL for ISO 13751 APL; the Cintcode system for BCPL; Free Pascal in its ISO mode for ISO 7185 Pascal (search).
- **Emulated**: an implementation from the period of the definition, run on an emulator of its machine. SIMH (open-simh at 87eb7d5e) covers about 45 machine families, from the Manchester Baby and the IBM 650, 1401, 1620 and 7094 to the PDP-1 to PDP-11, the VAX and the Burroughs B5500; Hercules runs IBM's own FORTRAN G and H, PL/I(F), ANS COBOL and ALGOL F under MVS 3.8j [K]. The image pins the emulator by source commit and the system disks by SHA-256, and the job template is part of the image.
- **Reconstructed**: an implementation written after the period from the defining document, where no period implementation survives or none was ever built; it is cited, or written in this repository as an RCE program from the document, with the section of the document each construct the table uses comes from.

The `suite` field records the language's historical validation suite where one exists: CCVS85 for COBOL, Wichmann's tests for ALGOL 60, the NIST FORTRAN Compiler Validation System for FORTRAN 77, the Pascal Validation Suite, and the Ada Conformity Assessment Test Suite (ACATS) for Ada [K]. The counter reports each kind in its own column, because a gate on a reconstructed implementation checks the table against a reading of the document, not against an independent implementation of the period.

**ONE-22 CI and the ratchet.** `.github/workflows/lang.yml` runs one job per table and toolchain, pulling the image by its digest. On every push it runs G1 to G3 for each table whose inputs changed: the core, the lowering, the definitions, the library, the table or the image. Every night, and before every release label (C.15), it runs G1 to G5 for every table. A table's failure list, `lang/<L>/failing.txt`, names the cases that fail. CI fails when a case outside the list fails, and also when a case on the list passes, until the list is shortened (the two-way list of protobuf's conformance suite, `conformance_test.cc` 368-371, protobuf at 1d3458ec). A list never grows: a commit that would make a passing case fail changes the table, the definition or the library in the same commit (CONTRIBUTING rule 3). Thrift's 982 known cross-test failures (`test/known_failures_Linux.json`, Thrift at f0eb8b84) show what a list that may grow becomes.

**ONE-23 States.** Each language in the registry is in exactly one state:

- **registered**: a row in the registry, no table;
- **printed**: a complete table with no oracle record yet, or with only oracles still being built;
- **failing**: a gate that runs, with a non-empty failure list;
- **passing**: G1 to G5 green on every listed toolchain.

These are states of evidence, never classes of language. Every language moves through them on the same terms, and none is held at a lower class because of its arithmetic, its data or its age.

## 8. Migration

**ONE-24 Rust, Python and JavaScript move with identical traces.** The four existing targets become the first four tables (`lang/rust2024`, `lang/python3`, `lang/js-es2020`, `lang/scala3`), and the target names of `rce emit` become those table names. Each initially carries integers as today's runtime does, so that no trace moves: Rust as i128 checked against each type's range (`rt.rs:282-340`), Python as unbounded integers, JavaScript as BigInt for every integer type (`rt.js:35-39`). For Rust, Python and JavaScript the move is complete when:

1. on every program of G3 and every case of G4 that the old emitter passes, the new program prints the bytes of the reference's trace, and the old program's output under `--trace --fuel-trace` and `--run` equals the same bytes, so the old emitter serves as a second oracle during the move;
2. `tests/emit.rs`, `fuel.rs`, `write_back.rs`, `map_order.rs`, `canonical_json.rs` and `traces.rs` stay green, rewritten only where they pass arguments that ONE-5 retires;
3. the conformance counts are at least 1,459 passing and 0 failing for each, and the cases that were not applicable because of a construct of the table of section 3.5 become applicable.

Then the per-target paths of `emit/*.rs` and the four runtimes are deleted in one commit. The six probe strings of section 0 are expected to change the Python and JavaScript output when ONE-18 lands, toward the reference; that is a correction counted by G2, not a regression.

**ONE-25 Scala is fixed by the move, and the fix is checked.** Before the move, the current Scala adapter is run once more with Scala 3.10.0 and version 21 of the Java Virtual Machine (JVM) pinned (as `tests/emit.rs:82` pins them and `emit_adapter.py` does not), with one build directory per case, and with the whole of standard error kept, so that the cause of each of the 1,189 failures is recorded. The move then removes each recorded cause, or the record shows which one it does not:

- the failing path is the `--run` path, `run_tasks` and `Sched` (`rt.scala:1500-1907`), JVM threads and `LinkedBlockingQueue`; the scheduler becomes library code, the same on every table;
- the compiler's "Non local returns" diagnostic at `lvar_above`, a warning or an error depending on the Scala version, cannot arise: the core has no lambdas (defunctionalised) and every `ret` returns from its own function;
- the global `FUEL` (`rt.scala:46-74`) has no threads to be wrong for: the scheduler saves and restores the fuel of each task;
- `NoClassDefFoundError: UnitT$`, consistent with parallel builds sharing one `.scala-build` directory (`emit_adapter.py:22-23,52`), is removed by the oracle record's isolated runs;
- break and continue, emulated today by exceptions (`try { while { try {..} catch Cont } } catch Brk`), become the exit counter of ONE-15 or rows for `scala.util.boundary`, whichever the row vectors admit;
- the narrow escape table (`rt.scala:191-200`) becomes `rt/show`.

Scala then counts when its gate passes, like any other table.

## 9. Waves of tables

**ONE-26 Waves.** Tables are written in five waves. Within a wave, a language with a maintained oracle comes before one with an emulated oracle, which comes before one with a reconstructed oracle, and then registry order applies. No wave waits for a later one, and a language outside every wave is written when its table is.

- **Wave 0, migration**: Rust, Python, JavaScript, Scala (section 8).
- **Wave 1, the modern systems and platform languages**: C (ISO C99), Go, Zig, TypeScript, Java, Kotlin, Swift, C#, Fortran 2008, C++, then the languages of the registry's Tier 2 (Ruby to Tcl, with Lua 5.1 and Lua 5.4 as two languages) and the present-day languages of the former Tier 4 (Io, Clean, Idris, ATS, Koka, Effekt, Roc, Unison, Verse, Carbon, Hylo, Vale, Austral, Cyclone, Jai, Nickel, CUE, Dhall, Pkl, Scratch). The C table needs no `#pragma`: the core is in ANF, one floating operation per statement, and ISO C permits contraction only within one expression, while an assignment removes extra range and precision (ISO/IEC 9899:1999, 6.5 paragraph 8 and 5.1.2.3 [K]).
- **Wave 2, the historical languages**: first those with maintained oracles (FORTRAN 77, COBOL-85, ALGOL 60, ALGOL 68, Simula 67, BCPL, ISO 7185 Pascal, Ada 83, PL/I, APL, SNOBOL4, Prolog, Smalltalk-80, Forth, MUMPS, REXX, Modula-2, Oberon, Icon, Logo, Minimal BASIC; the maintained implementations of the languages after APL in this list are named from knowledge [K]), then the period implementations under SIMH and Hercules (FORTRAN II and IV, LISP 1.5, FOCAL, Burroughs ALGOL, IBM PL/I(F) and ANS COBOL), then the reconstructed ones. The registry's grammar source for these is ANTLR grammars-v4, which has most of the historical languages the tree-sitter list lacks (algol60, apl, bcpl, cobol85, focal, fortran77, logo, mumps, pl1, simula, snobol, and Ada 83 to 2022); emission needs no grammar.
- **Wave 3, the entrants of 2024 to 2026 already registered**: the 72 languages for or by machine authors and the 39 general-purpose languages of Tier 5. Each oracle is the implementation at a pinned commit; an archived implementation is pinned at its last commit. An implementation that consults a language model to compile or to run needs a compile step that is pinned, local and deterministic (ONE-20, two identical runs); until one exists, its table stays at "printed".
- **Wave 4, hardware and intermediate representations**, read with `one-library-every-fabric.md`:
  - **C**, from wave 1, is also the table for hosts without an operating system (XF-13): the same table on a microcontroller toolchain, with `io.byte` reaching a serial port and the oracle's device code reading it.
  - **LLVM IR**, the textual form at one LLVM release, run by `lli` and by `clang`: integers of any width give native rows at every integer type, the `llvm.*.with.overflow` intrinsics give `bad.*` rows, and `fadd`, `fmul` and the other operations without fast-mathematics flags give binary64 rows; output through a declared `putchar` or `write`.
  - **WebAssembly**, in the text format, run by a pinned wasmtime: `i32` and `i64` with explicit overflow tests, IEEE binary64 whose only nondeterminism, NaN payloads, F4 and F5 keep out of every value, and output through WASI (the WebAssembly System Interface) `fd_write`. A component is the adapter of XF-5. Direct emission removes rustc from the path (Tier 3).
  - **Verilog** (IEEE 1364-2005) and VHDL (IEEE 1076), run by pinned simulators (Verilator, Icarus Verilog, GHDL) with a test bench that reads a byte-stream port: each core function prints as a state machine with one state per statement group, `mem` is a memory array, recursion uses the explicit stack, and state sits in registers when its size is static and in addressed memory otherwise (XF-12). A passing Verilog gate is a simulation result; synthesis and a board run are separate evidence (XF-12, `energy-calibration.md`), and the counter claims neither.
  - **Graphics processors** (WGSL, SPIR-V, CUDA, OpenCL, GLSL, HLSL, MSL): WGSL has no binary64 and no recursion, so binary64 is the software definition over u32 and recursion the explicit stack; output goes to a storage buffer that the device code reads. The binary32 class of XF-11 governs only speed rows.
  - **The other rows of Tier 3** (SQL, Datalog, TLA+, Lean, Coq, spreadsheet formulas, eBPF, Solidity and the rest) meet the floor through their own constructs: recursive queries, fixpoints, recursion on a fuel bound, bounded loops. Each table follows when its floor rows pass G1.

## 10. The public counter

**ONE-27 The counter.** `scripts/claim-counts.py` computes, from the tables, their oracle records, the CI results and the catalogue snapshot, one line that `lang/REGISTRY.md` opens with and `README.md` quotes, and `claim-counts.py --check` fails CI when the quoted line differs from the computed one:

> Languages with a passing gate: N of M entries of HOPL (snapshot DATE, SHA-256 …), and K not yet in that snapshot. By oracle: maintained a, emulated b, reconstructed c. Printed without an oracle: p. Failing: f, with s cases on their lists. Registered without a table: r.

- **The catalogue.** HOPL, the online History of Programming Languages, lists 8,945 languages from the 18th century to the present (search). It is the only public catalogue found that spans every period, which the direction requires. The snapshot is `lang/catalogue/hopl.tsv` (the entry's identifier, name and year), committed with the date it was taken and its SHA-256. A commit that refreshes it changes the denominator and nothing else. hopl.info could not be reached from the environment of this work, so the first snapshot fixes M; until then the counter prints M as unknown.
- **What counts.** Only passing tables count. A table counts for the one entry its `catalogue` field names, never for the entries of other languages its text also runs on. A table whose entry is not in the snapshot counts in K, which is where a language designed after the snapshot appears first.
- **By period.** `lang/REGISTRY.md` also gives, for each decade of first appearance in the snapshot, the passing languages and the entries, so that coverage of past periods is visible beside the total.
- **A second, machine-readable column.** The registry also prints the count against the 562 programming languages of GitHub Linguist (`lib/linguist/languages.yml` at 73ee9a3b), because that list is pinned by commit. HOPL remains the denominator of the counter.
- **The value at `d91a9fc8`** is 0 of M: no table exists. For scale, systems that keep semantics consistent across targets maintain about 10 to 30 each (ANTLR 10, Kaitai Struct 13, Haxe 9, Thrift about 28, Fable 6); the lexical catalogues hold 443 to 836 entries; Rosetta Code's 1,001 language directories (mirror of 2026-07-16) are written by hand, with a median of 15 tasks each. A count of thousands is reachable only if a language costs one file and every table reports its state.

## 11. Changes to `emit-targets.md`

1. **Lines 3-6**: add this proposal to what the registry is derived from.
2. **Lines 14-19**: the counts become 5,790 generator lines with 498 per-target branches and 8,169 runtime lines; "a single emitter that walks the Form IR and reads a per-target table" becomes the emitter of ONE-1, which walks the core.
3. **Section 0, lines 23-41**: `lang/L/print.toml` is the table of ONE-6 to ONE-8, with its oracle records. The sentence that a table "refuses" what a language leaves undefined (`:40-41`) becomes, for emission: no row relies on behaviour the language leaves undefined, and the construct prints from its definition. The lift's handling of such behaviour is left to the lift's own text, and this proposal does not change it.
4. **Section 1, lines 52-68**: the table of fields (`int.checked`, `int.div`, `int.rem`, `float.strict`, `text.unit`, `map.order`, `option`, `error`, `sha256`, `canon_json`, `spawn`, `par`) is replaced by the rows of ONE-7 and ONE-8: text is bytes, maps are library code, options and errors are sums, tasks are state machines (ONE-4), and `par` is sequential in source order. `cost.regime` moves to `energy-calibration.md`, because it calibrates joules and prints nothing.
5. **Lines 70-74**: "one runtime kernel per target, ~300 lines … hand-written where it does not" becomes the library of ONE-16, written once in RCE; nothing is written per language.
6. **Lines 76-81**: the gate (a) to (e) becomes ONE-19 to ONE-23. "Registered" at (a) to (c) becomes "passing" at G1 to G5 on every listed toolchain; "encapsulated" at (d) and (e) is unchanged.
7. **Lines 85-87 and the column "Leaks repaired by the table"**: the column becomes "Rows the table lacks", generated from the tables rather than written by hand.
8. **Tier 0, lines 91-97**: Scala's status becomes "failing: 52 pass, 1,189 fail, results stale (ONE-25)". XState and SCXML lose "control projection only": they are full targets through their data models, and their T9-F standing is unchanged.
9. **Tier 3, lines 145-167**: Verilog, SystemVerilog and VHDL read "state in registers when its size is static, in addressed memory otherwise" in place of "bounded state only" (XF-12 makes the same change); APL, J and K/q read "lane-wise operations print as array rows; the rest prints through definitions" in place of "lane-wise arrays only"; the graphics-processor row reads "whole programs print through definitions; device regions (5.13) are where rows add speed" in place of "device regions (5.13) only"; eBPF reads "the bounded loops of 6.7 map directly"; Regex, JSON Schema, Protocol Buffers and Cap'n Proto read "held by the types and values of S and E (ONE-10)" in place of "not programs".
10. **Tier 4, lines 169-176**, "registered for completeness, no platform of their own", is removed. Its historical languages join wave 2 and its present-day languages wave 1 (ONE-26). The sentence "Each gets a table row when someone needs it; none blocks the encapsulation criterion" is removed.
11. **Tier 5, lines 184-190**: "a language whose semantics the table cannot pin to 6.3 is observation-only (section 3)" is removed; every entrant is a full target in wave 3.
12. **Section 3, lines 338-347**, "What is refused", becomes "How each language is held". The observation-only paragraph is removed. "Hand-written emitters after the one-emitter change" becomes "a language enters as one table file; no code is written for a language". "Any target-specific semantics … a target that cannot mean that thing is not a target" becomes "the Form has one meaning on every target; where a language's native operation means something else, the table has no row for it, and the definition prints the Form's meaning in that language".
13. **Section 4, lines 349-359**: the order of work becomes ONE-26 and the order of work at the end of this proposal; the generated registry is `lang/REGISTRY.md`, with the counter of ONE-27.

## 12. Changes to the specification and the ledger

- **T9 (`RCE-SPEC.md:2402-2412`)**: T9-L becomes the set of tables whose gate passes (ONE-23), read from `lang/REGISTRY.md`, in place of `rce emit rust | python | js`.
- **10.5 (`:3066-3200`)**: the table of targets gains `core` and `trace` and refers every other name to `lang/<L>/print.toml`; `rust`, `python`, `js` and `scala` become `rust2024`, `python3`, `js-es2020` and `scala3`. The sentence "a target that the table cannot pin to 6.3 is observation-only, never a projection" (`:3087-3088`) is removed. The program emitters read the core, not the checked module. `--fuel=N`, `--fuel-trace`, `--trace`, `--run` and `--fuel-batch` are retired in favour of the one output of ONE-5. "Every task is a thread (a worker in JavaScript)" becomes "every task is a state machine of the library scheduler".
- **10.5, the canonical-order sentence (`:3078`)**: "The Form JSON lists objects in canonical order (6.1)" does not hold of the bytes `rce emit form` writes (top-level keys run `S, E, errors, …`); the text or the output is corrected.
- **6.3 N7**: the Unicode pin extends to the escape set of display, enumerated as data (ONE-18), and `tests/unicode_pin.rs` gains a scan for `{:?}` on text.
- **6.3 N9**: VALUE_LIMIT is stated but implemented nowhere in `spec/checker/src` or its tests; it is implemented in the reference, or the text is changed, before G1 vectors at N9 are generated.
- **C.6.3**: "`par` and `for ... par` charge `c_fork` per branch (or chunk)" (`:3839`) disagrees with the reference, which charges none in `forpar_exec` (`conc.rs:869-919`); since the chunk count depends on the host's processor count (`rt.py:1522-1526`), the text drops the per-chunk charge, or the chunk count is fixed by the text.
- **F.5.4, W-CTFN-2**: an emitter prints a module with a `ct fn`; the registry reports, for each table, whether a timing measurement of its printed `ct fn` code exists.
- **Appendix F, W-RES-1 for `Shell.cmd`, `ShellArg`, `Cap.narrow`, `ct_choose` and contract clauses**: "run by the reference evaluator only" becomes lowered to the core (section 3.5).
- **`spec/proposals/LEDGER.md`**: one row, "One emitter, and each language as one table (`one-emitter.md`): ONE-1 to ONE-27", decision open.

## 13. Order of work

This expands step 1 of `emit-targets.md` section 4.

0. **Scala's record.** Re-run the Scala conformance as ONE-25 states, with the full standard error kept.
1. **The typed tree.** The checker records the static type, the binding and the resolved method of every node.
2. **The core.** Its JSON, its checker (W1 to W6), a core evaluator (`rce run-core`), and `rce emit core`.
3. **The lowering**, construct by construct, each tested differentially against `interp` on `spec/examples` and the 1,979 conformance tests, the constructs of Appendix F included.
4. **The definitions and the library** (sections 5.2 to 5.6), each tested once against the reference on the vectors of G1 and G2 and on `spec/traces/`.
5. **The printer and the four tables of wave 0**, with the migration gates of ONE-24 and ONE-25; then the per-target code is deleted.
6. **The gate in CI**: `lang/vectors/`, the oracle images, `lang.yml`, the failure lists and the counter.
7. **Waves 1 to 4**, in the order of ONE-26.
8. **Lean.** `RCE.Core` (`Driver`, `Events`, `Log`) is extended to the core, the lowering is proved against it, and each definition is proved once, as `loop_correct` is. T9 for a table then rests on its rows alone, each checked by G1; today it rests on a whole emitter and a whole runtime per target.

## 14. What stays open

1. **The reading of "no version selection."** This proposal reads it as ONE-12: one table per language definition, many toolchains per table, and oracle commands that may name the definition an implementation is to accept (`-std=c99`, Free Pascal's ISO mode) but nothing that changes meaning. That reading needs confirming.
2. **Memory of a table without aggregate rows.** `memory` bounds the live data of every program printed through such a table. N9 bounds each value, not the total live data of a run, so no fixed `memory` is shown to suffice for every program; a program that exceeds it stops with a distinct status, which the gate records as a failure. Whether the specification should bound total live data is open.
3. **Speed.** Software binary64 and digit arithmetic are correct and slow, especially on emulated period machines. Rows add speed and are gated one by one; `limit` per case is a guess until measured.
4. **Licences.** Some period systems that run under emulation may not be redistributable as CI images. Such an oracle runs on a runner that holds the licence, and its results are published with the image digest.
5. **Reconstructed oracles.** Which documents, and whose reading of them, serve as the oracle for a language with no surviving implementation is open; the counter keeps them in their own column.
6. **Host mode.** Appendix E needs an input construct beside `out`; this proposal covers the trace only.
7. **Entrants that consult a model** at compile or run time (wave 3) have no deterministic oracle until a pinned local compile step exists.
8. **Constant time.** A timing gate for `ct fn` per table is not specified here.
9. **The copy of the pre-state** for each event costs O(|state|). An elision rule (pass the original when no halt can follow the first consuming write) is checkable through W4's dominance form and is not yet stated.
10. **The size of a text under N9.** The reference counts characters (`interp.rs:236`), while N7 counts bytes for `len`.
11. **Device codes.** Whether an injective decoding of a device's output meets "byte for byte" should be confirmed; this proposal uses the identity wherever an implementation writes octets.

## Sources outside this repository

In order of first citation.

- HOPL, the online History of Programming Languages, hopl.info (search; not reached from this work).
- gfortran `-std`; GnuCOBOL `config/*.conf` and `-std` (OCamlPro mirror at b3959a62); Free Pascal `{$mode}` (search); Nim 2.0 changelog, `--jsbigint64` (Nim at c382d74a); SymPy `sympy/printing/fortran.py` (at 9a1e9e24); Haxe `#if` (at 931ad718); protobuf `descriptor.proto` 714-735 and Editions (at 1d3458ec).
- D. E. Knuth, The Art of Computer Programming, volume 2, section 4.3.1, Algorithm D [K].
- IEEE 754-2019, Standard for Floating-Point Arithmetic.
- J. R. Hauser, Berkeley SoftFloat [K].
- Haxe `std/haxe/Int32.hx` and `std/haxe/Int64.hx` (at 931ad718).
- B. W. Kernighan, "Ratfor: a preprocessor for a rational Fortran", Software: Practice and Experience 5(4), 1975 (search).
- G. L. Steele and J. L. White, "How to print floating-point numbers accurately", PLDI 1990 [K]; U. Adams, "Ryū: fast float-to-string conversion", PLDI 2018 [K].
- FIPS 180-4, Secure Hash Standard.
- The Python Language Reference, release 3.13, docs.python.org/3/reference.
- ANSI X3.9-1978, Programming Language FORTRAN.
- MIL-STD-1753, FORTRAN, DoD Supplement to American National Standard X3.9-1978 [K].
- GNU Fortran (GCC) 13.3.0, Ubuntu 13.3.0-6ubuntu2~24.04.1, used for the hand checks of sections 6.2 and 6.3.
- Kaitai Struct issue #746, github.com/kaitai-io/kaitai_struct/issues/746 (web).
- NIST CCVS85 results in GnuCOBOL `tests/cobol85/summary.txt` (at b3959a62).
- MARST, gnu.org/software/marst, release 2.6 with Wichmann's validation tests (search); cim, gnu.org/s/cim (search); GNU APL (search); BCPL Cintcode, cl.cam.ac.uk/~mr10/BCPL.html (search); Free Pascal ISO mode (search).
- SIMH, open-simh at 87eb7d5e; Hercules and MVS 3.8j [K].
- NIST FORTRAN Compiler Validation System, the Pascal Validation Suite, ACATS [K].
- protobuf conformance suite, `conformance/conformance_test.cc` 368-371 and the per-language failure lists (at 1d3458ec).
- Apache Thrift, `test/known_failures_Linux.json` (at f0eb8b84).
- ISO/IEC 9899:1999, Programming languages: C, 6.5 paragraph 8 and 5.1.2.3 [K].
- ANTLR grammars-v4 (at bfb7229f); tree-sitter wiki, List of parsers (at 85fe7ee0).
- WebAssembly core specification and WASI [K]; IEEE 1364-2005, Verilog Hardware Description Language [K].
- GitHub Linguist, `lib/linguist/languages.yml` (at 73ee9a3b).
- ANTLR 4 (at 7d577039), Kaitai Struct compiler (at 5c22f31f), Fable (at be20361f), for the counts of section 10; Rosetta Code data mirror, acmeism/RosettaCodeData (snapshot of 2026-07-16).
