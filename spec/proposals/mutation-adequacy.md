# Proposal: mutation adequacy, the information an obligation carries

Status: open. The first slice is implemented (ledger row of 2026-11-02):
the mutant set, the scores and the survivors of section 3 as
`rce mutants FILE` (10.14 of the specification), W-MUT-1 in `rce check`
and W-MUT-2 in `rce mutants`. It is specified in sections 6.12, 7, 7.1,
10.2 and 10.14 of the specification. Owner: David Charlot.

The proposal is derived from 6.9 (examples and checks as admission), 6.10
(behaviour and `rce equiv`), 6.11 (laws), 10.13 (`rce repair`, whose rung d
supplies the edits), [intent-holes.md](intent-holes.md) (vacuous intent,
W-HOLE-1 and W-HOLE-2) and the ledger row of 2026-10-23 on unobservable
intent (11 of 13 bounded state machines have examples that read fields no
step emits). Related work that also judges obligations by the changes they
catch: Thermite, Prove and Vow set mutation challenges, and Djinnlang
requires each specification to be unambiguous (the languages are listed in
[emit-targets.md](emit-targets.md)).

## Terms

- **Obligation**: an `example` (6.9), a `check` (6.9) or a `law` (6.11) of
  the program's own text. `check`, `requires` and `ensures` statements in
  a body are parts of that body, as in `rce repair`.
- **Definition**: a function, a method of an `impl`, a step, an actor, a
  constant or a state declaration of the program's own text (the terms of
  10.13). Definitions of imported modules are not the program's.
- **Single edit**: one edit of rung d of `rce repair` (10.13) at one
  expression of a definition: a binary operator replaced by another of its
  class (`+ - * / %`, `< <= > >=`, `== !=`, `and or`, `& | ^`, `<< >>`),
  the operands of an ordering comparison swapped, an integer literal n
  replaced by n + 1, n - 1, 0 or 1 (at least 0, not n), a boolean literal
  negated, two arguments of a call with two to four arguments swapped, a
  condition negated or its `not` removed.
- **Mutant**: the program with one single edit. The **mutant set** is every
  mutant, in a fixed order: definitions in source order, expressions in
  preorder, edits in the order above. Only the first MUTANT_BOUND (2,000,
  the default bound of rung d) are judged, and every report gives the bound
  and the number of single edits.
- **In Form**: a mutant the checker gives the reports the program has (none,
  for an admitted program), whose constants and initial state evaluate. A
  mutant that is not in Form is not a program and is counted apart.
- **Kill**: an obligation kills an in-Form mutant when it fails on it: an
  example or a check does not hold, or a law fails.
- **Equivalent**: an in-Form mutant that no obligation kills and that
  `rce equiv` (10.11) judges to have the program's behaviour, exactly (both
  explorations closed). It is not a change of behaviour, so it is excluded
  from the score and counted apart. Killing beats equivalence: an
  obligation that fails on a mutant has observed a difference, so a killed
  mutant is never excluded.
- **Survivor**: an in-Form mutant that no obligation kills and that is not
  equivalent.

## In plain words

1. An obligation tells a program apart from the programs it rejects. If no
   single edit of the program makes the obligation fail, the obligation
   tells the program apart from none of them: within those edits it
   carries zero bits about the program's behaviour.
2. The checker makes every single edit, judges each with every obligation
   that can see it, and counts what each obligation kills.
3. An obligation that kills nothing requires change (W-MUT-1). A mutant
   that nothing kills is reported with its diff and, where `rce equiv`
   finds one, the events after which it behaves differently (W-MUT-2, a
   recommendation).

## 1. The principle

An obligation O is a predicate over programs, and admission keeps the
programs on which it holds. Over a finite set M of programs, the
information O gives about which program of M is at hand is the entropy of
its verdict, H(p) for the fraction p of M on which it fails. When O holds
on every member of M, p = 0 and H = 0: O carries no bits about M. Taking M
to be the program and its mutants, an obligation that no mutant fails
carries zero bits about the program's behaviour within those edits; one
that fails on k of n non-equivalent mutants partitions them, and the
union of all obligations' kills is the part of the mutant set the
obligations together rule out.

The measure is relative to the edits. A program with no expression the
edits can change (a step `-> p`) has no mutant there, and an obligation
about it is not measured, rather than called empty (section 4).

## 2. The mutant set

The edits are rung d's, reused from `spec/checker/src/repair.rs`
(`small_edits`), so the two tools agree on what a single edit is. They are
applied to every expression of every definition of the program's own text,
including constants and state declarations, and never to an example,
check or law line: those are what a mutant is judged by. Imported modules
are measured by their own obligations, so their definitions are not
mutated. The order is that of `rce repair`'s enumeration, so the first
MUTANT_BOUND mutants are fixed by the text, and so is admission.

## 3. Judging, scores and survivors

Each in-Form mutant is run on each obligation whose **reach** holds the
edited definition: for a check, the functions and constants it names and
what they name; for an example or a law, every step, actor and state
declaration and what they name. A definition outside the reach is never
evaluated by the obligation, so the run would equal the program's.

A run on a mutant has a fuel budget of twice the fuel the obligation took
on the program plus 100,000 (never more than the fuel of 6.7), and a law
a state bound of four times the states it explored (at least 10,000, at
most LAW_FUEL). A run that exhausts its budget below the fuel of 6.7 is
**undecided** for that obligation: neither a kill nor a hold. A mutant
undecided for some obligation and killed by none is counted apart.

- **Per obligation**: the mutants it kills, out of the in-Form mutants of
  its reach.
- **Per program**: killed ÷ non-equivalent, where non-equivalent = killed
  + survivors. Mutants not in Form, equivalent or undecided are counted
  apart.
- **Survivors**: each survivor is listed with its edit, its diff on the
  canonical text (`rce repair`'s form, pasteable as the change), and the
  shortest event sequence after which the program and the mutant give
  different outputs, from `rce equiv`. `rce equiv` compares the behaviour
  of the steps (6.10), so a survivor in a definition no step reaches, or
  one `fn main` reaches, or in a program with no step, is listed as not
  compared, with the reason.

## 4. The two rules

**W-MUT-1 (vacuous obligation).** An obligation requires change when it
is **measured** and holds, decided, on every in-Form mutant of its reach.

An obligation is measured when (a) it names a definition of the program
or of a module it imports, and its reach is within the bound; (b) at
least one in-Form mutant measures it: a mutant at an expression of a
definition of its **core** that its run on the program evaluated, where
the core of a check is its reach, of an example the steps its events can
select and what they name, and of a law every step and what it names (for
a law, every mutant of the core measures it); and (c) running it on every
mutant of its reach stays within one FUEL (50,000,000): the mutants of
its reach times the fuel it took on the program plus 10,000, or, for a
law, the mutants times the steps of its exploration at most 1,000,000.

(a) excludes `check 1 + 1 == 2` and `check " 12 ".toint() == 12`: an
obligation that reads no definition is a statement about the language and
its implementation (the checker's own tests and the conformance suite are
written so), and the mutant set of the program has nothing to say about it.
It holds on every mutant for the same reason as the obligations of (b), so
refusing it and not them would treat two empty measurements differently.
(b) excludes the obligations the edits cannot see: an example whose
selected step is `-> p`, or a check whose run reaches only imported code
and expressions without an edit, holds on every mutant because no edit
touches what it runs, not because it states nothing. Constants and state
declarations are left out of the core for the same reason: an example
that overwrites the initial value of a field is not measured by an edit
of that value. Nor is an example measured by an edit inside an `emit`:
an example states the outcome of its last step and the state after it,
never an output (6.9), so the edit can change only what the example does
not state (the listing `book/listings/ch16/hello_http.rce`, whose step
`last = st` has no edit and whose `emit reply(id, st, ...)` has one). (c) bounds the cost of the rule (section 6). An
unmeasured obligation is reported by `rce mutants` and is not refused.

**W-MUT-2 (surviving mutant).** A program with a survivor: recommended.

## 5. The methods (7.1)

The two rules are decided by the principle of section 1.

- **W-MUT-1 is "requires change".** A measured obligation that holds on
  every mutant carries zero bits about the program within the edits: it
  states nothing that the program, edited once, could fail to meet, so as
  evidence for admission it is empty while it reads as evidence. It cannot
  be altered, because the change has more than one reading: the obligation
  may be strengthened (another input, another expected value, a stated
  field the step writes) or removed, and which of these states the intent
  is the author's statement. It cannot be recommended, because admission
  is the place where an obligation's information counts, and an admitted
  program would keep an obligation that the gate counts and that rules out
  nothing.
- **W-MUT-2 is "recommended".** A survivor is a second behaviour that the
  obligations do not tell from the program's. Writing the obligation that
  does is a choice of intent the machine cannot make: the witness shows
  what the program does after the events, and stating that output as an
  example asserts that the program, not the mutant, is the one meant,
  which only the author can say (the mutant may be the intended program,
  and the survivor a bug). The mutant set is also a sample of the changes
  a program can undergo, and whether every survivor is a real change is
  `rce equiv`'s question, which is undecided beyond its bound. So a
  survivor never refuses a program; `rce mutants` prints it with the
  diff and the witness. `rce check` does not print it, since it judges
  only as many mutants as W-MUT-1 needs (section 6).

## 6. Cost

`rce check` decides W-MUT-1 lazily. Admission records, for each check and
example, the fuel it took and the address of every expression its run
evaluated (at most 65,536 per obligation). The gate then takes the
obligations in text order; for each measured one not yet killed it judges
the mutants that measure it first, then the other mutants its run
evaluated, then the rest of its reach, and stops at the first kill. On a
kill it runs that mutant on every other obligation not yet killed whose
run evaluated the edited expression. An edit at an expression a run did not
evaluate cannot change that run, so the order finds a kill early; the
verdict does not depend on the order, because an obligation is reported
only after it has been run on every in-Form mutant of its reach. A mutant
is made by swapping one expression of a working copy of the program and
swapping it back, and its type check skips every function body but the
edited one (the other bodies are unchanged and get the reports they had).

## 7. Corpus results

Measured on this branch on one machine (two processes at a time), with
`rce check` before and after the gate.

| Corpus | Programs | `rce check` before → after | Vacuous obligations (W-MUT-1) | Programs changed |
|---|---|---|---|---|
| `spec/examples`, `programs`, `apps` | 106 | 3.9 s → 5.8 s | 2 | 2 |
| `conformance/tests` (accept, run, trace, host) | 1,613 | 20.9 s → 24.3 s | 49 copies of one of the 2 | 49 (harvested copies) |
| `stdlib`, modules and checks files but `math` | 539 | 1,010 s → 1,346 s | 106 | 9 |
| `stdlib`, `math` checks shards | 5,659 | 2,303 s → 2,653 s on every tenth shard | 0 | 0 |
| checker test fixtures | | | 3 | 3 |

The stdlib row was measured before the vacuous checks were removed; most
of its cost is the full scans that end in a W-MUT-1 report
(`tests/os/path_checks.rce`, 3.4 s → 43.8 s with 38 of them). After the
change the nine programs take 5.0 s → 25.2 s.

`rce mutants` over the 97 programs of `spec/examples` and `programs` that
finish within 180 s (4 do not; the 4 `apps` take longer than 900 s), 91 of
which have a mutant: mean score 0.497, median 0.75, pooled 1,641 killed of
2,570 non-equivalent; 66 equivalent, 215 not in Form, 929 survivors.

## 8. Relation to intent-holes.md

W-HOLE-1 of [intent-holes.md](intent-holes.md) refuses an intent that the
identity fill meets; W-HOLE-2 scores an intent by the share of
*spectests* (mutated outputs) its obligations reject. W-MUT-1 and the
scores here mutate the program instead of the output, which needs no
reference output and applies to every admitted program, with or without
holes. The two measures meet: an obligation that rejects no spectest of
its own output rejects no mutant whose only change is that output. The
unobservable intent of the ledger row of 2026-10-23 (an example that reads
a field no step emits) is a case W-MUT-1 measures when the steps its
events select have an edit: such an example is killed only through the
state it reads, never through an output.

## 9. What it does not claim

- That a high score means the program is correct. A score is relative to
  the single edits; a program and an obligation set can be wrong together.
- That a survivor is a bug. It is a behaviour the obligations leave open.
- That every conjunct of a measured obligation is informative: W-MUT-1
  measures an obligation as written, so a vacuous row joined by `and` to
  one that kills is not found.
- That an unmeasured obligation is informative. It is outside what the
  edits can see, or above the cost bound.

## 10. What would show it wrong

If, on the ladder task sets of `scripts/ladder-tasks.py`, the programs
with high scores admit as many wrong-but-passing repairs as those with low
scores, the score does not measure what section 1 says, and the edits of
section 2 are replaced by whatever does.
