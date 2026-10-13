## 1. Notation and conformance

This section fixes how the rest of the spec is written and what it means
for an implementation to follow it. *EBNF* (extended Backus-Naur form) is a
notation for grammars; an *inference rule* states that a conclusion holds
when its premises hold. For example `Option = "[" Expression "]" .` says an
option is an expression in brackets.

**Grammar notation.** EBNF in the style of the Go specification:

```
Production  = name "=" Expression "." .
Expression  = Term { "|" Term } .
Term        = Factor { Factor } .
Factor      = name | token [ "…" token ] | Group | Option | Repetition .
Group       = "(" Expression ")" .
Option      = "[" Expression "]" .      (0 or 1)
Repetition  = "{" Expression "}" .      (0 or more)
```

Lower-case names are syntactic productions, `UPPER` names are tokens, and
quoted strings are literal tokens. `NEWLINE`, `INDENT` and `DEDENT` are
produced by the layout pass (2.6).

**Judgment notation.** Inference rules are written

```
  premise₁   …   premiseₙ
  ───────────────────────── [Name]
         conclusion
```

Γ is a typing context (name ↦ type, mode), Δ a context of live borrows, Σ the
declaration environment (types, kinds, traits, events, errors, regions). "iff"
means if and only if.

**Conformance.** An implementation (a) refuses every text that is not in the
language of section 3, or that breaks a rule of section 7, with a diagnostic
naming the rule; (b) for every accepted program, computes δ, λ and the fold of
section 6 exactly, including every edge in 6.3. Behavior the spec does not
define does not exist: there is no "undefined behavior" and no
"implementation-defined" result. The only implementation freedom is cost
(time, memory, energy), never value.

**Read receipt.** In the grammar notation, what does a Repetition `{ ... }` mean, in the spec's words?

<!-- body-sha256: c76056a1984c61b4ddce8745feb78c2cff67ef2295e40b3d2082f67e4cbecef4 -->
