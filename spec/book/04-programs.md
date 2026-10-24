## 4. Programs, names and scope

This section says what a whole program is: a set of modules, the names each
module can see, and the Form the program denotes. A *module* is one source
file; *scope* is the region of text where a name can be used. Example:
`use std.text` makes the module `std.text` visible as `text`.

### 4.1 Modules and packages

A source file is a module. Its path is given by `module a.b.c` or, if absent,
by its file path relative to the package root with `/` read as `.` and the
`.rce` extension removed. `use p.q` makes module `p.q` visible under the name
`q`; `use p.q as r` under `r`. Items are private to their module unless marked
`pub`. Import cycles are refused (W-MOD-2). The modules `std.text`,
`std.collections` and `std.math` are built in and always resolvable (S4), and
so is every module of the standard library (`stdlib/lib` of the repository,
pinned by hash in its manifest, and pinned by a program as the dependency
`stdlib`, F.5.3): `use enc`, `use data.events` and the
others resolve with no file given, no directive and no flag, in every tool
(`rce lib` lists them). A library module's own `check`, `example` and
`intent` items are not part of a program that imports it; its items are. A
`#!lib NAME` comment line places a library's items in front of the module,
unqualified, as the shipped binary does; it is a comment to every other
tool, and `use` is the form of this text. A
program is the set of modules given together, and `use` must name one of them
or a built-in module (W-NAME-1).

### 4.2 The program's Form

A **Form module** is a module (with what it imports) that declares at least one
`state`, `event` or `actor`. The Form of the program is built from all Form
modules reachable from the root (6.1). A module with only `fn`, types, `const`
and `impl` is a **library module**: it has no Form of its own and denotes a set
of pure functions.

A program whose root declares `fn main()` and no state is a **command**: it
denotes the Form with one region `Main` of unit type, one event `Start(args:
List[text])`, and δ given by running `main` (6.8).

### 4.3 Scope

Scopes nest: universe (built-in types and functions, 6.3–6.4) ⊃ package ⊃
module ⊃ item (generic parameters, parameters, state fields inside steps)
⊃ block. Declarations at module level are visible throughout the module
regardless of order (so `const N` may follow its use). Inside a block a name is
visible from its declaration to the end of the block. Shadowing a name from an
enclosing *block* is allowed with `let`; shadowing a parameter, a state field
or a module-level name in the same item is refused (W-NAME-2).

Namespaces: one namespace for values (variables, functions, consts, variants,
events, errors) and one for types (types, kinds, traits, states, actors).
Event, error and variant constructors live in both: as types (for patterns and
payload typing) and as constructor functions.

**Read receipt.** What is the outermost scope called? (one word)

<!-- body-sha256: 5ca7d8a15cdf8973b8489ee52c0c6f634b5b029aaf692148c113fd02b0064a8d -->
