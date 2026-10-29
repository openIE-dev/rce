# Traits and generics

```rce
{{#include ../listings/ch11/traits.rce}}
```

## Traits

A `trait` lists functions that a type promises to have. `impl Describe for
Point` keeps that promise for `Point`.

`impl forall[T: Describe] Describe for List[T]` says: for every type `T`
that implements `Describe`, a list of `T` implements it too. The word
`forall` is needed so the grammar can tell the parameter list apart from a
slice type `[T]`.

## Generic functions

`largest[T: Ord](xs: &[T]) -> T?` works for any type `T` that can be
ordered. `Ord` is a bound: a call with a type that cannot be compared
requires change. Inside, `copy(x)` is needed because `best` keeps its
own value while `xs` is only borrowed.

The arm `some(b) if x > b -> ...` shows a guard on a `match` arm, the same
idea as a guard on a step.

## What you learned

- `trait` and `impl ... for ...`.
- Generic functions and impls with bounds such as `T: Ord`.
