# Ownership and borrowing

RCE has no manual `free`, and a value cannot be used after it is released.
Two ideas, also found in Rust, make this work: every value has one owner,
and other code can borrow a value for a short time. A value is released when
the block that owns it ends. RCE's borrows are more restricted than Rust's,
which makes them simpler to check.

## Moving

Small fixed-size values such as integers, `bool` and `f64` are *copied*
when you pass them. Values that own memory, such as `text`, `List` and
`Map`, are *moved*: passing one to a function hands it over, and the old
name can no longer be used.

```rce
{{#include ../listings/ch09/refused/use_after_move.rce}}
```

```text
$ rce check use_after_move.rce
{{#include ../listings/ch09/refused/use_after_move.out}}
```

The first call to `take` moved `s`, so the second call is refused. If you
really need two independent values, `copy(s)` makes a full copy.

## Borrowing

```rce
{{#include ../listings/ch09/ownership.rce}}
```

- `&name` lends `name` for reading. The owner keeps it, so `length` can be
  called twice.
- `&mut a` lends `a` for changing. `push_all` appends to the caller's list.
- `copy(a)` makes a separate list.

## The one rule about borrows

While something is borrowed for changing, nothing else may look at it in
the same call:

```rce
{{#include ../listings/ch09/refused/mut_alias.rce}}
```

```text
$ rce check mut_alias.rce
{{#include ../listings/ch09/refused/mut_alias.out}}
```

## Why RCE borrows need no lifetimes

In RCE a borrow lives only as long as the call or statement that made it.
A borrow cannot be stored in a field, a list, an event or a state, and in
general cannot be returned. (A function may return a borrow of one of its
own parameters, such as a slice of a list it was given.) Because borrows
never outlive the statement that created them, the checker does not need
lifetime annotations, and the check is a local pass through each function.

The specification calls these *second-class* borrows, a design also used
by Swift's `inout` parameters and by the Hylo language.

## What you learned

- Copy types are copied; owning types are moved, and a moved name is dead.
- `&x` lends for reading, `&mut x` for writing, `copy(x)` duplicates.
- A mutable borrow excludes every other use for its duration.
