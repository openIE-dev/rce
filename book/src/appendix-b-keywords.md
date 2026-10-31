# Appendix B: keywords

RCE has one list of reserved words, the same in every module
(specification, section 2.4). None of them can be used as a name, except as
a member name directly after `.`, as in `c.close()`.

The keywords:

```text
actor     after     and       arena     as        bound     break
catch     chan      check     comptime  const     continue  default
device    elif      else      emit      error     event     example
extern    fail      false     fn        for       forall    from
if        impl      in        intent    iter      kind      let
linear    loop      match     module    mut       none      not
on        or        pass      pub       recv      return    select
self      send      spawn     state     step      struct    then
to        trait     true      try       type      use       var
while     with      yield
```

The concurrency words of Appendix C:

```text
par       branch    nondet    budget    deadline  priority
replicated interrupt session  lvar      group     lanes
barrier   scope     atomic    protocol  offer     choose
node      supervisor strategy child     restart
```

The words of Appendix F:

```text
requires  ensures   decreases ghost     ct
declassify endorse  release   label     law
```

and the type names `World`, `Template`, `Shell`, `Prompt`, `Header` and
`mathint`, which no module may declare.

`declassify`, `law`, `Template` and `mathint` name constructs the checker
does not implement: a program that uses one requires change under
`W-RES-1`, *reserved, not implemented*.

Some keywords belong to parts of the language this book does not cover:
`arena` (memory regions), `comptime` (evaluation while checking), `linear`
(values that must be used exactly once) and `device` (accelerator regions),
and among the concurrency words those listed at the end of chapter 14. The
specification describes each one.
