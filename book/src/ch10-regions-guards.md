# Several regions and guards

A program can declare more than one `state`. The specification calls each
one a *region*. One event can update several regions at once.

```rce
{{#include ../listings/ch10/cart.rce}}
```

`AddItem` has steps in both `Cart` and `Stats`. When it arrives, the step
of each region runs. Each step sees the state as it was *before* the
event, so the order in which regions are listed does not change the result.

## All or nothing

The cart holds at most three items. On a fourth, the guarded step keeps the
state:

```text
$ rce run cart.rce "AddItem(name: \"a\"), AddItem(name: \"b\"), AddItem(name: \"c\"), AddItem(name: \"d\")"
{{#include ../listings/ch10/cart_run.out}}
```

Look at the last line. `Stats` would have counted the fourth add, but
because the step of `Cart` kept the state, neither region changed. An event
applies to every region or to none. This is what makes a kept step safe to
ignore: nothing half-happened.

## Examples over several fields

With more than one field in the program, an example's expected outcome is a
condition on the final state, such as
`items == ["jam"] and adds == 2` in the intent above.

## What you learned

- Several `state` regions can react to the same event.
- Every step reads the state from before the event.
- An event is applied to all regions, or the state of all of them is kept.
