# Modules

Each source file is a *module*. A larger program is several files checked
together.

```rce
{{#include ../listings/ch12/shop/pricing.rce}}
```

```rce
{{#include ../listings/ch12/shop/orders.rce}}
```

- `module shop.pricing` gives the module its name. Without it, the name
  comes from the file path.
- `use shop.pricing` makes that module visible.
- `pub` marks an item other modules may use. Everything else stays private
  to its module.

Give all the files of a program to `rce` in one command:

```text
$ rce check shop/pricing.rce shop/orders.rce
{{#include ../listings/ch12/check_both.out}}
```

Checked alone, `orders.rce` is refused, because the module it uses is not
part of the program:

```text
$ rce check shop/orders.rce
{{#include ../listings/ch12/orders_alone.out}}
```

## Calling into another module

A used module is visible by its last name, so `shop.checkout` calls
`pricing.discount(...)`:

```rce
{{#include ../listings/ch12/shop/checkout.rce}}
```

```text
$ rce test shop/pricing.rce shop/checkout.rce
{{#include ../listings/ch12/check_calls.out}}
```

Only `pub` items can be reached this way. `round_down` in `shop.pricing`
has no `pub`, so a call to it from another module is refused:

```rce
{{#include ../listings/ch12/change/private_call.rce}}
```

```text
$ rce check ../shop/pricing.rce private_call.rce
{{#include ../listings/ch12/change/private_call.out}}
```

The modules `std.text`, `std.collections` and `std.math` are built in and
can always be used.

## What you learned

- `module`, `use` and `pub`.
- All files of a program are checked together.
- A used module is reached by its last name, and only its `pub` items are
  visible.
