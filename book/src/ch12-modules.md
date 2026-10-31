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

Checked alone, `orders.rce` requires change, because the module it uses is not
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
has no `pub`, so a call to it from another module requires change:

```rce
{{#include ../listings/ch12/change/private_call.rce}}
```

```text
$ rce check ../shop/pricing.rce private_call.rce
{{#include ../listings/ch12/change/private_call.out}}
```

## The standard library

The modules `std.text`, `std.collections` and `std.math` are built in and
can always be used. So is every module of the standard library, which the
RCE project keeps in `stdlib/lib` of the repository and compiles into
`rce`: `use enc`, `use text.utf8` or `use crypto` resolves with no file
given and nothing to install. `rce lib` lists the modules, one line each,
and `rce lib MODULE` prints the source of one:

```rce
{{#include ../listings/ch12/base64.rce}}
```

`enc.b64_encode` is a `pub` function of the module `enc`, reached by its
last name like any other module. A library module's own `check`, `example`
and `intent` items are not part of a program that uses it; its items are.

The library has no version number. Each module is named by the SHA-256
hash of its text in canonical layout, and `stdlib/lib/MANIFEST.json` pins
every module by that hash. `rce lib` ends with the hash of the whole
library and names any module whose text differs from its pin. `rce manifest
DIR` writes such a manifest for a directory of your own modules, and
under `rce manifest-check MANIFEST` a manifest whose modules or imports do
not match their pins requires change (rule `W-PIN-1`).

## What you learned

- `module`, `use` and `pub`.
- All files of a program are checked together.
- A used module is reached by its last name, and only its `pub` items are
  visible.
- The standard library is built into `rce`, reached with `use`, and pinned
  by hash; `rce lib` lists it.
