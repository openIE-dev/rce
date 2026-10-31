# Lists, maps and text

## Lists and maps

```rce
{{#include ../listings/ch08/collections.rce}}
```

- `List[i64]` is a list of `i64`. `[1, 2, 3, 4]` is a list literal.
- `&[i64]` is a borrowed *slice*: a view of a list or array that the
  function reads but does not own.
- `xs.map(fn(x: i64) => x * x)` applies a function to every element.
  `fn(x: i64) => ...` is a function written inline.
- `[x for x in xs if x % 2 == 0]` builds a new list from the old one.
- `fold` combines the elements into one value, starting from `0`.
- `Map[text, u32]` maps text keys to `u32` values. `m.get(w)` fails when
  the key is missing, and `catch 0` supplies a default.

## Text

```rce
{{#include ../listings/ch08/text.rce}}
```

Text is stored as UTF-8, so a length has to say what it counts.
`len_chars()` counts characters (Unicode scalar values) and `len_bytes()`
counts bytes: `"héllo"` has five characters and six bytes, because `é`
takes two bytes. Text has no plain `len()`; a program that calls it
requires change under `W-LEN-1`, so the unit is always visible:

```rce
{{#include ../listings/ch08/change/plain_len.rce}}
```

```text
$ rce check plain_len.rce
{{#include ../listings/ch08/change/plain_len.out}}
```

`s[1:3]` takes the bytes from position 1 up to 3, which here is exactly
the `é`. Slicing in the middle of a character traps. To go through text one
character at a time, use `chars()` or `for c in s`.

## Templates

A *template* builds text from literal parts and fields. `text.t"..."`
writes each field in braces, and turns it into text with `tostr()`; `{{`
and `}}` are literal braces:

```rce
{{#include ../listings/ch08/template.rce}}
```

A field is a name or a field path such as `o.item`, never an expression;
an empty field, a stray brace or an unclosed field requires change
(`W-TPL-2`). The checker expands a template at once into the concatenation
it means, so a template is not a value of its own. Chapter 19 shows
templates for text that goes into HTML, URLs and shell commands, where each
field is escaped for its context.

## What you learned

- `List`, `Map`, slices, inline functions and list comprehensions.
- Text is UTF-8, indexed in bytes, and measured with `len_bytes()` or
  `len_chars()`.
- `text.t"..."` templates fill fields into literal text.
