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

Text is stored as UTF-8. `len()` counts bytes, not characters: `"héllo"`
has five characters and six bytes, because `é` takes two bytes.
`s[1:3]` takes the bytes from position 1 up to 3, which here is exactly
the `é`. Slicing in the middle of a character traps. To go through text one
character at a time, use `chars()` or `for c in s`.

## What you learned

- `List`, `Map`, slices, inline functions and list comprehensions.
- Text is UTF-8, measured and indexed in bytes.
