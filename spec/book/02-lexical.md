## 2. Lexical grammar

This section says how source text is cut into *tokens* (the smallest
meaningful pieces: names, numbers, keywords, punctuation) and how
indentation becomes block structure. Getting this exact is what lets every
tool read a file the same way. Example: `let x = 0x1F  # note` is the tokens
`let`, `x`, `=`, `0x1F`; the comment is dropped.

### 2.1 Source text

A source file is UTF-8 encoded Unicode text. A byte-order mark is refused.
Line ends are LF or CR LF; a bare CR is refused. The horizontal tab U+0009 is
refused anywhere in a file (layout is by spaces only, 2.6). A file that does
not end in a line end is read as if it did.

### 2.2 Comments

`#` starts a comment that runs to the end of the line. A comment is part of the
line end that follows it, so a line holding only a comment is blank for
layout. There are no block comments. (`#` never occurs inside a token other
than a string.)

### 2.3 Tokens

Tokens are identifiers, keywords, operators and punctuation, and literals.
Spaces (U+0020) separate tokens and are otherwise ignored. Tokenization is
maximal munch: at each point the longest token that matches is taken. Two
consequences are stated so they are not surprises:

- `a<-b` lexes as `a`, `<-`, `b` (a send), never as `a < -b`. Write `a < -b`.
- `t.0.1` lexes `0.1` as a float. Write `(t.0).1`.

### 2.4 Identifiers and keywords

```
NAME = letter { letter | digit | "_" } .      (but not the single character "_")
letter = "A" … "Z" | "a" … "z" | "_" .
digit  = "0" … "9" .
```

Identifiers are ASCII. (Text values may hold any Unicode; names may not. This
keeps names comparable byte for byte, which the name-free hashing of 6.6
relies on.) `_` alone is the wildcard token.

Keywords are reserved and cannot be identifiers:

```
actor     after     and       arena     as        bound     break
chan      check     comptime  const     continue  default   device
elif      else      emit      error     event     example   extern
fail      false     fn        for       forall    from      if
impl      in        intent    iter      kind      let       linear
loop      match     module    mut       none      not       on
or        pass      pub       recv      refused   return    select
self      send      spawn     state     step      struct    then
to        trait     true      try       type      use       var
while     with      yield
```

`device`, `iter`, `linear` and `yield` are new in draft 0.5. `lalr` is not a
keyword: drafts up to 0.4 reserved it in `rce.lark` and the tree-sitter
scanner by mistake, without listing it here; draft 0.5 removes it from both.

`catch` is also a keyword. Keywords are reserved everywhere except as a
member name directly after `.` (`x.type`, `c.close()`), where the grammar's
`member_name` admits them; this is the only place a keyword can be a name
(change S3). Note: `class`, `unsafe`, `null`, `goto`, `async`
and `await` are not keywords; `class` and `unsafe` used as statements are
syntax errors, and `unsafe`, `Rc`, `Arc`, `RefCell` as names are refused by
rule W-SAFE-1 (section 7) so prototype code that used them is diagnosed
clearly.

### 2.5 Literals

```
INT    = decimal | "0x" hexdig { [ "_" ] hexdig } | "0o" octdig { [ "_" ] octdig }
       | "0b" bindig { [ "_" ] bindig } .
decimal = digit { [ "_" ] digit } .
FLOAT  = decimal "." decimal [ exponent ] | decimal exponent .
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
STRING = `"` { char | escape } `"` .
char   = any Unicode scalar value except `"`, `\` and line end .
escape = `\"` | `\\` | `\n` | `\r` | `\t` | `\0` | `\u{` hexdig{1,6} `}` .
```

There are no literal suffixes (`1u8` is not a token): the type of a literal
comes from context (5.3). A `\u{…}` escape that is not a Unicode scalar value
(a surrogate, or above U+10FFFF) is refused. `-` is never part of a numeric
literal; `-1` is negation applied to `1`, except in patterns (3.9).
`true`, `false` and `none` are keyword literals.

### 2.6 Layout

RCE uses indentation for blocks, as Python does, and the rule is the same one
Python's tokenizer uses:

1. Inside `(`…`)`, `[`…`]` and `{`…`}` line ends and indentation are ignored.
2. Otherwise each line end that ends a non-blank line emits `NEWLINE`.
3. The layout pass keeps a stack of indentation widths, initially `[0]`. On
   the first token of a logical line with indentation width w (number of
   leading spaces): if w > top, push w and emit `INDENT`; if w < top, pop
   while w < top and emit one `DEDENT` per pop, and w must then equal top
   (otherwise the text is refused: "inconsistent dedent"); if w = top, emit
   nothing.
4. At end of file, emit `NEWLINE` if needed and one `DEDENT` per stack entry
   above 0.

Blank and comment-only lines produce no tokens. There is no line
continuation character; open a bracket instead.

### 2.7 Language version

A module may declare its language version on its first line, as the
comment

```
# rce-version: 0.6
```

The line must start with `#` in column 0 and hold, after optional spaces,
`rce-version:` and a version; trailing spaces are allowed. The known
versions are `0.5` and `0.6`; any other is refused (W-VER-1). A module
without the declaration is version 0.5. Version 0.5 is the language of
sections 2 to 10. Version 0.6 adds Appendix C: its 23 words become
keywords (C.1) and its constructs and rules apply. A version 0.5 module
never sees Appendix C: its words are names there, and a valid 0.5 program
is accepted unchanged by every conforming implementation. Because the
declaration is a comment, the grammar of section 3 and its LALR(1) check
are unchanged for 0.5; a 0.6 module is parsed by section 3 with the rules
of C.2 applied, which is checked LALR(1) the same way. Each module of a program has its own version; a 0.6 module
may `use` a 0.5 module and the reverse. (The reference checker has a
non-normative `conc06` build feature that reads undeclared modules as 0.6,
for the draft's own test corpora; a declared version always wins.)

**Read receipt.** Which code point is the horizontal tab, as written in this section?

<!-- body-sha256: dd32cf81b6db8c0c1fde74f2660ddf8d6d5616eb6450422dca4c1eb272aa41f5 -->
