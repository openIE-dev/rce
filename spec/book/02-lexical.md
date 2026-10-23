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

**One reserved list.** The language has one closed list of reserved words, the same in every module,
and nothing selects or extends it. It is: the keywords above; the 23 concurrency words of C.1 (`par`, `branch`,
`nondet`, `budget`, `deadline`, `priority`, `replicated`, `interrupt`, `session`, `lvar`, `group`, `lanes`,
`barrier`, `scope`, `atomic`, `protocol`, `offer`, `choose`, `node`, `supervisor`, `strategy`, `child`,
`restart`); the ten words of Appendix F (`requires`, `ensures`, `decreases`, `ghost`, `ct`, `declassify`,
`endorse`, `release`, `label` and `law`); and the type names `World`, `Template`, `Shell`, `Prompt`, `Header`, `mathint`
and the eight indexed names of 5.16, which no module may declare (W-NAME-2). A keyword or concurrency word is
refused as a name by the grammar (W-SYN-1). A word of Appendix F whose construct this implementation has
(`ct`, `endorse`, `release`, `label`, `requires`, `ensures`, `ghost`, `decreases`) is refused as a name
(W-NAME-2) unless it is used as its construct. A reserved word or type name whose construct this
implementation does not have (`declassify`, `law`, `Template`, `mathint`) is refused with the rule W-RES-1 and
the words *reserved, not implemented*; it is never read as an ordinary name and never silently ignored (F.12).
Directly after `.` a reserved word is a member name, so
`c.offer()` and `c.choose(L)` remain method calls.

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

### 2.7 Release label

A module may carry a release label on its first line, as the comment

```
# rce-version: 0.6
```

The line must start with `#` in column 0 and hold, after optional spaces, `rce-version:` and a label;
trailing spaces are allowed. The label is tolerated and ignored: it selects nothing, gates nothing, is not
hashed, does not appear in the Form, and no value of it is refused. A module without it means exactly what the
same module with it means. There is one language, the one this document defines, and every module of a
program is in it; modules may `use` each other whatever labels they carry. Because the label is a comment,
the grammar of section 3 and its LALR(1) check do not mention it.

A release label names exactly one spec commit (C.15): the text of this document at that commit. An
implementation of a release implements that text and no other, and a later commit that changes the meaning
of a valid program of the earlier text is a new release with a new label, never a silent change to an old
one. The cost of a standard library method call (C.6.4) is the one cost of that text, in every module and
wherever the call is executed from.

**Read receipt.** Which code point is the horizontal tab, as written in this section?

<!-- body-sha256: 4eeab6a05c17dd46d9db462dc4c801fcde4ded882648bafd843fde43777a4dd4 -->
