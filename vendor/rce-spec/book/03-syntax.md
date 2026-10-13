## 3. Syntactic grammar

This section is the complete grammar: which token sequences are programs.
Its EBNF is read by the machine check (3.12) and checked to be
*LALR(1)* (parsable left to right with one token of lookahead and no
conflicts), so every program has exactly one parse. Example:
`fn half(x: u64) -> u64 = x / 2` is a function declaration with an arrow body.

The grammar is LALR(1) with no conflicts (checked, 3.12). Where a reader might
see two parses, the grammar admits exactly one; the resolutions are listed in
3.11.

### 3.1 Modules

```
SourceFile   = { NEWLINE } [ ModuleHeader ] { Item } .
ModuleHeader = "module" Path NEWLINE .
Path         = NAME { "." NAME } .
Item         = UseDecl | IntentDecl | ExampleDecl | CheckDecl
             | [ "pub" ] ( ConstDecl | TypeDecl | StructDecl | KindDecl
                         | TraitDecl | FnDecl | ActorDecl )
             | StateDecl | EventDecl | ErrorDecl | StepDecl | ImplDecl
             | ExternDecl .
UseDecl      = "use" Path [ "as" NAME ] NEWLINE .
```

### 3.2 Intent

```
IntentDecl  = "intent" STRING NEWLINE [ INDENT IntentLine { IntentLine } DEDENT ] .
IntentLine  = ExampleDecl | CheckDecl | STRING NEWLINE .
ExampleDecl = "example" [ EventRef { "," EventRef } ] "->" Outcome NEWLINE .
EventRef    = NAME [ "(" [ Args ] ")" ] .
Outcome     = Expr | "refused" ":" NAME .
CheckDecl   = "check" Expr NEWLINE .
```

### 3.3 Data declarations

```
ConstDecl  = "const" NAME [ ":" Type ] "=" Expr NEWLINE .
TypeDecl   = "type" NAME [ Generics ] "=" Type NEWLINE .
StructDecl = [ "linear" ] "struct" NAME [ Generics ] NEWLINE [ INDENT Field { Field } DEDENT ] .
Field      = NAME ":" Type [ "=" Expr ] NEWLINE .
StateDecl  = "state" NAME NEWLINE INDENT Field { Field } DEDENT .
EventDecl  = "event" NAME [ Payload ] NEWLINE .
ErrorDecl  = "error" NAME [ Payload ] NEWLINE .
Payload    = "(" [ FieldItem { "," FieldItem } [ "," ] ] ")" .
FieldItem  = NAME ":" Type .
KindDecl   = "kind" NAME [ Generics ] NEWLINE INDENT Variant { Variant } DEDENT .
Variant    = NAME [ Payload ] NEWLINE .
```

### 3.4 Traits and implementations

```
TraitDecl   = "trait" NAME [ Generics ] [ ":" Bounds ] NEWLINE
              INDENT TraitMember { TraitMember } DEDENT .
TraitMember = "type" NAME [ ":" Bounds ] NEWLINE | FnSig NEWLINE .
ImplDecl    = "impl" [ "forall" Generics ] Type [ "for" Type ] NEWLINE
              INDENT ImplMember { ImplMember } DEDENT .
ImplMember  = TypeDecl | FnDecl .
Generics    = "[" GParam { "," GParam } "]" .
GParam      = [ "comptime" ] NAME [ ":" Bounds ] .
Bounds      = TypePath { "+" TypePath } .
```

`impl Show for Point` implements trait `Show` for `Point`; `impl Point`
(no `for`) adds inherent functions. `forall [T: Show]` binds the impl's type
parameters (3.11 R6 says why the keyword is needed).

### 3.5 Functions, steps and actors

```
FnSig     = [ "comptime" ] "fn" NAME [ Generics ] "(" [ Params ] ")" [ "->" Type ] .
FnDecl    = FnSig ( Block | "=" Expr NEWLINE ) .
ExternDecl = "extern" STRING FnSig NEWLINE .            (5.11)
Params    = Param { "," Param } [ "," ] .
Param     = [ "comptime" ] [ NAME ] NAME ":" Type | SelfParam .
SelfParam = "self" | "&" "self" | "&" "mut" "self" .
StepDecl  = "step" NAME "on" Pattern [ "if" Expr ] StepBody .
StepBody  = "->" Expr NEWLINE | Block .
ActorDecl = "actor" NAME NEWLINE INDENT ActorMember { ActorMember } DEDENT .
ActorMember = Field | Handler | FnDecl .
Handler   = "on" Pattern [ "if" Expr ] StepBody .
```

In `Param`, `from src: u8` declares argument label `from` and parameter name
`src` (Swift); `src: u8` alone uses `src` as both.

### 3.6 Types

Listed from loosest to tightest binding.

```
Type        = OptType [ "!" OptType ] .                  error union  T ! E
OptType     = BorrowType [ "?" ] .                       option       T?
BorrowType  = "&" [ "mut" ] BorrowType | RegionType .    borrow       &T, &mut T
RegionType  = BaseType [ "@" NAME ] .                    region       T @r
BaseType    = TypePath
            | "(" ")"                                    unit
            | "(" Type "," [ Type { "," Type } [ "," ] ] ")"   tuple
            | "(" Type ")"
            | "[" Type "]"                               slice
            | "[" Type ";" Expr "]"                      array, comptime length
            | "chan" BaseType                            channel
            | "iter" BaseType                            iterator (result of an iterator function only)
            | "fn" "(" [ TypeList ] ")" "->" BaseType .  function
TypePath    = Path [ "[" TypeList "]" ] .
TypeList    = Type { "," Type } [ "," ] .
```

### 3.7 Statements

```
Block      = NEWLINE INDENT Stmt { Stmt } DEDENT .
Stmt       = SimpleStmt | LetStmt | VarStmt | IfStmt | WhileStmt | LoopStmt
           | ForStmt | MatchStmt | ReturnStmt | BreakStmt | ContinueStmt
           | EmitStmt | FailStmt | SpawnStmt | SelectStmt | ArenaStmt
           | DeviceStmt | YieldStmt | ComptimeStmt | CheckDecl | PassStmt .
SimpleStmt = Expr [ AssignOp Expr | "<-" Expr ] NEWLINE .
AssignOp   = "=" | "+=" | "-=" | "*=" | "/=" | "%=" | "&=" | "|=" | "^=" | "<<=" | ">>=" .
LetStmt    = "let" Pattern [ ":" Type ] "=" Expr NEWLINE .
VarStmt    = "var" NAME [ ":" Type ] "=" Expr NEWLINE .
IfStmt     = "if" Expr Block { "elif" Expr Block } [ "else" Block ] .
WhileStmt  = "while" Expr [ Bound ] Block .
LoopStmt   = "loop" [ Bound ] Block .
Bound      = "bound" Expr .
ForStmt    = "for" Pattern "in" Expr Block .
MatchStmt  = "match" Expr NEWLINE INDENT Arm { Arm } DEDENT .
Arm        = Pattern [ "if" Expr ] ArmBody .
ArmBody    = "->" InlineStmt | Block .
InlineStmt = SimpleStmt | ReturnStmt | BreakStmt | ContinueStmt | PassStmt | EmitStmt | FailStmt | YieldStmt .
ReturnStmt = "return" [ Expr ] NEWLINE .
BreakStmt  = "break" NEWLINE .      ContinueStmt = "continue" NEWLINE .
PassStmt   = "pass" NEWLINE .       EmitStmt     = "emit" Expr NEWLINE .
FailStmt   = "fail" Expr NEWLINE .  SpawnStmt    = "spawn" Expr NEWLINE .
YieldStmt  = "yield" Expr NEWLINE .
SelectStmt = "select" NEWLINE INDENT SelectArm { SelectArm } DEDENT .
SelectArm  = "recv" Pattern "from" Expr ArmBody
           | "send" Expr "to" Expr ArmBody
           | "after" Expr ArmBody
           | "default" ArmBody .
ArenaStmt  = "with" "arena" NAME [ Bound ] Block .   (capacity, 5.7)
DeviceStmt = "with" "device" NAME [ Bound ] Block .  (accelerator region, 5.13)
ComptimeStmt = "comptime" Block .
```

The grammar accepts any expression left of `=`; rule W-PLACE (section 7)
then requires it to be a place (a variable, field, index or tuple member
path). This is how Python handles the same problem and keeps the grammar
LALR(1).

### 3.8 Expressions

Precedence, loosest first. All binary levels are left-associative except
comparison and range, which are non-associative (at most one operator).

| Level | Operators | Production |
|---|---|---|
| 1 | `try e`, `try e else e`, `if c then a else b`, `fn(p) => e` | Expr |
| 2 | `a catch b`, `a catch \|x\| b` | CatchExpr |
| 3 | `or` | OrExpr |
| 4 | `and` | AndExpr |
| 5 | `not` (prefix) | NotExpr |
| 6 | `== != < <= > >= in not in` (non-assoc) | CmpExpr |
| 7 | `..` `..=` (non-assoc) | RangeExpr |
| 8 | `\|` | BorExpr |
| 9 | `^` | BxorExpr |
| 10 | `&` | BandExpr |
| 11 | `<< >>` | ShiftExpr |
| 12 | `+ -` | AddExpr |
| 13 | `* / %` | MulExpr |
| 14 | `as T` | CastExpr |
| 15 | prefix `- ~ & &mut <- comptime` | UnaryExpr |
| 16 | call `f(…)`, index `a[i]`, slice `a[i:j]`, member `.x`, `.0` | Postfix |

Unlike C, bitwise operators bind tighter than comparison, so `x & 1 == 1`
means `(x & 1) == 1`.

```
Expr      = "try" CatchExpr [ "else" CatchExpr ]
          | "if" Expr "then" CatchExpr "else" CatchExpr
          | "fn" "(" [ Params ] ")" "=>" CatchExpr
          | CatchExpr .
CatchExpr = OrExpr [ "catch" [ "|" NAME "|" ] OrExpr ] .
OrExpr    = AndExpr { "or" AndExpr } .
AndExpr   = NotExpr { "and" NotExpr } .
NotExpr   = "not" NotExpr | CmpExpr .
CmpExpr   = RangeExpr [ CmpOp RangeExpr ] .
CmpOp     = "==" | "!=" | "<" | "<=" | ">" | ">=" | "in" | "not" "in" .
RangeExpr = BorExpr [ ( ".." | "..=" ) BorExpr ] .
BorExpr   = BxorExpr { "|" BxorExpr } .
BxorExpr  = BandExpr { "^" BandExpr } .
BandExpr  = ShiftExpr { "&" ShiftExpr } .
ShiftExpr = AddExpr { ( "<<" | ">>" ) AddExpr } .
AddExpr   = MulExpr { ( "+" | "-" ) MulExpr } .
MulExpr   = CastExpr { ( "*" | "/" | "%" ) CastExpr } .
CastExpr  = UnaryExpr [ "as" BaseType ] .
UnaryExpr = ( "-" | "~" | "&" [ "mut" ] | "<-" | "comptime" ) UnaryExpr | Postfix .
Postfix   = Primary
          | Postfix "(" [ Args ] ")"
          | Postfix "[" Expr { "," Expr } "]"
          | Postfix "[" [ Expr ] ":" [ Expr ] "]"
          | Postfix "." ( NAME | INT ) .
Args      = Arg { "," Arg } [ "," ] .
Arg       = [ NAME ":" ] Expr .
Primary   = NAME | "self" | INT | FLOAT | STRING | "true" | "false" | "none"
          | "(" ")" | "(" Expr ")" | "(" Expr "," [ Expr { "," Expr } [ "," ] ] ")"
          | "[" "]" | "[" Expr { "," Expr } [ "," ] "]" | "[" Expr ";" Expr "]"
          | "[" Expr "for" Pattern "in" OrExpr [ "if" OrExpr ] "]"
          | "{" "}" | "{" MapEntry { "," MapEntry } [ "," ] "}"
          | "chan" "[" Type "]" "(" [ Expr ] ")" .
MapEntry  = Expr ":" Expr .
```

### 3.9 Patterns

```
Pattern     = AtomPattern { "|" AtomPattern } .
AtomPattern = "_" | Path | Path "(" [ PatArg { "," PatArg } [ "," ] ] ")"
            | LitPattern [ ( ".." | "..=" ) LitPattern ]
            | "(" Pattern "," [ Pattern { "," Pattern } [ "," ] ] ")"
            | "(" Pattern ")" .
PatArg      = [ NAME ":" ] Pattern .
LitPattern  = INT | "-" INT | FLOAT | STRING | "true" | "false" | "none" .
```

### 3.10 Statement and expression boundaries

A `Block` always begins with `NEWLINE INDENT`, so a block can never be
confused with an expression. A one-line form exists for the common cases:
`fn f(x: T) -> U = e`, `step S on E -> e`, and `Pattern -> stmt` in `match`
and `select`.

### 3.11 How apparent ambiguities are resolved

Each is resolved in the grammar itself, so there is no precedence
declaration and no conflict to resolve at parse time.

- **R1 dangling else.** `else` only follows a `Block` (statement `if`) or is
  mandatory (`if … then … else …` expression). A statement `if` ends at its
  `DEDENT`, so the next `else` at the same indentation belongs to it.
- **R2 `try … else` against `if … then … else`.** The branches of an `if`
  expression and the operands of `try` are `CatchExpr`, which cannot contain a
  bare `try` or `if`. `if c then try a else b` is a syntax error; write
  `if c then (try a else b) else d`.
- **R3 comparison chains.** `0 < a < 10` is a syntax error (non-associative).
  Write `a in 0..10`.
- **R4 assignment against equality.** `=` is a statement form, never an
  expression; `(x = 1)` is a syntax error.
- **R5 generic call against index.** `f[T](x)` parses as `index(f, T)` then a
  call; name resolution (4.3) turns an index of a generic function or type by
  type arguments into instantiation, as in Go.
- **R6 impl generics against slice types.** `impl [T] …` would be ambiguous
  with the slice type `[T]`, so impl parameters need `forall`:
  `impl forall[T: Show] Show for List[T]`.
- **R7 labels.** In argument and pattern lists `NAME ":"` is always a label.
  Map literals `{k: v}` are in a different bracket, so `{a: 1}` is a map whose
  key is the variable `a` (write `{"a": 1}` for a text key).
- **R8 variant patterns against bindings.** A `Path` pattern is a unit variant
  if name resolution finds a variant of that name in scope, otherwise it binds
  a fresh variable (Rust's rule). W-PAT-1 refuses a binding that shadows a
  variant name, so a misspelled variant is never silently a binding.
- **R9 send.** `<-` after an expression at statement level is a send;
  in prefix position it is a receive. The two never meet: `a <- <- b` is
  "send the value received from b to a".

### 3.12 Machine check

`rce-spec-check` (in `spec/checker/`, Rust only):

1. reads every rule of this section from its code blocks (the annotation
   column beside a rule is prose and is skipped), expands the EBNF the way
   Lark did (`[ X ]` and `{ X }` inlined as with and without, `{ X }` as one
   left-recursive helper rule), builds the LALR(1) tables and fails on any
   shift/reduce or reduce/reduce conflict (verified: adding
   `| AddExpr "+" AddExpr` to `AddExpr` makes it fail with 3 conflicts);
   current result: 100 rules, 472 productions, 932 states, 0 conflicts;
2. parses the 51 programs in `examples/`, with and without their final
   newline; all must parse;
3. parses the 12 programs in `examples/reject/` (tab indent, chained
   comparison, `if` expression without `else`, `try` in a `then` branch,
   inconsistent dedent, `class`, assignment inside an expression, empty `kind`,
   state with no fields, `step` without `on`, example without `->`,
   unterminated string); all must be refused.

Each program is parsed three ways (the LALR(1) tables on the reference
lexer's tokens, the Rust parser, and the tree-sitter port) and all three must
agree. Until 2026-10-12 this check was `spec/check.py` on `spec/rce.lark`
with Lark in strict mode; both are retired.

`spec/tree-sitter/` is the same grammar for tree-sitter (`grammar.js`, with
layout, bracket depth, reserved-keyword NAME and longest-match numbers and
`<-` in `src/scanner.c`). The former `tree-sitter/compare.py N SEED` parsed every
example and N fuzz programs, each program's syntax mutants and SOUP
lexical-stress mutants (a random token jammed in or a space deleted) with
Lark and tree-sitter and required that both refuse, or both accept with identical
trees (rule names, nesting, kept tokens and their text). Results: 81 examples
plus 10,000 programs (seed 7, SOUP=2): 69,323 inputs, 15,205 accepted by both
with identical trees, 54,118 refused by both, 0 disagreements; 10,000
programs (seed 11, SOUP=10): 149,365 inputs, 35,837 identical trees, 113,528
refused by both, 0 disagreements.
Draft 0.4 grammar (with `extern` and arena `bound`): 91 examples plus 2,000
programs (seed 7, SOUP=2): 13,983 inputs, 3,060 identical trees, 10,923
refused by both, 0 disagreements. Draft 0.5 grammar and generator (`linear`, `iter`,
`yield`, `device`, no `lalr`): 127 examples plus 2,000 programs (seed 7,
SOUP=2): 16,092 inputs, 3,185 identical trees, 12,907 refused by both,
0 disagreements.

**Read receipt.** Which numbered resolution rule in 3.11 covers `send`? (for example R1)

<!-- body-sha256: 33c18e881e88f99afaba8b4797da5e26db03da9fae7aa1d4798491ff79cc5a81 -->
