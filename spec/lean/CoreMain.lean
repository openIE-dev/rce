import RCE.Core.Check
import RCE.Core.Events
import Lean.Data.Json
/-! S6 executable: reads one core program per line as JSON (produced by the
    checker's `rce-conform-lean` elaborator) and prints the run's outcome,
    outputs and ticks used. -/
open Lean RCE.Core

/-- An integer, as a JSON number or (beyond 64 bits) a decimal string. -/
def jInt (j : Json) : Except String Int :=
  match j with
  | .str s => match s.toInt? with | some n => pure n | none => throw s!"bad int {s}"
  | _ => j.getInt?

partial def jTy (j : Json) : Except String Ty := do
  match j with
  | .str "bool" => pure .bool
  | .str "text" => pure .text
  | .str "unit" => pure .unit
  | .str "f64" => pure .f64
  | .str "f32" => pure .f32
  | .arr #[.str "int", lo, hi] => pure (.int (← jInt lo) (← jInt hi))
  | .arr #[.str "prod", a, b] => pure (.prod (← jTy a) (← jTy b))
  | .arr #[.str "sum", a, b] => pure (.sum (← jTy a) (← jTy b))
  | _ => throw s!"bad type {j.compress}"

def jOp : String → Except String BinOp
  | "add" => pure .add | "sub" => pure .sub | "mul" => pure .mul | "div" => pure .div
  | "rem" => pure .rem | "lt" => pure .lt | "eq" => pure .eq | "and" => pure .and
  | "or" => pure .or | "cat" => pure .cat | "band" => pure .band | "bor" => pure .bor
  | "bxor" => pure .bxor | "shl" => pure .shl | "shr" => pure .shr
  | "min" => pure .min | "max" => pure .max | "sadd" => pure .sadd | "ssub" => pure .ssub
  | "smul" => pure .smul | "wadd" => pure .wadd | "wsub" => pure .wsub | "wmul" => pure .wmul
  | s => throw s!"bad op {s}"

partial def jExpr (j : Json) : Except String Expr := do
  match j with
  | .arr #[.str "var", i] => pure (.var (← i.getNat?))
  | .arr #[.str "int", n] => pure (.lit (.int (← jInt n)))
  | .arr #[.str "raise", c] => pure (.raise (← c.getNat?))
  | .arr #[.str "try", a, c] => pure (.try_ (← jExpr a) (← c.getNat?))
  | .arr #[.str "inl", a] => pure (.inl (← jExpr a))
  | .arr #[.str "inr", a] => pure (.inr (← jExpr a))
  | .arr #[.str "bool", b] => pure (.lit (.bool (← b.getBool?)))
  | .arr #[.str "text", s] => pure (.lit (.text (← s.getStr?)))
  | .arr #[.str "unit"] => pure (.lit .unit)
  | .arr #[.str "f64", b] => pure (.lit (.f64 (← b.getNat?)))
  | .arr #[.str "f32", b] => pure (.lit (.f32 (← b.getNat?)))
  | .arr #[.str "un", .arr #[.str "trunc32", lo, hi], t, a] => do
    pure (.un (.trunc32 (← lo.getInt?) (← hi.getInt?)) (← jTy t) (← jExpr a))
  | .arr #[.str "tof", a] => pure (.tof (← jExpr a))
  | .arr #[.str "un", .arr #[.str "ineg", lo, hi], t, a] => do
    pure (.un (.ineg (← lo.getInt?) (← hi.getInt?)) (← jTy t) (← jExpr a))
  | .arr #[.str "un", .str o, t, a] => do
    let op ← match o with
      | "fneg" => pure UnOp.fneg | "tof32" => pure .tof32 | "widen" => pure .widen
      | "iabs" => pure UnOp.iabs | "fabs" => pure .fabs | "sqrt" => pure .sqrt
      | "floor" => pure .floor | "ceil" => pure .ceil | "round" => pure .round
      | s => throw s!"bad unary op {s}"
    pure (.un op (← jTy t) (← jExpr a))
  | .arr #[.str "bin", .str op, t, a, b] => pure (.bin (← jOp op) (← jTy t) (← jExpr a) (← jExpr b))
  | .arr #[.str "ite", c, a, b] => pure (.ite (← jExpr c) (← jExpr a) (← jExpr b))
  | .arr #[.str "let", a, b] => pure (.let_ (← jExpr a) (← jExpr b))
  | .arr #[.str "pair", a, b] => pure (.pair (← jExpr a) (← jExpr b))
  | .arr #[.str "fst", a] => pure (.fst (← jExpr a))
  | .arr #[.str "snd", a] => pure (.snd (← jExpr a))
  | .arr #[.str "call", f, a] => pure (.call (← f.getNat?) (← jExpr a))
  | .arr #[.str "cast", lo, hi, a] => pure (.cast (← jInt lo) (← jInt hi) (← jExpr a))
  | _ => throw s!"bad expr {j.compress}"

partial def jStmt (j : Json) : Except String Stmt := do
  match j with
  | .arr #[.str "let", e] => pure (.let_ (← jExpr e))
  | .arr #[.str "let", e, _] => pure (.let_ (← jExpr e))
  | .arr #[.str "emit", e] => pure (.emit (← jExpr e))
  | .arr #[.str "emit", e, _] => pure (.emit (← jExpr e))
  | .arr #[.str "set", i, .null, e] => pure (.set (← i.getNat?) none (← jExpr e))
  | .arr #[.str "set", i, .arr #[lo, hi], e] =>
    pure (.set (← i.getNat?) (some (← lo.getInt?, ← hi.getInt?)) (← jExpr e))
  | .arr #[.str "setop", i, .str op, t, e] => pure (.setop (← i.getNat?) (← jOp op) (← jTy t) (← jExpr e))
  | .arr #[.str "if", .arr arms, .arr els] => do
    let arms ← arms.toList.mapM fun a => do
      match a with
      | .arr #[c, .arr b] => pure (← jExpr c, ← b.toList.mapM jStmt)
      | _ => throw "bad arm"
    pure (.ifs arms (← els.toList.mapM jStmt))
  | .arr #[.str "while", c, n, .arr b] => pure (.while_ (← jExpr c) (← n.getNat?) (← b.toList.mapM jStmt))
  | .arr #[.str "break"] => pure .brk
  | .arr #[.str "continue"] => pure .cont
  | .arr #[.str "forpar", a, b, _, n, i, .str op, ta, idv, .arr body] => do
    let .lit v ← jExpr idv | throw "forpar: identity"
    pure (.forpar (← jExpr a) (← jExpr b) (← n.getNat?) (← i.getNat?) (← jOp op) (← jTy ta) v
      (← body.toList.mapM jStmt))
  | _ => throw s!"bad stmt {j.compress}"

def jFn (j : Json) : Except String (Ty × Ty × Expr) := do
  pure (← jTy (← j.getObjVal? "arg"), ← jTy (← j.getObjVal? "ret"), ← jExpr (← j.getObjVal? "body"))

/-- The reference's name for an integer type, by its range (types.rs `int_range`). -/
def intName (lo hi : Int) : String :=
  let names : List (String × Int × Int) :=
    [("i8", -128, 127), ("i16", -32768, 32767), ("i32", -2147483648, 2147483647),
     ("i64", -9223372036854775808, 9223372036854775807), ("u8", 0, 255), ("u16", 0, 65535),
     ("u32", 0, 4294967295), ("u64", 0, 18446744073709551615)]
  match names.find? (fun (_, a, b) => a == lo && b == hi) with
  | some (n, _, _) => n
  | none => s!"[{lo}, {hi}]"

def trapName : Trap → String
  | .overflow v lo hi => s!"overflow: {v} is outside {intName lo hi}"
  | .divZero => "division by zero"
  | .shift => "shift amount out of range"
  | .nonFinite => "non-finite float"
  | .user c => s!"user {c}"
  | .loopBound => "loop bound"

/-- `check` mode: is every expression of the program well typed? Functions
    by `checkProg` (sound for `WtProg`, `checkProg_sound`); each statement's
    expressions by `inf` (sound for `HasTy`, `inf_sound`) in the context the
    statements before it build, with the elaborator's types as hints. The
    statement layer itself has no typing judgement yet, so this walk is
    trusted code over proved expression checks. -/
partial def ckStmts (S : Sig) (Γ : List Ty) : List Json → Except String (List Ty)
  | [] => pure Γ
  | j :: js => do
    let need (e : Expr) (h : Option Ty) (t? : Option Ty) : Except String Ty :=
      match inf S Γ e h with
      | some t => if t? == none || t? == some t then pure t else throw s!"expected {repr t?}, got {repr t}"
      | none => throw s!"not shown well typed: {j.compress}"
    match j with
    | .arr #[.str "let", e, ty] => do
      let t ← jTy ty; let t' ← need (← jExpr e) (some t) (some t); ckStmts S (t' :: Γ) js
    | .arr #[.str "emit", e, ty] => do
      let t ← jTy ty; let _ ← need (← jExpr e) (some t) (some t); ckStmts S Γ js
    | .arr #[.str "set", i, _, e] => do
      let some t := Γ.get? (← i.getNat?) | throw "set: unbound"
      let _ ← need (← jExpr e) (some t) (some t); ckStmts S Γ js
    | .arr #[.str "setop", i, .str op, ty, e] => do
      let ta ← jTy ty
      unless Γ.get? (← i.getNat?) == some ta do throw "setop: variable type"
      unless opRes (← jOp op) ta == some ta do throw "setop: operator"
      let _ ← need (← jExpr e) (some ta) (some ta); ckStmts S Γ js
    | .arr #[.str "if", .arr arms, .arr els] => do
      for a in arms do
        match a with
        | .arr #[c, .arr b] => do let _ ← need (← jExpr c) (some .bool) (some .bool); let _ ← ckStmts S Γ b.toList
        | _ => throw "bad arm"
      let _ ← ckStmts S Γ els.toList; ckStmts S Γ js
    | .arr #[.str "while", c, _, .arr b] => do
      let _ ← need (← jExpr c) (some .bool) (some .bool); let _ ← ckStmts S Γ b.toList; ckStmts S Γ js
    | .arr #[.str "break"] | .arr #[.str "continue"] => ckStmts S Γ js
    | .arr #[.str "forpar", a, b, ity, _, i, .str op, ty, idv, .arr body] => do
      let it ← jTy ity; let ta ← jTy ty
      let _ ← need (← jExpr a) (some it) (some it); let _ ← need (← jExpr b) (some it) (some it)
      unless Γ.get? (← i.getNat?) == some ta do throw "forpar: target type"
      unless opRes (← jOp op) ta == some ta do throw "forpar: operator"
      let _ ← need (← jExpr idv) (some ta) (some ta)
      let _ ← ckStmts S (it :: Γ) body.toList; ckStmts S Γ js
    | _ => throw s!"check: statement {j.compress}"

def handleCheck (j : Json) : Except String Json := do
  let fns ← (← (← j.getObjVal? "fns").getArr?).toList.mapM jFn
  unless checkProg fns do throw "a function body not shown well typed"
  let _ ← ckStmts (sig fns) [] (← (← j.getObjVal? "main").getArr?).toList
  pure (Json.mkObj [("typed", true)])

def trapJ : Trap → Json
  | .user c => Json.mkObj [("user", Json.num (c : Nat))]
  | t => Json.mkObj [("trap", trapName t)]

/-- S4 event run (Events.lean `deltaS`): per event the new field values
    (shown, environment order) and emits, or the refusal; stops at the
    first refusal. -/
def handleEvents (j : Json) : Except String Json := do
  let fns ← (← (← j.getObjVal? "fns").getArr?).toList.mapM jFn
  let fuel ← (← j.getObjVal? "fuel").getNat?
  let mut regs : List (List Val × List StepD) := []
  for r in (← (← j.getObjVal? "regions").getArr?).toList do
    let mut fs : List Val := []
    for e in (← (← r.getObjVal? "init").getArr?).toList do
      match (runC fns fuel [] (← jExpr e)).1 with
      | .ok v => fs := fs ++ [v]
      | _ => throw "state initializer did not evaluate"
    let steps ← (← (← r.getObjVal? "steps").getArr?).toList.mapM fun s => do
      match s with
      | .arr #[tag, g, .arr b] =>
        let g' ← match g with | .null => pure none | g => some <$> jExpr g
        pure ((← tag.getNat?), g', ← b.toList.mapM jStmt)
      | _ => throw "bad step"
    regs := regs ++ [(fs, steps)]
  let showR (rs : List (List Val × List StepD)) : Json :=
    Json.arr (rs.map fun r => Json.arr (r.1.map (Json.str ∘ showVal)).toArray).toArray
  let init := showR regs
  let mut out : Array Json := #[]
  for ev in (← (← j.getObjVal? "events").getArr?).toList do
    let .arr #[tag, .arr pj] := ev | throw "bad event"
    let mut pv : List Val := []
    for e in pj.toList do
      match (runC fns fuel [] (← jExpr e)).1 with
      | .ok v => pv := v :: pv
      | _ => throw "event payload did not evaluate"
    match deltaS fns fuel (← tag.getNat?) pv regs with
    | .ok s o =>
      regs := (s.zip regs).map fun (fs, r) => (fs, r.2)
      out := out.push (Json.mkObj [("state", showR regs), ("emits", Json.arr (o.map Json.str).toArray)])
    | .refuse t => out := out.push (Json.mkObj [("refused", trapJ t)]); break
    | .oof => out := out.push (Json.mkObj [("refused", Json.mkObj [("trap", "comptime fuel exhausted")])]); break
    | .stuck => throw "stuck (soundness says unreachable)"
  pure (Json.mkObj [("init", init), ("steps", Json.arr out)])

def handle (line : String) : Json := Id.run do
  if let .ok j := Json.parse line then
    if j.getObjVal? "regions" matches .ok _ then
      return match handleEvents j with
        | .ok v => v
        | .error e => Json.mkObj [("error", e)]
    if j.getObjVal? "check" matches .ok _ then
      return match handleCheck j with
        | .ok v => v
        | .error e => Json.mkObj [("typed", false), ("why", e)]
  let r : Except String Json := do
    let j ← Json.parse line
    let fns ← (← (← j.getObjVal? "fns").getArr?).toList.mapM jFn
    let body ← (← (← j.getObjVal? "main").getArr?).toList.mapM jStmt
    let fuel ← (← j.getObjVal? "fuel").getNat?
    let (outs, oc, left) := runMain fns fuel body
    let outcome := match oc with
      | .done => "done" | .trap t => "trap:" ++ trapName t
      | .oof => "trap:comptime fuel exhausted" | .stuck => "stuck"
    pure (Json.mkObj [("outcome", outcome), ("outputs", Json.arr (outs.map Json.str).toArray),
                      ("ticks", Json.num (fuel - left : Nat))])
  match r with
  | .ok v => v
  | .error e => Json.mkObj [("error", e)]

partial def loop (i : IO.FS.Stream) (o : IO.FS.Stream) : IO Unit := do
  let line ← i.getLine
  if line.isEmpty then return
  if line.trim.isEmpty then loop i o else
  o.putStrLn (handle line.trim).compress
  o.flush
  loop i o

def main : IO Unit := do loop (← IO.getStdin) (← IO.getStdout)
