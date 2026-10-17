import RCE.Core.Count
/-! S6: the run driver for `fn main` over the core. A main body is a list of
    statements, each `let` (bind the value as de Bruijn index 0) or `emit`.
    Ticks follow the reference (book 12 Fuel): one for entering `main`, one
    per statement, and the expression's own ticks from `evalC`. Exhaustion
    ends the run with the outputs so far (rule E1). -/
namespace RCE.Core

/-- Statements of a block (S2/S6, book 6 statements): `let`/`var` bind
    the value as de Bruijn index 0; `set` and `setop` assign index `i`;
    `ifs` is an `if`/`elif`/`else` chain; `while_` runs with a bound;
    `arms` and `iter` are the internal continuations of `ifs` and `while_`
    and cost no tick of their own. -/
inductive Stmt where
  | let_ (e : Expr)
  | emit (e : Expr)
  /-- `x = e`, checked into `[lo, hi]` when the variable is an integer -/
  | set (i : Nat) (rng : Option (Int × Int)) (e : Expr)
  /-- `x op= e` at operand type `ta` -/
  | setop (i : Nat) (op : BinOp) (ta : Ty) (e : Expr)
  | ifs (arms : List (Expr × List Stmt)) (els : List Stmt)
  | while_ (c : Expr) (bound : Nat) (body : List Stmt)
  | brk
  | cont
  | arms (arms : List (Expr × List Stmt)) (els : List Stmt)
  | iter (n : Nat) (c : Expr) (bound : Nat) (body : List Stmt)
  /-- `for x in a..b par` (C.8, draft 0.6) with one reduction target, the
      outer variable `i` updated by `op=` at type `ta`, identity `idv`:
      every item runs the body from the identity, and the contributions are
      combined over the fixed tree, then into the value before the loop -/
  | forpar (a b : Expr) (bound : Nat) (i : Nat) (op : BinOp) (ta : Ty) (idv : Val) (body : List Stmt)
  /-- internal continuation of `forpar`: item `j` of `n` from `x0`,
      contributions so far in `acc` (newest first); no tick of its own -/
  | fpstep (j n : Nat) (x0 : Int) (i : Nat) (op : BinOp) (ta : Ty) (idv before : Val)
      (acc : List Val) (body : List Stmt)

/-- How a statement list ended. -/
inductive Fl where
  | norm | brk | cont | trap (t : Trap) | oof | stuck

def escText (s : String) : String :=
  s.foldl (fun acc c => acc ++ match c with
    | '"' => "\\\"" | '\\' => "\\\\" | '\n' => "\\n" | '\t' => "\\t" | c => c.toString) ""

def showVal : Val → String
  | .int n => toString n
  | .bool b => if b then "true" else "false"
  | .text s => "\"" ++ escText s ++ "\""
  | .unit => "()"
  | .pair a b => "(" ++ showVal a ++ ", " ++ showVal b ++ ")"
  | .inl v => "inl(" ++ showVal v ++ ")"
  | .inr v => "inr(" ++ showVal v ++ ")"
  | .f64 b => RCE.F64.text b
  | .f32 b => RCE.F64.text32 b

/-- How a run ended. -/
inductive Outcome where
  | done | trap (t : Trap) | oof | stuck

/-- Leave a block: drop the bindings it introduced (the environment grows
    at the front). -/
def leave (n : Nat) (ρ : List Val) : List Val := ρ.drop (ρ.length - n)

/-- Read the assigned place once more (interp.rs `Stmt::Assign` evaluates
    the place after the right-hand side): one tick when any is left; with
    none left the read fails silently and the value is stored unchecked. -/
def placeRead (f : Nat) : Bool × Nat :=
  match f with
  | 0 => (false, 0)
  | f + 1 => (true, f)

/-- The fixed tree T of C.8 (host.rs `tree_split`): the left part is the
    largest power of two below the length. `d` is structural depth. -/
def treeR (op : BinOp) (ta : Ty) (idv : Val) : Nat → List Val → Res
  | _, [] => .ok idv
  | _, [v] => .ok v
  | 0, _ => .stuck
  | d + 1, vs =>
    let h := 2 ^ Nat.log2 (vs.length - 1)
    match treeR op ta idv d (vs.take h) with
    | .ok l => match treeR op ta idv d (vs.drop h) with
      | .ok r => binop op ta l r
      | e => e
    | e => e

/-- Run a statement list on depth `d` and budget `f` in environment `ρ`:
    outputs, how it ended, the environment, the ticks left. `d` only makes
    the definition structural; `runMain` gives `2·fuel + 2`, more than the
    statements fuel can pay for. -/
def execS (P : Prog) : Nat → Nat → List Val → List Stmt → List String × Fl × List Val × Nat
  | 0, f, ρ, _ => ([], .oof, ρ, f)
  | _ + 1, f, ρ, [] => ([], .norm, ρ, f)
  | d + 1, f, ρ, s :: ss =>
    let k (r : List String × Fl × List Val × Nat) : List String × Fl × List Val × Nat :=
      match r.2.1 with
      | .norm => let r2 := execS P d r.2.2.2 r.2.2.1 ss; (r.1 ++ r2.1, r2.2)
      | _ => r
    match s with
    | .arms [] els =>
      let r := execS P d f ρ els
      k (r.1, r.2.1, leave ρ.length r.2.2.1, r.2.2.2)
    | .arms ((c, b) :: rest) els =>
      match runC P f ρ c with
      | (.ok (.bool true), f1) =>
        let r := execS P d f1 ρ b
        k (r.1, r.2.1, leave ρ.length r.2.2.1, r.2.2.2)
      | (.ok (.bool false), f1) => execS P d f1 ρ (.arms rest els :: ss)
      | (.ok _, f1) => ([], .stuck, ρ, f1)
      | (.err t, f1) => ([], .trap t, ρ, f1)
      | (.oof, f1) => ([], .oof, ρ, f1)
      | (.stuck, f1) => ([], .stuck, ρ, f1)
    | .iter n c bnd body =>
      match runC P f ρ c with
      | (.ok (.bool true), f1) =>
        if n + 1 > bnd then ([], .trap .loopBound, ρ, f1) else
        let r := execS P d f1 ρ body
        let ρ' := leave ρ.length r.2.2.1
        match r.2.1 with
        | .norm | .cont =>
          let r2 := execS P d r.2.2.2 ρ' (.iter (n + 1) c bnd body :: ss); (r.1 ++ r2.1, r2.2)
        | .brk => let r2 := execS P d r.2.2.2 ρ' ss; (r.1 ++ r2.1, r2.2)
        | fl => (r.1, fl, ρ', r.2.2.2)
      | (.ok (.bool false), f1) => execS P d f1 ρ ss
      | (.ok _, f1) => ([], .stuck, ρ, f1)
      | (.err t, f1) => ([], .trap t, ρ, f1)
      | (.oof, f1) => ([], .oof, ρ, f1)
      | (.stuck, f1) => ([], .stuck, ρ, f1)
    | .fpstep j n x0 i op ta idv before acc body =>
      if j ≥ n then
        match treeR op ta idv acc.length acc.reverse with
        | .ok t => match binop op ta before t with
          | .ok v => k ([], .norm, ρ.set i v, f)
          | .err e => ([], .trap e, ρ.set i before, f)
          | _ => ([], .stuck, ρ, f)
        | .err e => ([], .trap e, ρ.set i before, f)
        | _ => ([], .stuck, ρ, f)
      else
        let r := execS P d f (.int (x0 + j) :: ρ.set i idv) body
        let ρ' := leave (ρ.length + 1) r.2.2.1
        match r.2.1 with
        | .norm | .cont =>
          match ρ'.get? (i + 1) with
          | some c =>
            let r2 := execS P d r.2.2.2 (ρ'.drop 1)
              (.fpstep (j + 1) n x0 i op ta idv before (c :: acc) body :: ss)
            (r.1 ++ r2.1, r2.2)
          | none => (r.1, .stuck, ρ, r.2.2.2)
        | fl => (r.1, fl, (ρ'.drop 1).set i before, r.2.2.2)
    | _ =>
      match f with
      | 0 => ([], .oof, ρ, 0)
      | f + 1 =>
        match s with
        | .ifs arms els => execS P d f ρ (.arms arms els :: ss)
        | .while_ c bnd body => execS P d f ρ (.iter 0 c bnd body :: ss)
        | .brk => ([], .brk, ρ, f)
        | .cont => ([], .cont, ρ, f)
        | .forpar a b bnd i op ta idv body =>
          match runC P f ρ a with
          | (.ok (.int x), f1) =>
            match runC P f1 ρ b with
            | (.ok (.int y), f2) =>
              let n := (y - x).toNat
              if n > bnd then ([], .trap .loopBound, ρ, f2) else
              match ρ.get? i with
              | some before => execS P d f2 ρ (.fpstep 0 n x i op ta idv before [] body :: ss)
              | none => ([], .stuck, ρ, f2)
            | (.ok _, f2) => ([], .stuck, ρ, f2)
            | (.err t, f2) => ([], .trap t, ρ, f2)
            | (.oof, f2) => ([], .oof, ρ, f2)
            | (.stuck, f2) => ([], .stuck, ρ, f2)
          | (.ok _, f1) => ([], .stuck, ρ, f1)
          | (.err t, f1) => ([], .trap t, ρ, f1)
          | (.oof, f1) => ([], .oof, ρ, f1)
          | (.stuck, f1) => ([], .stuck, ρ, f1)
        | .let_ e =>
          match runC P f ρ e with
          | (.ok v, f1) => k ([], .norm, v :: ρ, f1)
          | (.err t, f1) => ([], .trap t, ρ, f1)
          | (.oof, f1) => ([], .oof, ρ, f1)
          | (.stuck, f1) => ([], .stuck, ρ, f1)
        | .emit e =>
          match runC P f ρ e with
          | (.ok v, f1) => k (["emit " ++ showVal v], .norm, ρ, f1)
          | (.err t, f1) => ([], .trap t, ρ, f1)
          | (.oof, f1) => ([], .oof, ρ, f1)
          | (.stuck, f1) => ([], .stuck, ρ, f1)
        | .set i rng e =>
          match runC P f ρ e with
          | (.ok v, f1) =>
            let (rd, f2) := placeRead f1
            match rd, rng, v with
            | true, some (lo, hi), .int x =>
              match chk lo hi x with
              | .err t => ([], .trap t, ρ, f2)
              | _ => k ([], .norm, ρ.set i v, f2)
            | _, _, _ => k ([], .norm, ρ.set i v, f2)
          | (.err t, f1) => ([], .trap t, ρ, f1)
          | (.oof, f1) => ([], .oof, ρ, f1)
          | (.stuck, f1) => ([], .stuck, ρ, f1)
        | .setop i op ta e =>
          match runC P f ρ e with
          | (.ok y, f1) =>
            match f1 with
            | 0 => ([], .oof, ρ, 0)
            | f1 + 1 =>
              match ρ.get? i with
              | none => ([], .stuck, ρ, f1)
              | some x =>
                match binop op ta x y with
                | .ok v => k ([], .norm, ρ.set i v, (placeRead f1).2)
                | .err t => ([], .trap t, ρ, f1)
                | _ => ([], .stuck, ρ, f1)
          | (.err t, f1) => ([], .trap t, ρ, f1)
          | (.oof, f1) => ([], .oof, ρ, f1)
          | (.stuck, f1) => ([], .stuck, ρ, f1)
        | _ => ([], .stuck, ρ, f)

/-- `fn main()`: one tick to enter, then the body. -/
def runMain (P : Prog) (fuel : Nat) (ss : List Stmt) : List String × Outcome × Nat :=
  match fuel with
  | 0 => ([], .oof, 0)
  | f + 1 =>
    let r := execS P (2 * fuel + 2) f [] ss
    let oc : Outcome := match r.2.1 with
      | .norm | .brk | .cont => .done
      | .trap t => .trap t
      | .oof => .oof
      | .stuck => .stuck
    (r.1, oc, r.2.2.2)

/-- The S6 probe, checked against the reference (`rce-conform`) at fuel 9,
    10, 13, 14: `fn f(x: i64) -> i64 = x * 2`, main `let a = f(3)`,
    `emit "hi"`, `emit a + 1`. -/
def probeP : Prog := [(.int (-9223372036854775808) 9223372036854775807,
  .int (-9223372036854775808) 9223372036854775807,
  .bin .mul (.int (-9223372036854775808) 9223372036854775807) (.var 0) (.lit (.int 2)))]
def probeM : List Stmt :=
  [.let_ (.call 0 (.lit (.int 3))), .emit (.lit (.text "hi")),
   .emit (.bin .add (.int (-9223372036854775808) 9223372036854775807) (.var 0) (.lit (.int 1)))]

-- outputs emitted at each budget (the reference's counts; text is checked by the adapter)
example : (runMain probeP 9 probeM).1.length = 0 := by decide
example : (runMain probeP 10 probeM).1.length = 1 := by decide
example : (runMain probeP 13 probeM).1.length = 1 := by decide
example : (runMain probeP 14 probeM).1.length = 2 := by decide

end RCE.Core
