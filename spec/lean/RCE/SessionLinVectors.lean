/-
  Test vectors for `RCE.SessionLin` (linear endpoints). `main` prints one
  JSON line per vector: an RCE body over session endpoints (`src`) and
  what `chkBody` returns for it (`ok`, or the rule the body requires change under).
  `chkBody` is proved sound for the linear judgment `Ty` and `Ty` is
  proved safe (`RCE.SessionLin.chkBody_safe`).

  Rendering: `x <- 1` sends, `let v = <- x` receives, `x.close()` closes,
  `let m = x` moves `x` to the new name `m`, `if k > 0` / `else` is a
  branch (an empty arm is `emit 0`). Each parameter `cI` has its own
  protocol `PI` of message steps. Moves are generated at the top level of
  the body only (a `let` inside an arm is scoped to the arm in RCE).

  `checker/tests/lean_session_lin.rs` runs this file, compares it with
  `checker/tests/vectors/session-lin-lean.jsonl`, and checks the checker
  accepts exactly the `ok` bodies (no W-SES rule) and reports the rule
  Lean gives for each body that requires change.
-/
import RCE.SessionLin
open RCE.SessionRec RCE.SessionLin

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616
def pick (s n : Nat) : Nat := (s / 65536) % n

def genP : Nat → Nat → P
  | 0, _ => .done
  | d + 1, s => if pick s 2 == 0 then .send (genP d (lcg s)) else .recv (genP d (lcg s))

def setL : List St → Nat → St → List St := RCE.SessionLin.set

/-- a body that mostly follows the protocols (so many are accepted) and
    sometimes deviates (so every rule is reached) -/
def genB : Nat → Bool → List St → Nat → Prog × List St
  | 0, _, Γ, _ => (.nil, Γ)
  | f + 1, top, Γ, s =>
    let s1 := lcg s
    let s2 := lcg s1
    let x := pick s1 Γ.length
    let r := pick s 16
    let cont (st : Prog → Prog) (Γ' : List St) := let (k, Γk) := genB f top Γ' s2; (st k, Γk)
    if r < 9 then
      -- follow the protocol at x (or deviate if it is gone or done)
      match get Γ x with
      | some (.live (.send p)) => cont (.act x .send) (setL Γ x (.live p))
      | some (.live (.recv p)) => cont (.act x .recv) (setL Γ x (.live p))
      | some (.live .done) => if pick s2 3 == 0 then cont (.close x) (setL Γ x .gone) else (.nil, Γ)
      | _ => if pick s2 4 == 0 then cont (.act x .send) Γ else (.nil, Γ)
    else if r < 10 then cont (.act x (if pick s2 2 == 0 then .send else .recv)) Γ
    else if r < 11 then cont (.close x) (setL Γ x .gone)
    else if r < 13 && top then
      match get Γ x with
      | some (.live p) => cont (.move x) (setL Γ x .gone ++ [.live p])
      | _ => cont (.move x) Γ
    else if r < 16 && top then
      let (a, Γa) := genB (f / 2) false Γ (lcg s2)
      let (b, _) := genB (f / 2) false Γ (lcg (lcg s2))
      cont (.branch a b) Γa
    else (.nil, Γ)

def ind (n : Nat) : String := String.mk (List.replicate (4 * n) ' ')

def nm (n : Nat) (x : Nat) : String := if x < n then s!"c{x}" else s!"m{x}"

/-- RCE lines of a body; `n` parameters, `w` names in scope, `v` fresh
    value counter -/
def render (n : Nat) (d : Nat) : Prog → Nat → Nat → List String × Nat × Nat
  | .nil, w, v => ([], w, v)
  | .act x .send k, w, v =>
    let (r, w, v) := render n d k w v; ((ind d ++ nm n x ++ " <- 1") :: r, w, v)
  | .act x .recv k, w, v =>
    let (r, w', v') := render n d k w (v + 1); ((ind d ++ s!"let v{v} = <- " ++ nm n x) :: r, w', v')
  | .close x k, w, v =>
    let (r, w, v) := render n d k w v; ((ind d ++ nm n x ++ ".close()") :: r, w, v)
  | .move x k, w, v =>
    let (r, w', v') := render n d k (w + 1) v; ((ind d ++ s!"let {nm n w} = " ++ nm n x) :: r, w', v')
  | .branch a b k, w, v =>
    let (la, _, v) := render n (d + 1) a w v
    let (lb, _, v) := render n (d + 1) b w v
    let la := if la.isEmpty then [ind (d + 1) ++ "emit 0"] else la
    let lb := if lb.isEmpty then [ind (d + 1) ++ "emit 0"] else lb
    let (r, w, v) := render n d k w v
    ([ind d ++ "if k > 0"] ++ la ++ [ind d ++ "else"] ++ lb ++ r, w, v)

def protoLines : P → List String
  | .send k => "    send i64" :: protoLines k
  | .recv k => "    recv i64" :: protoLines k
  | _ => []

def jstr (s : String) : String := "\"" ++ (s.replace "\n" "\\n") ++ "\""

def vector (seed : Nat) : String :=
  let s0 := lcg (seed * 104729 + 7)
  let n := 1 + pick s0 2
  let ps := (List.range n).map fun i => genP (1 + pick (lcg (s0 + i)) 3) (lcg (s0 * 31 + i))
  let Γ : List St := ps.map .live
  let (body, _) := genB 8 true Γ (lcg (s0 + 99))
  let protos := (List.range n).zip ps |>.map fun (i, p) => "\n".intercalate ((s!"protocol P{i}") :: protoLines p)
  let params := ", ".intercalate ((List.range n).map fun i => s!"c{i}: session P{i}")
  let (ls, _, _) := render n 1 body n 0
  let ls := if ls.isEmpty then [ind 1 ++ "emit 0"] else ls
  let src := "\n\n".intercalate protos ++ "\n\n" ++ "\n".intercalate ((s!"fn body({params}, k: i64)") :: ls) ++ "\n"
  let res := match chkBody Γ body with | .ok _ => "ok" | .error e => e
  "{\"src\":" ++ jstr src ++ ",\"expect\":" ++ jstr res ++ "}"

def main : IO Unit := do
  for s in List.range 120 do
    IO.println (vector (s + 1))
