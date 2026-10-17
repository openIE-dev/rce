import RCE.SessionGlobal
/-! Vectors linking `RCE.SessionGlobal.proj` (T28g) to the checker's
  `mpst::proj`. `main` prints one JSON line per random well-formed global
  type over roles 0..3: `g` (prefix form: `e`, `m p q K`, `b p q A B`) and
  `proj`, the projection on each role (`e`, `s q K`, `r p K`, `S q A B`,
  `B p A B`) or null where plain merge fails. When every role projects, the
  global type is deadlock free (`deadlock_free`); when it has messages only,
  `global_prio` gives the priorities the checker's `session_prio::decide`
  must accept. -/
namespace RCE.SessionGlobal

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616

def gs : G → String
  | .done => "e"
  | .msg p q k => s!"m {p} {q} {gs k}"
  | .br p q a b => s!"b {p} {q} {gs a} {gs b}"

def ls : L → String
  | .done => "e"
  | .send q k => s!"s {q} {ls k}"
  | .recv p k => s!"r {p} {ls k}"
  | .sel q a b => s!"S {q} {ls a} {ls b}"
  | .bra p a b => s!"B {p} {ls a} {ls b}"

/-- a random global type of depth at most `d`; returns the next seed -/
def gen : Nat → Nat → G × Nat
  | 0, s => (.done, s)
  | d + 1, s =>
    let s1 := lcg s
    let p := (s1 / 65536) % 4
    let q0 := (s1 / 1048576) % 3
    let q := if q0 ≥ p then q0 + 1 else q0
    match (s1 / 16777216) % 6 with
    | 0 => (.done, s1)
    | 1 | 2 | 3 => let (k, s2) := gen d s1; (.msg p q k, s2)
    | 4 => let (a, s2) := gen d s1; (.br p q a a, s2)
    | _ => let (a, s2) := gen d s1; let (b, s3) := gen d s2; (.br p q a b, s3)

def vec (s : Nat) : String × Nat :=
  let (g, s') := gen 5 s
  let pr := (List.range 4).map fun r => match proj g r with
    | some l => "\"" ++ ls l ++ "\""
    | none => "null"
  ("{\"g\":\"" ++ gs g ++ "\",\"proj\":[" ++ ",".intercalate pr ++ "]}", s')

end RCE.SessionGlobal

open RCE.SessionGlobal in
def main : IO Unit := do
  let mut s := 20261015
  for _ in [0:200] do
    let (l, s') := vec s
    IO.println l
    s := s'
