import RCE.SessionAsync
/-! Vectors linking `RCE.SessionAsync.proj` (T28a, full merge) to the
  checker's `mpst_async::proj`. `main` prints one JSON line per random
  well-formed static global type over roles 0..3: `g` in prefix form
  (`e`, `c p q ha hb A B`, flags `0`/`1`, an absent branch is `e`) and
  `proj`, the projection on each role (`e`, `s q ha hb A B`, `r p ha hb A B`)
  or null where full merge fails. Where every role projects, `deadlock_free`
  says every configuration reachable asynchronously from the projections with
  empty queues is final or steps, and every receiver accepts its queue head;
  the checker explores those configurations and confirms it. -/
namespace RCE.SessionAsync

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616

def fl (b : Bool) : String := if b then "1" else "0"

def gs : G → String
  | .done => "e"
  | .comm p q _ ha hb a b => s!"c {p} {q} {fl ha} {fl hb} {gs a} {gs b}"

def ls : L → String
  | .done => "e"
  | .snd q ha hb a b => s!"s {q} {fl ha} {fl hb} {ls a} {ls b}"
  | .rcv p ha hb a b => s!"r {p} {fl ha} {fl hb} {ls a} {ls b}"

/-- a random static global type of depth at most `d`; returns the next seed -/
def gen : Nat → Nat → G × Nat
  | 0, s => (.done, s)
  | d + 1, s =>
    let s1 := lcg s
    let p := (s1 / 65536) % 4
    let q0 := (s1 / 1048576) % 3
    let q := if q0 ≥ p then q0 + 1 else q0
    match (s1 / 16777216) % 7 with
    | 0 => (.done, s1)
    | 1 | 2 => let (k, s2) := gen d s1; (.comm p q none true false k .done, s2)
    | 3 => let (k, s2) := gen d s1; (.comm p q none false true .done k, s2)
    | 4 => let (a, s2) := gen d s1; (.comm p q none true true a a, s2)
    | _ => let (a, s2) := gen d s1; let (b, s3) := gen d s2; (.comm p q none true true a b, s3)

def vec (s : Nat) : String × Nat :=
  let (g, s') := gen 5 s
  let pr := (List.range 4).map fun r => match proj g r with
    | some l => "\"" ++ ls l ++ "\""
    | none => "null"
  ("{\"g\":\"" ++ gs g ++ "\",\"proj\":[" ++ ",".intercalate pr ++ "]}", s')

end RCE.SessionAsync

open RCE.SessionAsync in
def main : IO Unit := do
  let mut s := 20261015
  for _ in [0:300] do
    let (l, s') := vec s
    IO.println l
    s := s'
