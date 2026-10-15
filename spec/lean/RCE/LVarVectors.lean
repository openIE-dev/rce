/-
Test vectors from the executable LVar semantics (T29a). `main` prints one
JSON line per vector: a run of `stepAt` (LVarAlloc.lean, proved sound
against `Step` by `stepAt_sound`) over the lattice `L7` (0..6 under max,
top 6), with the operations in the order the schedule ran them and their
results, and the outcome. checker/tests/lean_lvar.rs regenerates this
output with `lake env lean --run`, compares it with the checked-in
checker/tests/vectors/lvar-lean.jsonl, and replays every vector on the
evaluator (and the emitters), requiring the same results and the same
error. Locations come from the counter: thread i's k-th `new` is `(i, k)`.
-/
import RCE.LVarAlloc
open RCE.LVarQD RCE.LVarAlloc

/-- 0..6 under max, top 6 (the RCE cell `lvar_new(0, 6)` over integers) -/
abbrev L7 := Fin 7

def L7.join (a b : L7) : L7 := if a.val ≤ b.val then b else a

instance : Lat L7 where
  join := L7.join
  bot := 0
  top := 6
  join_assoc := by decide
  join_comm := by decide
  join_idem := by decide
  bot_join := by decide
  top_join := by decide
  bot_ne_top := by decide

abbrev Loc := Nat × Nat

def mk (k : Nat) : L7 := ⟨k % 7, by omega⟩

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616
def r (s n : Nat) : Nat := (s / 65536) % n

/-- thread i: `new (i, 0)`, then `len` operations on the cells of all threads -/
def thread (nth i len s : Nat) : List (Op Loc L7) :=
  let rec go (k s : Nat) : List (Op Loc L7) :=
    match k with
    | 0 => []
    | k + 1 =>
      let s1 := lcg s
      let l : Loc := (r s1 nth, 0)
      let s2 := lcg s1
      let op : Op Loc L7 :=
        match r s2 10 with
        | 0 | 1 | 2 | 3 => .put l (mk (r (lcg s2) 6 + 1))
        | 4 => .put l (mk (r (lcg s2) 7))
        | 5 | 6 | 7 => .get l [mk (r (lcg s2) 5 + 1)]
        | _ => .freeze l
      op :: go k s2
  .new (i, 0) :: go len s

def prog (seed : Nat) : List (List (Op Loc L7)) :=
  let nth := r seed 2 + 2
  (List.range nth).map fun i => thread nth i (r (lcg (seed + i)) 4 + 1) (lcg (seed * 31 + i))

def sched (seed n : Nat) : List Nat :=
  let rec go (k s : Nat) : List Nat :=
    match k with
    | 0 => []
    | k + 1 => r s 3 :: go k (lcg s)
  go 24 (lcg (seed + 99)) ++ (List.range (8 * n)).map (· % 3)

def jlist (xs : List String) : String := "[" ++ ",".intercalate xs ++ "]"
def sloc (l : Loc) : String := s!"[{l.1},{l.2}]"

def sop : Op Loc L7 → String
  | .put l d => s!"\"put\",{sloc l},{d.val}"
  | .get l q => s!"\"get\",{sloc l},{q.head!.val}"
  | .freeze l => s!"\"freeze\",{sloc l},0"
  | .new l => s!"\"new\",{sloc l},0"

def headOp (c : Cfg Loc L7) (i : Nat) : Option (Op Loc L7) :=
  match pick c.ths i with
  | some (_, ⟨_, op :: _⟩, _) => some op
  | _ => none

def lastRes (c : Cfg Loc L7) (i : Nat) : Nat :=
  match pick c.ths i with
  | some (_, t, _) => (t.done.getLast?.map (·.2.val)).getD 0
  | none => 0

/-- steps taken (op, result) and the outcome -/
def trace : Cfg Loc L7 → List Nat → List String × String
  | c, [] => ([], if c.ths.all (·.todo.isEmpty) then "ok" else "stuck")
  | c, i :: is =>
    match stepAt c i, headOp c i with
    | some .err, some (.put l d) =>
      let why := if Lat.join (c.val l) d = Lat.top then "lvar conflict" else "lvar put after freeze"
      ([s!"[{i}," ++ sop (.put l d) ++ ",0]"], why)
    | some (.ok c'), some op =>
      let rest := trace c' is
      ((s!"[{i}," ++ sop op ++ s!",{lastRes c' i}]") :: rest.1, rest.2)
    | _, _ => trace c is

def vector (seed : Nat) : Option String :=
  let P := prog seed
  let n := (P.map List.length).foldl (· + ·) 0
  let (steps, out) := trace (init P) (sched seed n)
  if out == "stuck" then none
  else some ("{\"seed\":" ++ toString seed ++ ",\"threads\":" ++ toString P.length ++
    ",\"steps\":" ++ jlist steps ++ ",\"out\":\"" ++ out ++ "\"}")

def main : IO Unit := do
  let mut n := 0
  for s in List.range 400 do
    if n < 60 then
      match vector (s + 1) with
      | some v => IO.println v; n := n + 1
      | none => pure ()
