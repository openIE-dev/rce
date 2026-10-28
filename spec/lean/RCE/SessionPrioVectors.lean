/-
  Test vectors for `RCE.SessionPrio` (deadlock freedom by priorities) and
  W-SES-5. `main` prints one JSON line per vector: an RCE program (`src`)
  and what Lean decides for it (`ok` or `W-SES-5`).

  A program is `main` plus workers it spawns. Each channel `c` is made in
  `main` as `let (a{c}, b{c}) = chan[session P..](1)`; its end `a` and end
  `b` are held by `main` or passed to one worker. Each process (main and
  every worker) does the steps of the ends it holds in some interleaving;
  `main` spawns each worker at some point of its own sequence.

  The network: process 0 is `main`, process `w` is worker `w`. Session
  channel `c` is `c`; the `t`-th step of `c` (by either end) is the action
  `(c, t)`. A spawn is a one-step channel `nc + w - 1` between `main` (at
  the spawn) and the worker (its first action), so a worker acts only after
  it is spawned. Lean accepts (`ok`) when the program meets W-SES-5's tree
  shape (each spawn passes at most one endpoint, and `main` acts on a
  channel only after it has passed one of its ends on), or when
  `SessionPrio.decideP` finds priorities (then `SessionPrio.decideP_sound`
  gives deadlock freedom for every reachable state; the tree shape has
  `SessionDead.deadlock_free`). Otherwise it reports `W-SES-5`.

  `checker/tests/lean_session_prio.rs` runs this file, compares it with
  `checker/tests/vectors/session-prio-lean.jsonl`, and checks that the
  checker accepts exactly the `ok` programs and reports the others with
  W-SES-5.
-/
import RCE.SessionPrio
open RCE.SessionPrio

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616
def pick (s n : Nat) : Nat := (s / 65536) % n

/-- a protocol from end `a`'s side: `true` send, `false` receive -/
abbrev Proto := List Bool

/-- one step of a process: `.step c side k` (side `true` is end `a`) or
    `.spawn w` (in `main` only) -/
inductive Op where
  | step (c : Nat) (side : Bool) (k : Nat)
  | spawn (w : Nat)

/-- a random interleaving of lists, keeping each list's order -/
def interleave : Nat → List (List Op) → Nat → List Op
  | 0, _, _ => []
  | f + 1, ls, s =>
    let ls := ls.filter (fun l => !l.isEmpty)
    if ls.isEmpty then [] else
    let i := pick s ls.length
    match ls.getD i [] with
    | [] => []
    | x :: rest => x :: interleave f (ls.set i rest) (lcg s)

structure Prog where
  nc : Nat
  protos : List Proto
  /-- holders of end `a` and end `b` of each channel (0 is main) -/
  holders : List (Nat × Nat)
  nw : Nat
  procs : List (List Op)

def gen (seed : Nat) : Prog :=
  let s0 := lcg (seed * 7919 + 13)
  let nc := 1 + pick s0 3
  let nwMax := 1 + pick (lcg s0) 3
  let protos := (List.range nc).map fun c =>
    let s := lcg (s0 + 31 * c + 1)
    (List.range (1 + pick s 2)).map fun k => pick (lcg (s + k)) 2 == 0
  let hold := (List.range nc).map fun c =>
    let s := lcg (lcg (s0 * 3 + c))
    let ha := pick s (nwMax + 1)
    let hb := if pick (lcg s) 9 == 0 then ha else (ha + 1 + pick (lcg (lcg s)) nwMax) % (nwMax + 1)
    (ha, hb)
  -- workers that hold something, renumbered 1..nw
  let used := (List.range nwMax).map (· + 1) |>.filter fun w => hold.any fun (a, b) => a == w || b == w
  let ren := fun h => if h == 0 then 0 else (used.indexOf h) + 1
  let hold := hold.map fun (a, b) => (ren a, ren b)
  let nw := used.length
  let ops := fun p => (List.range nc).foldr (fun c acc =>
      let (ha, hb) := hold.getD c (0, 0)
      let n := (protos.getD c []).length
      let ea := if ha == p then [(List.range n).map (Op.step c true)] else []
      let eb := if hb == p then [(List.range n).map (Op.step c false)] else []
      ea ++ eb ++ acc) []
  let mainOps := ops 0 ++ (List.range nw).map fun w => [Op.spawn (w + 1)]
  let main := interleave 64 mainOps (lcg (s0 + 77))
  let workers := (List.range nw).map fun w => interleave 64 (ops (w + 1)) (lcg (s0 + 101 * w + 5))
  { nc, protos, holders := hold, nw, procs := main :: workers }

/-- the network of `SessionPrio` -/
def net (g : Prog) : List (List Act) × List (Nat × Nat) :=
  let tr : Op → Act := fun
    | .step c _ k => (c, k)
    | .spawn w => (g.nc + w - 1, 0)
  let L := g.procs.enum.map fun (p, ops) =>
    (if p == 0 then [] else [(g.nc + p - 1, 0)]) ++ ops.map tr
  let es := g.holders ++ (List.range g.nw).map fun w => (0, w + 1)
  (L, es)

/-- W-SES-5's tree shape: no spawn passes two endpoints, and `main` acts
    on a channel only after one of its ends has gone to a worker -/
def treeOK (g : Prog) : Bool :=
  let twice := (List.range g.nw).any fun w =>
    (g.holders.filter fun (a, _) => a == w + 1).length + (g.holders.filter fun (_, b) => b == w + 1).length > 1
  let main := g.procs.getD 0 []
  let spawnedBefore : Nat → Nat → Bool := fun i w =>
    (main.take i).any fun | .spawn w' => w' == w | _ => false
  let early := main.enum.any fun (i, op) =>
    match op with
    | .step c _ _ =>
      let (ha, hb) := g.holders.getD c (0, 0)
      let other := if ha == 0 then hb else ha
      -- main holds both ends while the other end is still here
      ha == 0 && hb == 0 || (other != 0 && !spawnedBefore i other)
    | .spawn _ => false
  !twice && !early

def protoName (p : Proto) : String := "P_" ++ String.mk (p.map fun b => if b then 's' else 'r')

def ind := "    "

def render (g : Prog) : String :=
  let pats := (g.protos ++ g.protos.map (·.map (!·))).foldl (fun acc p => if acc.contains p then acc else acc ++ [p]) []
  let protoDecls := pats.map fun p =>
    "\n".intercalate (s!"protocol {protoName p}" :: p.map fun b => ind ++ (if b then "send i64" else "recv i64"))
  let endName := fun (c : Nat) (side : Bool) => (if side then "a" else "b") ++ toString c
  let protoOf := fun (c : Nat) (side : Bool) =>
    let p := g.protos.getD c []
    if side then p else p.map (!·)
  let line := fun (v : Nat) (op : Op) =>
    match op with
    | .step c side k =>
      let send := (protoOf c side).getD k true
      if send then (ind ++ endName c side ++ " <- 1", v) else (ind ++ s!"let v{v} = <- " ++ endName c side, v + 1)
    | .spawn w =>
      let args := (List.range g.nc).foldr (fun c acc =>
        let (ha, hb) := g.holders.getD c (0, 0)
        (if ha == w then [endName c true] else []) ++ (if hb == w then [endName c false] else []) ++ acc) []
      (ind ++ s!"spawn w{w}(" ++ ", ".intercalate args ++ ")", v)
  let body := fun (ops : List Op) =>
    let (ls, _) := ops.foldl (fun (acc : List String × Nat) op => let (l, v) := line acc.2 op; (acc.1 ++ [l], v)) ([], 0)
    ls
  let workers := (List.range g.nw).map fun i =>
    let w := i + 1
    let params := (List.range g.nc).foldr (fun c acc =>
      let (ha, hb) := g.holders.getD c (0, 0)
      (if ha == w then [s!"a{c}: session {protoName (protoOf c true)}"] else []) ++
      (if hb == w then [s!"b{c}: session {protoName (protoOf c false)}"] else []) ++ acc) []
    "\n".intercalate ((s!"fn w{w}(" ++ ", ".intercalate params ++ ")") :: body (g.procs.getD w []))
  let chans := (List.range g.nc).map fun c => ind ++ s!"let (a{c}, b{c}) = chan[session {protoName (protoOf c true)}](1)"
  let main := "\n".intercalate (("fn main()" :: chans) ++ body (g.procs.getD 0 []) ++ [ind ++ "emit 0"])
  "\n\n".intercalate (protoDecls ++ workers ++ [main]) ++ "\n"

def jstr (s : String) : String := "\"" ++ (s.replace "\n" "\\n") ++ "\""

def vector (seed : Nat) : String :=
  let g := gen seed
  let (L, es) := net g
  let tree := treeOK g
  let prio := decideP L es
  let res := if tree || prio then "ok" else "W-SES-5"
  "{\"src\":" ++ jstr (render g) ++ ",\"expect\":" ++ jstr res ++ ",\"tree\":" ++ toString tree ++ ",\"prio\":" ++ toString prio ++ "}"

def main : IO Unit := do
  for s in List.range 150 do
    IO.println (vector (s + 1))
