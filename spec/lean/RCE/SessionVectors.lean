/-
  Test vectors extracted from `RCE.SessionRec` (T28) through the encoding
  of `RCE.SessionEnc`. `main` prints one JSON line per vector: a surface
  protocol `s` as RCE source, its dual `sdual s` as RCE source, and a joint
  run of the encoded endpoints `enc s` and `enc (sdual s)` under `Red`
  (choices taken by a fixed generator). At every point of the run the
  line records the actions each endpoint can take (`PStep` after silent
  `mu` unfoldings), the action endpoint A took, and at the end whether
  both endpoints are `done`.

  `checker/tests/lean_session.rs` runs this file with
  `lake env lean --run`, compares the output with the checked-in
  `checker/tests/vectors/session-lean.jsonl`, parses both protocols with
  the checker, requires the checker's `dual` of the first to equal the
  second, and drives the checker's protocol automaton
  (`check06b::ProtoCursor`) on both sides through the run, requiring the
  same enabled actions at every point and the same end.
-/
import RCE.SessionEnc
open RCE.SessionRec

deriving instance BEq for P

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616
def pick (s n : Nat) : Nat := (s / 65536) % n

/-- A surface protocol. `inLoop` allows `brk`; `body` asks for a message
    step (a loop body begins with one, W-SES-3); a `brk` is the last step
    of its sequence. -/
def gen : Nat → Bool → Bool → Nat → SP
  | 0, inLoop, body, s =>
    if body then .send .done else if inLoop && pick s 2 == 0 then .brk else .done
  | d + 1, inLoop, body, s =>
    let s1 := lcg s
    let s2 := lcg s1
    match (if body then pick s 4 else pick s 9) with
    | 0 => .send (gen d inLoop false s1)
    | 1 => .recv (gen d inLoop false s1)
    | 2 => .choose (gen d inLoop false s1) (gen d inLoop false s2)
    | 3 => .offer (gen d inLoop false s1) (gen d inLoop false s2)
    | 4 => .send (gen d inLoop false s1)
    | 5 => .recv (gen d inLoop false s1)
    | 6 => .loop (gen d true true s1) (gen d inLoop false s2)
    | 7 => if inLoop then .brk else .done
    | _ => .done

def ind (n : Nat) : String := String.mk (List.replicate (4 * n) ' ')

/-- RCE source lines of a step sequence at indent `n` -/
partial def lines (n : Nat) : SP → List String
  | .done => []
  | .brk => [ind n ++ "break"]
  | .send k => (ind n ++ "send i64") :: lines n k
  | .recv k => (ind n ++ "recv i64") :: lines n k
  | .choose a b => [ind n ++ "choose", ind (n + 1) ++ "L"] ++ lines (n + 2) a ++ [ind (n + 1) ++ "R"] ++ lines (n + 2) b
  | .offer a b => [ind n ++ "offer", ind (n + 1) ++ "L"] ++ lines (n + 2) a ++ [ind (n + 1) ++ "R"] ++ lines (n + 2) b
  | .loop b r => (ind n ++ "loop") :: lines (n + 1) b ++ lines n r

def src (name : String) (s : SP) : String :=
  "\n".intercalate (("protocol " ++ name) :: lines 1 s)

/-- unfold `mu` at the head (the silent step of `PStep`) -/
def unfoldN : Nat → P → P
  | 0, p => p
  | f + 1, .mu a => unfoldN f (subst 0 (.mu a) a)
  | _, p => p

def actsOf : P → List String
  | .send _ => ["send"]
  | .recv _ => ["recv"]
  | .choose _ _ => ["choose:L", "choose:R"]
  | .offer _ _ => ["offer:L", "offer:R"]
  | _ => []

def jstr (s : String) : String := "\"" ++ s ++ "\""
def jlist (xs : List String) : String := "[" ++ ",".intercalate xs ++ "]"

/-- a joint run under `Red`: at each point the enabled actions of both
    endpoints and the one A takes -/
def run : Nat → P → P → Nat → List String × Bool
  | 0, a, b, _ => ([], unfoldN 64 a == .done && unfoldN 64 b == .done)
  | f + 1, a, b, s =>
    let a := unfoldN 64 a
    let b := unfoldN 64 b
    let ea := actsOf a
    let eb := actsOf b
    let here (act : String) := "{\"a\":" ++ jlist (ea.map jstr) ++ ",\"b\":" ++ jlist (eb.map jstr) ++ ",\"take\":" ++ jstr act ++ "}"
    let go (act : String) (a' b' : P) := let (r, e) := run f a' b' (lcg s); (here act :: r, e)
    match a, b with
    | .send a', .recv b' => go "send" a' b'
    | .recv a', .send b' => go "recv" a' b'
    | .choose x y, .offer x' y' => if pick s 2 == 0 then go "choose:L" x x' else go "choose:R" y y'
    | .offer x y, .choose x' y' => if pick s 2 == 0 then go "offer:L" x x' else go "offer:R" y y'
    | _, _ => ([], a == .done && b == .done)


def vector (seed : Nat) : String :=
  let s := gen 5 false false (lcg (seed * 7919 + 13))
  let s := match s with | .done => SP.send .done | x => x
  let (steps, fin) := run 24 (enc s .done .done) (enc (sdual s) .done .done) (lcg (seed + 3))
  "{\"a\":" ++ jstr ((src "A" s).replace "\n" "\\n") ++ ",\"b\":" ++ jstr ((src "B" (sdual s)).replace "\n" "\\n") ++
    ",\"run\":" ++ jlist steps ++ ",\"done\":" ++ toString fin ++ "}"

def main : IO Unit := do
  for s in List.range 60 do
    IO.println (vector (s + 1))
