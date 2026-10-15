/-
  Test vectors extracted from `RCE.Admit.window` (T40). `main` prints one
  JSON line per vector: the inputs in an arrival order and the window's
  entries as computed by the Lean definition. `checker/tests/lean_admit.rs`
  runs this file with `lake env lean --run`, compares the output with the
  checked-in `checker/tests/vectors/admit-lean.jsonl`, and feeds every
  vector to `host.rs` in its arrival order, requiring the same entries.

  Key encoding (injective for prio < 256, src < 100, seq < 1000,
  pay < 1000): priority descending, then source, then seq, then payload,
  which is the HC-15 key with fixed-width source ids and payload text.
-/
import RCE.Admit
open RCE.Admit

structure In where
  prio : Nat
  src : Nat
  seq : Nat
  pay : Nat
deriving Repr

def key (i : In) : Nat := (((255 - i.prio) * 100 + i.src) * 1000 + i.seq) * 1000 + i.pay

def lcg (s : Nat) : Nat := (s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616

/-- Inputs of `ns` sources with contiguous seqs, priority 200 or 0. -/
def inputs (seed : Nat) : List In :=
  let rec go (src : Nat) (fuel : Nat) (s : Nat) : List In :=
    match fuel with
    | 0 => []
    | f + 1 =>
      let k := (s / 65536) % 4 + 1
      let rec each (q : Nat) (fq : Nat) (s : Nat) : List In :=
        match fq with
        | 0 => []
        | g + 1 =>
          let s' := lcg s
          let p := if (s' / 65536) % 3 == 0 then 200 else 0
          { prio := p, src := src, seq := q, pay := (s' / 4096) % 1000 } :: each (q + 1) g s'
      each 1 k (lcg s) ++ go (src + 1) f (lcg (lcg s))
  go 0 ((seed % 3) + 2) (lcg seed)

/-- A permutation standing in for arrival timing, plus one retransmission. -/
def shuffle (l : List In) (s : Nat) : List In :=
  let rec go (l : List In) (s : Nat) (fuel : Nat) : List In :=
    match fuel, l with
    | 0, _ => l
    | _, [] => []
    | f + 1, _ =>
      let i := (s / 65536) % l.length
      match l.get? i with
      | some x => x :: go (l.eraseIdx i) (lcg s) f
      | none => l
  let p := go l s l.length
  match p.head? with
  | some x => p ++ [x]
  | none => p

def showIn (i : In) : String := s!"[{i.prio},{i.src},{i.seq},{i.pay}]"
def showKey (i : In) : String := s!"[{i.src},{i.seq}]"
def jlist (xs : List String) : String := "[" ++ ",".intercalate xs ++ "]"

def vector (seed : Nat) : String :=
  let a := shuffle (inputs seed) (lcg (seed + 7))
  "{\"arrival\":" ++ jlist (a.map showIn) ++ ",\"window\":" ++ jlist ((window key a).map showKey) ++ "}"

def main : IO Unit := do
  for s in List.range 40 do
    IO.println (vector (s + 1))
