import RCE
/-! Translation-validation harness: random core step bodies, printed as Rust by
    the verified emitter, with the expected outcome computed by `exec`. -/
open RCE RCE.Emit

structure G where s : Nat
def G.next (g : G) : Nat × G := let s := (g.s * 6364136223846793005 + 1442695040888963407) % 18446744073709551616; (s / 65536, ⟨s⟩)
def G.below (g : G) (n : Nat) : Nat × G := let (x, g) := g.next; (x % n, g)

instance : Inhabited G := ⟨⟨0⟩⟩
instance : Inhabited Expr := ⟨.lit (.int 0)⟩
instance : Inhabited Stmt := ⟨.skip⟩
def ty : Ty := .int (-1000) 1000
partial def genE (g : G) (d : Nat) : Expr × G :=
  let (k, g) := g.below (if d = 0 then 4 else 8)
  match k with
  | 0 => let (n, g) := g.below 41; (.lit (.int (n - 20 : Int)), g)
  | 1 => let (i, g) := g.below 5; (.field i, g)
  | 2 => let (i, g) := g.below 3; (.arg i, g)
  | 3 => let (b, g) := g.below 9; (if b = 0 then .trap .overflow else .lit (.bool (b % 2 = 0)), g)
  | 7 => let (c, g) := genE g (d-1); let (a, g) := genE g (d-1); let (b, g) := genE g (d-1); (.ite c a b, g)
  | _ => let (o, g) := g.below 7
         let op := [BinOp.add, .sub, .mul, .div, .rem, .lt, .eq].getD o .add
         let (a, g) := genE g (d-1); let (b, g) := genE g (d-1); (.bin op ty a b, g)

partial def genS (g : G) (d : Nat) : Stmt × G :=
  let (k, g) := g.below (if d = 0 then 4 else 8)
  match k with
  | 0 => let (i, g) := g.below 5; let (e, g) := genE g 2; (.assign i e, g)
  | 1 => let (e, g) := genE g 2; (.emit e, g)
  | 2 => let (b, g) := g.below 6; (if b = 0 then .fail "no" else .skip, g)
  | 3 => let (i, g) := g.below 4; let (e, g) := genE g 1; (.assign i e, g)
  | 4 | 5 => let (a, g) := genS g (d-1); let (b, g) := genS g (d-1); (.seq a b, g)
  | 6 => let (c, g) := genE g 2; let (a, g) := genS g (d-1); let (b, g) := genS g (d-1); (.ifs c a b, g)
  | _ => let (n, g) := g.below 4; let (i, g) := g.below 4; let (m, g) := g.below 6
         let (b, g) := genS g (d-1)
         let step : Stmt := .assign i (.bin .add ty (.field i) (.lit (.int 1)))
         (.loop n (.bin .lt ty (.field i) (.lit (.int (m : Int)))) (.seq b step), g)

def genVals (g : G) (n : Nat) : List Val × G :=
  (List.range n).foldl (fun (acc, g) _ =>
    let (k, g) := g.below 12
    let v : Val := if k = 0 then .bool true else .int ((k : Int) - 5)
    (acc ++ [v], g)) ([], g)

def pList (vs : List Val) : String := "vec![" ++ ", ".intercalate (vs.map pVal) ++ "]"
def pRef : Refusal → String
  | .user x => s!"Refusal::User({repr x})"
  | .trap m => s!"Refusal::Trap({pTrap m})"
def pRes : SRes → String
  | .ok fs os => s!"Ok(({pList fs}, {pList os}))"
  | .ref r => s!"Err({pRef r})"

def build (nprog ninp : Nat) : String := Id.run do
  let mut g : G := ⟨20261012⟩
  let mut fns := ""
  let mut tests := ""
  for p in List.range nprog do
    let (s, g1) := genS g 4; g := g1
    fns := fns ++ emitFn s!"p{p}" s
    for _ in List.range ninp do
      let (fs, g1) := genVals g 4; let (args, g2) := genVals g1 2; g := g2
      let exp := exec args s fs []
      tests := tests ++ s!"  \{ let mut fs = {pList fs}; let mut os: Vec<Val> = vec![]; let r = p{p}(&mut fs, &mut os, &{pList args}).map(|_| (fs, os)); let e: Result<(Vec<Val>, Vec<Val>), Refusal> = {pRes exp}; if r != e \{ bad += 1; eprintln!(\"p{p}: \{:?} vs \{:?}\", r, e); } n += 1; if e.is_err() \{ refused += 1; } }\n"
  return prelude ++ fns ++ "fn main() {\n  let mut n = 0u64; let mut bad = 0u64; let mut refused = 0u64;\n" ++ tests ++
    "  println!(\"cases {} mismatches {} refused {}\", n, bad, refused);\n  if bad > 0 { std::process::exit(1) }\n}\n"

#eval IO.FS.writeFile "emit-test/main.rs" (build 300 10)
