import RCE.Emit
/-! The Rust printer for MiniRust (the text the emitter writes). -/
namespace RCE.Emit

def prelude : String := "#![allow(unused_mut, unused_variables, unreachable_code, unused_braces, unused_assignments, dead_code)]
#[derive(Clone, Copy, PartialEq, Debug)] pub enum Val { Int(i128), Bool(bool) }
#[derive(Clone, Copy, PartialEq, Debug)] pub enum Trap { Overflow, DivZero, BadIndex, LoopBound }
#[derive(Clone, PartialEq, Debug)] pub enum Refusal { User(&'static str), Trap(Trap) }
impl From<Trap> for Refusal { fn from(t: Trap) -> Self { Refusal::Trap(t) } }
pub enum Op { Add, Sub, Mul, Div, Rem, Lt, Eq }
fn get(v: &[Val], i: usize) -> Result<Val, Trap> { v.get(i).copied().ok_or(Trap::BadIndex) }
fn set(v: &mut Vec<Val>, i: usize, x: Val) -> Result<(), Trap> { if i < v.len() { v[i] = x; Ok(()) } else { Err(Trap::BadIndex) } }
fn truthy(v: Val) -> Result<bool, Trap> { match v { Val::Bool(b) => Ok(b), _ => Err(Trap::BadIndex) } }
fn chk(lo: i128, hi: i128, r: Option<i128>) -> Result<Val, Trap> { match r { Some(r) if lo <= r && r <= hi => Ok(Val::Int(r)), _ => Err(Trap::Overflow) } }
fn binop(op: Op, t: Option<(i128, i128)>, a: Val, b: Val) -> Result<Val, Trap> {
  match (op, t, a, b) {
    (Op::Add, Some((lo, hi)), Val::Int(x), Val::Int(y)) => chk(lo, hi, x.checked_add(y)),
    (Op::Sub, Some((lo, hi)), Val::Int(x), Val::Int(y)) => chk(lo, hi, x.checked_sub(y)),
    (Op::Mul, Some((lo, hi)), Val::Int(x), Val::Int(y)) => chk(lo, hi, x.checked_mul(y)),
    (Op::Div, Some((lo, hi)), Val::Int(x), Val::Int(y)) => if y == 0 { Err(Trap::DivZero) } else { chk(lo, hi, x.checked_div(y)) },
    (Op::Rem, Some((lo, hi)), Val::Int(x), Val::Int(y)) => if y == 0 { Err(Trap::DivZero) } else { chk(lo, hi, x.checked_rem(y)) },
    (Op::Lt, _, Val::Int(x), Val::Int(y)) => Ok(Val::Bool(x < y)),
    (Op::Eq, _, x, y) => Ok(Val::Bool(x == y)),
    _ => Err(Trap::BadIndex),
  }
}
"

def pInt (n : Int) : String := s!"({n}i128)"
def pVal : Val → String
  | .int n => s!"Val::Int({pInt n})"
  | .bool b => s!"Val::Bool({b})"
def pTrap : Trap → String
  | .overflow => "Trap::Overflow" | .divZero => "Trap::DivZero"
  | .badIndex => "Trap::BadIndex" | .loopBound => "Trap::LoopBound"
def pOp : BinOp → String
  | .add => "Op::Add" | .sub => "Op::Sub" | .mul => "Op::Mul" | .div => "Op::Div"
  | .rem => "Op::Rem" | .lt => "Op::Lt" | .eq => "Op::Eq"
def pTy : Ty → String
  | .int lo hi => s!"Some(({pInt lo}, {pInt hi}))"
  | .bool => "None"

def RE.toRust : RE → String
  | .ok v => s!"Ok::<Val, Trap>({pVal v})"
  | .err m => s!"Err::<Val, Trap>({pTrap m})"
  | .fld i => s!"get(&*fs, {i})"
  | .arg i => s!"get(args, {i})"
  | .bin op t a b => s!"(match {a.toRust} \{ Err(m) => Err(m), Ok(x) => match {b.toRust} \{ Err(m) => Err(m), Ok(y) => binop({pOp op}, {pTy t}, x, y) } })"
  | .ite c a b => s!"(match {c.toRust} \{ Err(m) => Err(m), Ok(v) => match truthy(v) \{ Err(m) => Err(m), Ok(true) => {a.toRust}, Ok(false) => {b.toRust} } })"

def RS.toRust : RS → String
  | .skip => "{}"
  | .set i e => s!"\{ let v = {e.toRust}?; set(fs, {i}, v)?; }"
  | .seq a b => a.toRust ++ "\n" ++ b.toRust
  | .ifelse c a b => s!"if truthy({c.toRust}?)? \{\n{a.toRust}\n} else \{\n{b.toRust}\n}"
  | .loop b => s!"loop \{\n{b.toRust}\n}"
  | .brk => "break;"
  | .fail x => s!"return Err(Refusal::User({repr x}));"
  | .trap m => s!"return Err(Refusal::Trap({pTrap m}));"
  | .letK d n => s!"let mut k_{d}: u64 = {n};"
  | .decK d => s!"k_{d} -= 1;"
  | .ifKZero d s => s!"if k_{d} == 0 \{\n{s.toRust}\n}"
  | .ifNot c s => s!"if !truthy({c.toRust}?)? \{\n{s.toRust}\n}"
  | .push e => s!"os.push({e.toRust}?);"

/-- The emitted step body as a Rust function. -/
def emitFn (name : String) (s : Stmt) : String :=
  s!"pub fn {name}(fs: &mut Vec<Val>, os: &mut Vec<Val>, args: &[Val]) -> Result<(), Refusal> \{\n{(compile 0 s).toRust}\nOk(())\n}\n"

end RCE.Emit
