//! P3 floats under HC-13. WebAssembly float arithmetic is IEEE 754 with
//! round-to-nearest-even, no contraction and no flush to zero, except for
//! one freedom: the bits of a NaN result (sign and payload) are left to the
//! machine. x86_64 makes 0/0 the negative quiet NaN, aarch64 the positive
//! one, and a module can see those bits (reinterpret, store then load).
//!
//! So every module is rewritten before it runs: after each float
//! instruction that may make a NaN, a NaN result is replaced by the
//! canonical quiet NaN (f32 0x7fc00000, f64 0x7ff8000000000000). Bit
//! operations (neg, abs, copysign, reinterpret, load, store) are exact
//! already. With that, a module's results are a function of its inputs on
//! every host. SIMD is not enabled (relaxed SIMD is nondeterministic by
//! design).
//!
//! Floats are admitted only when the probe passes: a fixed table of add,
//! sub, mul, div, sqrt, min, max, rounding, promote and demote over edge
//! inputs (signed zeros, subnormals, ties, overflow, infinities, NaNs) run
//! through the rewriter on the engine, whose digest must equal the digest
//! of the same table on the CPU reference (Rust, IEEE) and the constant
//! [`REFERENCE`]. The outcome is logged as `host.Exact` (HC-13).
use wasm_encoder::reencode::Reencode;
use wasmparser::{FunctionBody, Operator, Parser, Payload};

pub const CANON_F32: u32 = 0x7fc0_0000;
pub const CANON_F64: u64 = 0x7ff8_0000_0000_0000;

/// The probe digest on an IEEE CPU; equal on every conforming host.
pub const REFERENCE: &str = "47a0426609e8de5f";

fn may_nan_f32(op: &Operator) -> bool {
    use Operator::*;
    matches!(op, F32Add | F32Sub | F32Mul | F32Div | F32Sqrt | F32Min | F32Max | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32DemoteF64)
}
fn may_nan_f64(op: &Operator) -> bool {
    use Operator::*;
    matches!(op, F64Add | F64Sub | F64Mul | F64Div | F64Sqrt | F64Min | F64Max | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64PromoteF32)
}

struct Canon {
    /// number of parameters of each defined function, in order
    params: Vec<u32>,
    next: usize,
}

impl Reencode for Canon {
    type Error = std::convert::Infallible;

    fn parse_function_body(&mut self, code: &mut wasm_encoder::CodeSection, func: FunctionBody<'_>) -> Result<(), wasm_encoder::reencode::Error<Self::Error>> {
        let np = self.params.get(self.next).copied().unwrap_or(0);
        self.next += 1;
        let mut locals = vec![];
        let mut n = np;
        for pair in func.get_locals_reader()? {
            let (cnt, ty) = pair?;
            n += cnt;
            locals.push((cnt, self.val_type(ty)?));
        }
        // float scratch locals only where needed, so an integer-only module
        // stays integer-only (and runs where floats are not admitted)
        let (mut u32_, mut u64_) = (false, false);
        let mut scan = func.get_operators_reader()?;
        while !scan.eof() {
            let op = scan.read()?;
            u32_ |= may_nan_f32(&op);
            u64_ |= may_nan_f64(&op);
        }
        let t32 = n;
        let t64 = n + u32_ as u32;
        if u32_ {
            locals.push((1, wasm_encoder::ValType::F32));
        }
        if u64_ {
            locals.push((1, wasm_encoder::ValType::F64));
        }
        let mut f = wasm_encoder::Function::new(locals);
        let mut r = func.get_operators_reader()?;
        while !r.eof() {
            let op = r.read()?;
            let (c32, c64) = (may_nan_f32(&op), may_nan_f64(&op));
            f.instruction(&self.instruction(op)?);
            use wasm_encoder::Instruction as I;
            if c32 {
                // t = x; select(t, canon, t == t)
                f.instruction(&I::LocalTee(t32));
                f.instruction(&I::F32Const(f32::from_bits(CANON_F32).into()));
                f.instruction(&I::LocalGet(t32));
                f.instruction(&I::LocalGet(t32));
                f.instruction(&I::F32Eq);
                f.instruction(&I::Select);
            }
            if c64 {
                f.instruction(&I::LocalTee(t64));
                f.instruction(&I::F64Const(f64::from_bits(CANON_F64).into()));
                f.instruction(&I::LocalGet(t64));
                f.instruction(&I::LocalGet(t64));
                f.instruction(&I::F64Eq);
                f.instruction(&I::Select);
            }
        }
        code.function(&f);
        Ok(())
    }
}

/// The module (binary or text) with every NaN-making float result
/// canonicalized.
pub fn canonicalize(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let bin = wat::parse_bytes(bytes).map_err(|e| format!("invalid module: {e}"))?;
    let mut types: Vec<u32> = vec![];
    let mut params = vec![];
    for p in Parser::new(0).parse_all(&bin) {
        match p.map_err(|e| format!("invalid module: {e}"))? {
            Payload::TypeSection(r) => {
                for g in r.into_iter_err_on_gc_types() {
                    types.push(g.map_err(|e| e.to_string())?.params().len() as u32);
                }
            }
            Payload::FunctionSection(r) => {
                for t in r {
                    let t = t.map_err(|e| e.to_string())?;
                    params.push(types.get(t as usize).copied().ok_or("bad type index")?);
                }
            }
            _ => {}
        }
    }
    let mut out = wasm_encoder::Module::new();
    Canon { params, next: 0 }.parse_core_module(&mut out, Parser::new(0), &bin).map_err(|e| format!("rewrite: {e:?}"))?;
    Ok(out.finish())
}

fn f64s() -> Vec<f64> {
    let mut v = vec![
        0.0, -0.0, 1.0, -1.0, 0.1, 3.0, 1.0 / 3.0, 1e308, -1e308, f64::MIN_POSITIVE, 5e-324, 2.5, -2.5, 0.5,
        9007199254740993.0, 1.0 + f64::EPSILON, f64::INFINITY, f64::NEG_INFINITY, f64::NAN,
    ];
    v.push(f64::from_bits(0x7ff0_0000_0000_0004)); // signalling NaN with a payload
    v.push(f64::from_bits(0xfff8_0000_0000_0001)); // negative NaN with a payload
    v
}

const OPS2: [&str; 6] = ["add", "sub", "mul", "div", "min", "max"];
const OPS1: [&str; 5] = ["sqrt", "ceil", "floor", "trunc", "nearest"];

pub fn canon_bits32(b: u32) -> u32 {
    canon32(f32::from_bits(b))
}
pub fn canon_bits64(b: u64) -> u64 {
    canon64(f64::from_bits(b))
}

fn canon64(x: f64) -> u64 {
    if x.is_nan() { CANON_F64 } else { x.to_bits() }
}
fn canon32(x: f32) -> u32 {
    if x.is_nan() { CANON_F32 } else { x.to_bits() }
}
fn mix(acc: u64, bits: u64) -> u64 {
    (acc ^ bits).wrapping_mul(0x0000_0100_0000_01b3)
}

macro_rules! wasm_minmax {
    ($t:ty, $a:expr, $b:expr, $min:expr) => {{
        let (a, b): ($t, $t) = ($a, $b);
        if a.is_nan() || b.is_nan() {
            <$t>::NAN
        } else if a == b {
            // signed zeros: min prefers -0, max prefers +0
            if $min { <$t>::from_bits(a.to_bits() | b.to_bits()) } else { <$t>::from_bits(a.to_bits() & b.to_bits()) }
        } else if $min {
            if a < b { a } else { b }
        } else if a > b {
            a
        } else {
            b
        }
    }};
}

fn ref2_64(op: &str, a: f64, b: f64) -> f64 {
    match op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => a / b,
        "min" => wasm_minmax!(f64, a, b, true),
        _ => wasm_minmax!(f64, a, b, false),
    }
}
fn ref1_64(op: &str, a: f64) -> f64 {
    match op {
        "sqrt" => a.sqrt(),
        "ceil" => a.ceil(),
        "floor" => a.floor(),
        "trunc" => a.trunc(),
        _ => a.round_ties_even(),
    }
}
fn ref2_32(op: &str, a: f32, b: f32) -> f32 {
    match op {
        "add" => a + b,
        "sub" => a - b,
        "mul" => a * b,
        "div" => a / b,
        "min" => wasm_minmax!(f32, a, b, true),
        _ => wasm_minmax!(f32, a, b, false),
    }
}
fn ref1_32(op: &str, a: f32) -> f32 {
    match op {
        "sqrt" => a.sqrt(),
        "ceil" => a.ceil(),
        "floor" => a.floor(),
        "trunc" => a.trunc(),
        _ => a.round_ties_even(),
    }
}

/// The probe digest on this CPU (the reference side).
pub fn reference_digest() -> String {
    let xs = f64s();
    let mut acc: u64 = 0xcbf2_9ce4_8422_2325;
    for &a in &xs {
        for op in OPS1 {
            acc = mix(acc, canon64(ref1_64(op, a)));
            acc = mix(acc, canon32(ref1_32(op, a as f32)) as u64);
        }
        acc = mix(acc, canon32(a as f32) as u64);
        acc = mix(acc, canon64((a as f32) as f64));
        for &b in &xs {
            for op in OPS2 {
                acc = mix(acc, canon64(ref2_64(op, a, b)));
                acc = mix(acc, canon32(ref2_32(op, a as f32, b as f32)) as u64);
            }
        }
    }
    format!("{acc:016x}")
}

/// The probe module: the same table, in WebAssembly text.
pub fn probe_wat() -> String {
    let xs = f64s();
    let c64 = |x: f64| format!("(f64.reinterpret_i64 (i64.const {}))", x.to_bits() as i64);
    // a as f32 in Rust is f32.demote_f64 in Wasm; the probe demotes in Wasm
    let c32 = |x: f64| format!("(f32.demote_f64 {})", c64(x));
    let mut b = String::new();
    let step = |b: &mut String, e: String| {
        b.push_str(&format!("local.get 0 {e} i64.xor i64.const 0x100000001b3 i64.mul local.set 0\n"));
    };
    for &a in &xs {
        for op in OPS1 {
            step(&mut b, format!("(i64.reinterpret_f64 (f64.{op} {}))", c64(a)));
            step(&mut b, format!("(i64.extend_i32_u (i32.reinterpret_f32 (f32.{op} {})))", c32(a)));
        }
        step(&mut b, format!("(i64.extend_i32_u (i32.reinterpret_f32 {}))", c32(a)));
        step(&mut b, format!("(i64.reinterpret_f64 (f64.promote_f32 {}))", c32(a)));
        for &x in &xs {
            for op in OPS2 {
                step(&mut b, format!("(i64.reinterpret_f64 (f64.{op} {} {}))", c64(a), c64(x)));
                step(&mut b, format!("(i64.extend_i32_u (i32.reinterpret_f32 (f32.{op} {} {})))", c32(a), c32(x)));
            }
        }
    }
    format!("(module (func (export \"probe\") (result i64) (local i64) i64.const 0xcbf29ce484222325 local.set 0\n{b}local.get 0))")
}

/// Run the probe on `engine` through the rewriter. The engine digest, or
/// why it could not run.
pub fn engine_digest(engine: &wasmi::Engine) -> Result<String, String> {
    let bin = canonicalize(probe_wat().as_bytes())?;
    let m = wasmi::Module::new(engine, &bin[..]).map_err(|e| e.to_string())?;
    let mut s = wasmi::Store::new(engine, ());
    let _ = s.set_fuel(u64::MAX);
    let i = wasmi::Linker::<()>::new(engine).instantiate_and_start(&mut s, &m).map_err(|e| e.to_string())?;
    let f = i.get_typed_func::<(), i64>(&s, "probe").map_err(|e| e.to_string())?;
    Ok(format!("{:016x}", f.call(&mut s, ()).map_err(|e| e.to_string())? as u64))
}

/// The HC-13 verdict for Wasm floats on this host: (admitted, the
/// `host.Exact` payload).
pub fn probe(engine: &wasmi::Engine) -> (bool, String) {
    let cpu = reference_digest();
    let eng = engine_digest(engine).unwrap_or_else(|e| format!("error: {e}"));
    let ok = cpu == REFERENCE && eng == REFERENCE;
    let ops = if ok { "[\"wasm.f32\",\"wasm.f64\"]" } else { "[]" };
    (ok, format!("{{\"exact\":{ops},\"probe\":\"{eng}\",\"reference\":\"{REFERENCE}\",\"cpu\":\"{cpu}\",\"nan\":\"canonical\"}}"))
}
