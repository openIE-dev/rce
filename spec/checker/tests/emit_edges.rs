//! T9-L at the edges of 6.3: a differential test of the emit targets
//! (spec 10.5) against the reference evaluator.
//!
//! The module is `tests/emit_edges.rce`, written from the text of 6.3 and
//! 5.15 alone. It has one event per operator (integer, unsigned, comparison,
//! float, f32 and text) and one example per edge of 6.3, each stating the
//! outcome the text gives. This test appends `SEEDED` examples drawn from
//! edge pools by a SplitMix64 generator with the recorded seed `SEED`. The
//! outcome of each seeded example comes from `model`, a reading of the same
//! text written here (where 6.3 gives a behaviour but no trap text, the
//! text is the reference evaluator's, and `model` says so at that line).
//!
//! The test then holds three things:
//!
//! 1. The whole module is admitted by `rce check`: every hand-written and
//!    seeded example holds in the reference evaluator (6.9, W-INT-1).
//! 2. Every target that runs prints the reference observations
//!    (`emit::observe`), compared line for line and byte for byte over the
//!    whole standard output.
//! 3. Run with `--trace`, every target prints, after each example, the head
//!    of that example's log D (6.1: canonical JSON, SHA-256 chain), and the
//!    heads equal those of the reference's Form (`interp::form_json`).
//!
//! Targets: rust (native), python, js, scala (when scala-cli is present),
//! and the Rust output compiled for wasm32-wasip1 and run under a WASI
//! runtime (wasmtime when installed, else Node's `node:wasi`). A target
//! whose tool is missing prints `SKIPPED` and the reason, and is never
//! counted as agreeing.
//!
//! With `RCE_EMIT_EDGES_OUT=DIR` the test also writes the record of the run
//! to DIR: `module.rce` (the module with its seeded examples),
//! `examples.jsonl` (one row per example: its source line, its reference
//! observation lines and the head of its log D), `reference.json` (counts
//! and digests) and one `target-NAME.json` per target (agrees, with the
//! tool and the digests of what it printed, or skipped and why).
use rce_spec_check::emit::{Lang, emit, observe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

#[path = "common/scala_java.rs"]
mod scala_java;
use scala_java::scala_java;

/// The recorded seed of the seeded edge values.
const SEED: u64 = 0x6_3_ED6E_2026_1027;
/// Seeded examples, one event each.
const SEEDED: usize = 500;
/// SHA-256 of the seeded example lines this generator writes for `SEED`;
/// a change to the generator or the pools changes it, in the same commit.
const SEEDED_SHA256: &str = "17f2cfd8bc2d6627e75d71373ee1df9b23777d822a75f0421611e4c8abc47a6b";
/// The display sweep: one text holding every `SWEEP`-th Unicode scalar
/// value, so every target's text display (10.5) is compared across all
/// seventeen planes, not only on the pools.
const SWEEP: u32 = 97;

// ------------------------------------------------------------ generator

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
}

const I64_EDGES: [i64; 31] = [
    i64::MIN, i64::MIN + 1, -(1 << 53) - 1, -(1 << 32), -(1 << 31) - 1, -65, -64, -63, -9, -8, -7, -3,
    -2, -1, 0, 1, 2, 3, 7, 8, 63, 64, 65, 1 << 31, 1 << 32, (1 << 53) + 1, 3037000499, 3037000500,
    i64::MAX / 2 + 1, i64::MAX - 1, i64::MAX,
];
const SHIFTS: [i64; 12] = [-1, 0, 1, 7, 31, 32, 62, 63, 64, 65, 127, i64::MIN];
const U64_EDGES: [u64; 13] = [
    0, 1, 2, 3, 1 << 31, 1 << 32, 4294967296 + 1, 1 << 53, (1 << 63) - 1, 1 << 63, (1 << 63) + 1,
    u64::MAX - 1, u64::MAX,
];
const U8_EDGES: [u8; 10] = [0, 1, 2, 7, 8, 127, 128, 200, 254, 255];
const F64_EDGES: [f64; 36] = [
    0.0, -0.0, 5e-324, -5e-324, 1e-323, 2.225073858507201e-308, 2.2250738585072014e-308,
    -2.2250738585072014e-308, 1e-300, 1e-45, 1.401298464324817e-45, 1.1754943508222875e-38,
    9.999999999999999e-6, 1e-5, 1e-4, 0.1, 0.2, 0.3, 0.49999999999999994, 0.5, -0.5, 1.0, -1.0,
    1.5, 2.5, -2.5, 3.0, 10.0, 4503599627370497.0, 9007199254740992.0, 9007199254740993.0,
    9999999999999998.0, 1e16, 9.223372036854776e18, 3.4028234663852886e38, 1.7976931348623157e308,
];
const PIECES: [&str; 44] = [
    "a", "Z", "e", "0", "7", "42", "-", "+", " ", "\t", "\n", "\u{b}", "\u{c}", "\r", "\u{1c}",
    "\u{1f}", "\u{85}", "\u{a0}", "\u{1680}", "\u{2000}", "\u{200a}", "\u{200b}", "\u{2028}",
    "\u{202f}", "\u{205f}", "\u{3000}", "\u{feff}", "\u{180e}", "\u{e9}", "\u{df}", "\u{130}",
    "\u{131}", "\u{fb03}", "\u{663}", "\u{ff11}", "\u{301}", "\u{ffff}", "\u{10000}", "\u{1f600}",
    "\u{10ffff}", "\u{e000}", "\u{d7ff}", "\u{0}", "\"\\",
];
const INT_TEXTS: [&str; 12] = [
    "9223372036854775807", "9223372036854775808", "-9223372036854775808", "-9223372036854775809",
    "170141183460469231731687303715884105727", "170141183460469231731687303715884105728",
    "-170141183460469231731687303715884105728", "+0", "-0", "007", "+", "1_0",
];

fn gen_i64(r: &mut SplitMix) -> i64 {
    match r.below(10) {
        0..=5 => *r.pick(&I64_EDGES),
        6..=7 => r.below(141) as i64 - 70,
        _ => r.next() as i64,
    }
}
fn gen_u64(r: &mut SplitMix) -> u64 {
    match r.below(10) {
        0..=5 => *r.pick(&U64_EDGES),
        6..=7 => r.below(100),
        _ => r.next(),
    }
}
fn gen_f64(r: &mut SplitMix) -> f64 {
    let x = match r.below(10) {
        0..=4 => *r.pick(&F64_EDGES),
        5..=6 => loop {
            let x = f64::from_bits(r.next());
            if x.is_finite() {
                break x;
            }
        },
        _ => (r.below(2_000_001) as f64 - 1_000_000.0) / 10f64.powi(r.below(8) as i32),
    };
    if r.below(4) == 0 { -x } else { x }
}
fn gen_text(r: &mut SplitMix) -> String {
    if r.below(5) == 0 {
        return r.pick(&INT_TEXTS).to_string();
    }
    (0..r.below(5)).map(|_| *r.pick(&PIECES)).collect()
}

// ------------------------------------------------------------ RCE literals

fn lit_i(v: i128) -> String {
    if v == i64::MIN as i128 { "-9223372036854775807 - 1".into() } else { v.to_string() }
}
/// Rust's `{:?}` of an f64 is its shortest round-trip digits, which is
/// also RCE FLOAT syntax (2.5) after an optional unary minus.
fn lit_f(x: f64) -> String {
    format!("{x:?}")
}
fn lit_t(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            ' '..='~' => o.push(c),
            _ => o.push_str(&format!("\\u{{{:X}}}", c as u32)),
        }
    }
    o.push('"');
    o
}

// ------------------------------------------------------------ the model of 6.3

enum Out {
    I(i128),
    U(u128),
    B(i128),
    F(f64),
    G(f32),
    T(String),
    C(bool),
    N(usize),
    Trap(String),
}
use Out::*;

fn ovf(v: i128, t: &str) -> Out {
    Trap(format!("overflow: {v} is outside {t}"))
}
fn ovf_u(v: u128) -> Out {
    Trap(format!("overflow: {v} is outside u64"))
}
/// N2: the exact result, checked against the static type's range
fn i64r(v: i128) -> Out {
    if v < i64::MIN as i128 || v > i64::MAX as i128 { ovf(v, "i64") } else { I(v) }
}
fn u64r(v: i128) -> Out {
    if v < 0 || v > u64::MAX as i128 { ovf(v, "u64") } else { U(v as u128) }
}
fn u8r(v: i128) -> Out {
    if !(0..=255).contains(&v) { ovf(v, "u8") } else { B(v) }
}
/// F4: a result that is not finite traps
fn fin(x: f64) -> Out {
    if x.is_finite() { F(x) } else { Trap("non-finite float".into()) }
}
/// N3 f32: `x as f32` rounds once; a value outside the binary32 range
/// traps (6.3 says this of the f32 operations; that the conversion traps
/// too is the reference evaluator's reading, which this model follows)
fn to_g(x: f64) -> Result<f32, Out> {
    let g = x as f32;
    if g.is_finite() { Ok(g) } else { Err(Trap("non-finite float".into())) }
}
fn gfin(g: f32) -> Out {
    if g.is_finite() { G(g) } else { Trap("non-finite float".into()) }
}
/// N7: the 25 White_Space code points of Unicode 18.0.0
fn ws(c: char) -> bool {
    matches!(c, '\u{9}'..='\u{d}' | '\u{20}' | '\u{85}' | '\u{a0}' | '\u{1680}' | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}
/// The observation display of a text (10.5): quoted, with `\t \r \n \\ \"
/// \0` and every other code point below U+0020, U+007F to U+009F and
/// U+0300 to U+036F written `\u{h}`; it reads no Unicode table (N7).
fn display_text(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        let k = c as u32;
        match c {
            '\t' => o.push_str("\\t"),
            '\r' => o.push_str("\\r"),
            '\n' => o.push_str("\\n"),
            '\\' => o.push_str("\\\\"),
            '"' => o.push_str("\\\""),
            '\0' => o.push_str("\\0"),
            _ if k < 0x20 || (0x7f..=0x9f).contains(&k) || (0x300..=0x36f).contains(&k) => {
                o.push_str(&format!("\\u{{{k:x}}}"))
            }
            _ => o.push(c),
        }
    }
    o.push('"');
    o
}
/// Float text (6.3 N3): shortest round-trip digits; decimal notation for
/// 1e-5 <= |x| < 1e16, else `d.ddde±N` with `+` on positive exponents;
/// `-0.0` prints `-0`.
fn float_text(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let a = x.abs();
    let e = format!("{a:e}");
    let (m, ex) = e.split_once('e').unwrap();
    let ex: i32 = ex.parse().unwrap();
    let digits: String = m.chars().filter(|c| *c != '.').collect();
    if (1e-5..1e16).contains(&a) {
        let p = ex + 1; // digits before the point
        let body = if p <= 0 {
            format!("0.{}{digits}", "0".repeat((-p) as usize))
        } else if p as usize >= digits.len() {
            format!("{digits}{}", "0".repeat(p as usize - digits.len()))
        } else {
            format!("{}.{}", &digits[..p as usize], &digits[p as usize..])
        };
        return format!("{sign}{body}");
    }
    let m = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits };
    if ex >= 0 { format!("{sign}{m}e+{ex}") } else { format!("{sign}{m}e{ex}") }
}
/// The exact decimal value of an integral f64 (N2 names the exact result).
fn exact_int(t: f64) -> String {
    let bits = t.abs().to_bits();
    let (e, m) = ((bits >> 52) as i32, bits & ((1 << 52) - 1));
    let (mut mant, shift) = if e == 0 { (m, -1074) } else { (m | (1 << 52), e - 1075) };
    let mut limbs: Vec<u64> = vec![]; // base 1e9, little end first
    if shift < 0 {
        mant >>= -shift; // integral, so no bit is lost
    }
    let mut v = mant;
    while v > 0 {
        limbs.push(v % 1_000_000_000);
        v /= 1_000_000_000;
    }
    for _ in 0..shift.max(0) {
        let mut carry = 0;
        for l in limbs.iter_mut() {
            let x = *l * 2 + carry;
            *l = x % 1_000_000_000;
            carry = x / 1_000_000_000;
        }
        if carry > 0 {
            limbs.push(carry);
        }
    }
    let mut s = limbs.last().map(|l| l.to_string()).unwrap_or("0".into());
    for l in limbs.iter().rev().skip(1) {
        s.push_str(&format!("{l:09}"));
    }
    if t < 0.0 { format!("-{s}") } else { s }
}
/// `toint`: White_Space stripped (N7), an optional sign, ASCII digits. A
/// value outside i64 traps with N2's text; 6.3 gives no text for a text
/// that is not an integer, and the reference's is `not an integer: ` and
/// the displayed text, with the integer read at i128 (a wider value is
/// not an integer).
fn toint(s: &str) -> Out {
    let t = s.trim_matches(ws);
    let bad = || Trap(format!("not an integer: {}", display_text(s)));
    let (neg, d) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if d.is_empty() || !d.bytes().all(|b| b.is_ascii_digit()) {
        return bad();
    }
    let mut v: i128 = 0;
    for b in d.bytes() {
        let x = (b - b'0') as i128;
        let n = v.checked_mul(10).and_then(|v| if neg { v.checked_sub(x) } else { v.checked_add(x) });
        match n {
            Some(n) => v = n,
            None => return bad(),
        }
    }
    i64r(v)
}
/// N8: an end above n, then reversed bounds, then a split character
fn slice(s: &str, p: usize, q: usize) -> Out {
    if p > s.len() || q > s.len() {
        Trap("slice index out of range".into())
    } else if p > q {
        Trap("slice bounds reversed".into())
    } else if !s.is_char_boundary(p) || !s.is_char_boundary(q) {
        Trap("not a char boundary".into())
    } else {
        T(s[p..q].to_string())
    }
}

/// One seeded event: its RCE text and the outcome 6.3 gives.
fn seeded(r: &mut SplitMix) -> (String, Out, usize) {
    const OPS: [&str; 40] = [
        "Add", "Sub", "Mul", "Div", "Rem", "Neg", "Shl", "Shr", "And", "Xor", "Not", "Lt", "UAdd",
        "USub", "UMul", "UDiv", "UShr", "BAdd", "BShl", "FAdd", "FSub", "FMul", "FDiv", "Sqrt",
        "Floor", "Round", "FMin", "FMax", "FLt", "FEq", "ToI", "ToF", "FText", "GMul", "ToG",
        "BLen", "Slice", "TLt", "Strip", "ToInt",
    ];
    let op = *r.pick(&OPS);
    let ii = |r: &mut SplitMix| (gen_i64(r), gen_i64(r));
    let ev = |args: &[(&str, String)]| {
        format!("{op}({})", args.iter().map(|(n, v)| format!("{n}: {v}")).collect::<Vec<_>>().join(", "))
    };
    let (text, out, values) = match op {
        "Add" | "Sub" | "Mul" | "Div" | "Rem" | "And" | "Xor" | "Lt" => {
            let (a, b) = ii(r);
            let (x, y) = (a as i128, b as i128);
            let out = match op {
                "Add" => i64r(x + y),
                "Sub" => i64r(x - y),
                "Mul" => i64r(x * y),
                "Div" | "Rem" if b == 0 => Trap("division by zero".into()),
                "Div" => i64r(x / y),
                // i64.MIN % -1: the quotient is out of range
                "Rem" if a == i64::MIN && b == -1 => ovf(1 << 63, "i64"),
                "Rem" => I(x % y),
                "And" => I((a & b) as i128),
                "Xor" => I((a ^ b) as i128),
                _ => C(a < b),
            };
            (ev(&[("a", lit_i(x)), ("b", lit_i(y))]), out, 2)
        }
        "Neg" | "Not" => {
            let a = gen_i64(r);
            let out = if op == "Neg" { i64r(-(a as i128)) } else { I(!a as i128) };
            (ev(&[("a", lit_i(a as i128))]), out, 1)
        }
        "Shl" | "Shr" => {
            let a = gen_i64(r);
            let k = if r.below(2) == 0 { *r.pick(&SHIFTS) } else { r.below(70) as i64 };
            // `<<` acts on the two's-complement representation (6.3 states
            // the trap on the amount only, and no overflow trap for `<<`;
            // the reference discards the bits shifted out)
            let out = if !(0..64).contains(&k) {
                Trap("shift amount out of range".into())
            } else if op == "Shl" {
                I((a << k) as i128)
            } else {
                I((a >> k) as i128)
            };
            (ev(&[("a", lit_i(a as i128)), ("k", lit_i(k as i128))]), out, 2)
        }
        "UAdd" | "USub" | "UMul" | "UDiv" => {
            let (a, b) = (gen_u64(r), gen_u64(r));
            let out = match op {
                "UAdd" => u64r(a as i128 + b as i128),
                "USub" => u64r(a as i128 - b as i128),
                "UMul" => {
                    let p = a as u128 * b as u128;
                    if p > u64::MAX as u128 { ovf_u(p) } else { U(p) }
                }
                _ if b == 0 => Trap("division by zero".into()),
                _ => U((a / b) as u128),
            };
            (ev(&[("a", a.to_string()), ("b", b.to_string())]), out, 2)
        }
        "UShr" => {
            let a = gen_u64(r);
            let k = r.below(70);
            let out = if k >= 64 { Trap("shift amount out of range".into()) } else { U((a >> k) as u128) };
            (ev(&[("a", a.to_string()), ("k", k.to_string())]), out, 2)
        }
        "BAdd" => {
            let (a, b) = (*r.pick(&U8_EDGES), *r.pick(&U8_EDGES));
            (ev(&[("a", a.to_string()), ("b", b.to_string())]), u8r(a as i128 + b as i128), 2)
        }
        "BShl" => {
            let a = if r.below(2) == 0 { *r.pick(&U8_EDGES) } else { r.below(256) as u8 };
            let k = r.below(11) as u8;
            let out = if k >= 8 { Trap("shift amount out of range".into()) } else { B(((a as u16) << k) as u8 as i128) };
            (ev(&[("a", a.to_string()), ("k", k.to_string())]), out, 2)
        }
        "FAdd" | "FSub" | "FMul" | "FDiv" | "FMin" | "FMax" | "FLt" | "FEq" | "GMul" => {
            let (x, y) = (gen_f64(r), gen_f64(r));
            let out = match op {
                "FAdd" => fin(x + y),
                "FSub" => fin(x - y),
                "FMul" => fin(x * y),
                "FDiv" => fin(x / y),
                // N5: IEEE order, and the receiver on a tie
                "FMin" => F(if y < x { y } else { x }),
                "FMax" => F(if y > x { y } else { x }),
                "FLt" => C(x < y),
                "FEq" => C(x == y),
                _ => match (to_g(x), to_g(y)) {
                    (Ok(a), Ok(b)) => gfin(a * b),
                    (Err(t), _) | (_, Err(t)) => t,
                },
            };
            (ev(&[("x", lit_f(x)), ("y", lit_f(y))]), out, 2)
        }
        "Sqrt" | "Floor" | "Round" | "ToI" | "FText" | "ToG" => {
            let x = gen_f64(r);
            let out = match op {
                "Sqrt" => fin(x.sqrt()),
                "Floor" => F(x.floor()),
                "Round" => F(x.round()), // ties away from zero (F8)
                "ToI" => {
                    let t = x.trunc();
                    if t < -9223372036854775808.0 || t >= 9223372036854775808.0 {
                        Trap(format!("overflow: {} is outside i64", exact_int(t)))
                    } else {
                        I(t as i64 as i128)
                    }
                }
                "FText" => T(float_text(x)),
                _ => match to_g(x) {
                    Ok(g) => G(g),
                    Err(t) => t,
                },
            };
            (ev(&[("x", lit_f(x))]), out, 1)
        }
        "ToF" => {
            let a = gen_i64(r);
            (ev(&[("a", lit_i(a as i128))]), F(a as f64), 1)
        }
        "BLen" | "Strip" | "ToInt" => {
            let s = gen_text(r);
            let out = match op {
                "BLen" => N(s.len()),
                "Strip" => T(s.trim_matches(ws).to_string()),
                _ => toint(&s),
            };
            (ev(&[("s", lit_t(&s))]), out, 1)
        }
        "Slice" => {
            let s = gen_text(r);
            let n = s.len() as u64;
            let (p, q) = (r.below(n + 3) as usize, r.below(n + 3) as usize);
            (ev(&[("s", lit_t(&s)), ("p", p.to_string()), ("q", q.to_string())]), slice(&s, p, q), 3)
        }
        _ => {
            let (s, t) = (gen_text(r), gen_text(r));
            let out = C(s < t); // N7: by code point
            (ev(&[("s", lit_t(&s)), ("r", lit_t(&t))]), out, 2)
        }
    };
    (text, out, values)
}

fn outcome(o: &Out) -> String {
    match o {
        I(v) => format!("iv == {}", lit_i(*v)),
        U(v) => format!("uv == {v}"),
        B(v) => format!("bv == {v}"),
        F(x) => format!("fv == {} and fv.tostr() == {}", lit_f(*x), lit_t(&float_text(*x))),
        G(g) => format!("gv as f64 == {}", lit_f(*g as f64)),
        T(s) => format!("tv == {}", lit_t(s)),
        C(b) => if *b { "cv".into() } else { "not cv".into() },
        N(k) => format!("nv == {k}"),
        Trap(m) => format!("refused: trap({})", lit_t(m)),
    }
}

// ------------------------------------------------------------ the module

/// Write one file of the run record when `RCE_EMIT_EDGES_OUT` is set.
fn record(name: &str, body: &str) {
    if let Some(d) = std::env::var_os("RCE_EMIT_EDGES_OUT") {
        let d = PathBuf::from(d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(name), body).unwrap();
    }
}
fn record_json(name: &str, v: serde_json::Value) {
    record(name, &(serde_json::to_string_pretty(&v).unwrap() + "\n"));
}
fn sha(s: &str) -> String {
    rce_spec_check::interp::d_sha(s)
}

/// The seeded examples, one line each, as the module carries them.
fn seeded_lines() -> String {
    let mut r = SplitMix(SEED);
    (0..SEEDED)
        .map(|_| {
            let (ev, out, _) = seeded(&mut r);
            format!("example {ev} -> {}\n", outcome(&out))
        })
        .collect()
}

struct Reference {
    src: String,
    module: rce_spec_check::ast::Module,
    want: Vec<String>,
    heads: Vec<String>,
}

fn reference() -> &'static Reference {
    static R: OnceLock<Reference> = OnceLock::new();
    R.get_or_init(|| {
        let base = std::fs::read_to_string("tests/emit_edges.rce").unwrap();
        let mut src = base.clone();
        src.push_str(&format!("# ---- {SEEDED} seeded examples, SplitMix64 seed {SEED:#x} (tests/emit_edges.rs)\n"));
        src.push_str(&seeded_lines());
        src.push_str(&format!("# ---- display sweep: every {SWEEP}th scalar value\n"));
        let sweep: String = (0..=0x10FFFFu32).step_by(SWEEP as usize).filter_map(char::from_u32).collect();
        src.push_str(&format!("example BLen(s: {}) -> nv == {}\n", lit_t(&sweep), sweep.len()));
        let module = rce_spec_check::program(&src).unwrap();
        let want = observe(&module);
        let form: serde_json::Value = serde_json::from_str(&rce_spec_check::interp::form_json(&module)).unwrap();
        let heads = form["logs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| {
                l["D"].as_array().and_then(|d| d.last()).map(|e| e["hash"].as_str().unwrap().to_string()).unwrap_or("0".repeat(64))
            })
            .collect();
        Reference { src, module, want, heads }
    })
}

#[test]
fn module_is_admitted_with_every_edge_and_seeded_outcome() {
    let r = reference();
    // 1. the reference evaluator meets every stated outcome (W-INT-1)
    assert_eq!(rce_spec_check::check_one(&r.src), Ok(()), "the module as written from 6.3 is not admitted");
    let base = std::fs::read_to_string("tests/emit_edges.rce").unwrap();
    let events = base.lines().filter(|l| l.starts_with("event ")).count();
    let edges = base.lines().filter(|l| l.starts_with("example ")).count();
    let mut g = SplitMix(SEED);
    let values: usize = (0..SEEDED).map(|_| seeded(&mut g).2).sum();
    let traps = r.src.lines().filter(|l| l.starts_with("example ") && l.contains("-> refused: trap(")).count();
    assert!(events >= 11, "{events} operators");
    assert!(values >= 500, "{values} seeded values");
    assert_eq!(r.heads.len(), edges + SEEDED + 1);
    let digest = sha(&seeded_lines());
    assert_eq!(digest, SEEDED_SHA256, "the seeded examples for seed {SEED:#x} changed");
    // the record: one row per example, then the counts and digests
    let src_lines: Vec<&str> = r.src.lines().collect();
    let mut rows = String::new();
    let mut k = 0;
    let mut i = 0;
    while i < r.want.len() {
        let line: usize = r.want[i].strip_prefix("example ").and_then(|n| n.parse().ok()).expect("every observation belongs to an example");
        let mut j = i + 1;
        while j < r.want.len() && !r.want[j].starts_with("example ") {
            j += 1;
        }
        rows.push_str(&serde_json::json!({"line": line, "example": src_lines[line - 1], "observations": &r.want[i..j], "head": r.heads[k]}).to_string());
        rows.push('\n');
        (i, k) = (j, k + 1);
    }
    assert_eq!(k, r.heads.len(), "one log per example");
    let observations = r.want.join("\n") + "\n";
    record("module.rce", &r.src);
    record("examples.jsonl", &rows);
    record_json(
        "reference.json",
        serde_json::json!({
            "module": "spec/checker/tests/emit_edges.rce",
            "module_sha256": sha(&base),
            "module_with_seeded_sha256": sha(&r.src),
            "seed": format!("{SEED:#x}"),
            "seeded_examples": SEEDED,
            "seeded_values": values,
            "seeded_sha256": digest,
            "display_sweep_step": SWEEP,
            "operators": events,
            "edge_examples": edges,
            "examples": r.heads.len(),
            "examples_ending_in_a_trap": traps,
            "observation_lines": r.want.len(),
            "observation_bytes": observations.len(),
            "observation_sha256": sha(&observations),
            "heads_sha256": sha(&(r.heads.join("\n") + "\n")),
            "examples_jsonl_sha256": sha(&rows),
        }),
    );
    eprintln!(
        "emit_edges: {events} operators, {edges} edge examples, {SEEDED} seeded examples with {values} seeded values (seed {SEED:#x}, sha256 {digest}), 1 display sweep, {traps} examples end in a trap, {} observation lines",
        r.want.len()
    );
}

// ------------------------------------------------------------ targets

fn tool(var: &str, def: &str) -> String {
    std::env::var(var).ok().filter(|v| !v.is_empty()).unwrap_or(def.to_string())
}
fn version(t: &str, arg: &str) -> Option<String> {
    let o = Command::new(t).arg(arg).output().ok()?;
    o.status.success().then(|| {
        let s = String::from_utf8_lossy(&o.stdout).trim().to_string() + String::from_utf8_lossy(&o.stderr).trim();
        s.lines().next().unwrap_or("").to_string()
    })
}
fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-emit-edges-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn out(c: &mut Command) -> Result<Vec<u8>, String> {
    let o = c.output().map_err(|e| e.to_string())?;
    if !o.status.success() {
        return Err(format!("exit {:?}: {}", o.status.code(), String::from_utf8_lossy(&o.stderr)));
    }
    Ok(o.stdout)
}

/// How a target runs one emitted file: Ok(a runner taking extra args).
type Runner = Box<dyn Fn(&[&str]) -> Result<Vec<u8>, String>>;

/// Compare a target's output with the reference: line for line, byte for
/// byte, and the D heads under `--trace`. Panics on any difference.
fn agree(name: &str, tool: &str, run: Runner) {
    let r = reference();
    let got = run(&[]).unwrap_or_else(|e| panic!("{name}: did not run: {}", &e[..e.len().min(3000)]));
    let want_bytes: Vec<u8> = r.want.iter().flat_map(|l| l.bytes().chain(std::iter::once(b'\n'))).collect();
    let text = String::from_utf8(got.clone()).unwrap_or_else(|_| panic!("{name}: output is not UTF-8"));
    let lines: Vec<&str> = text.lines().collect();
    let mut diffs = vec![];
    for i in 0..r.want.len().max(lines.len()) {
        let (w, g) = (r.want.get(i).map(|s| s.as_str()), lines.get(i).copied());
        if w != g {
            diffs.push(format!("line {}: want {w:?}\n         got  {g:?}", i + 1));
        }
    }
    assert!(
        diffs.is_empty(),
        "{name}: {} of {} observation lines differ from the reference:\n{}",
        diffs.len(),
        r.want.len(),
        diffs.iter().take(30).cloned().collect::<Vec<_>>().join("\n")
    );
    assert!(got == want_bytes, "{name}: lines agree, but the output bytes differ ({} bytes, want {})", got.len(), want_bytes.len());
    let traced = run(&["--trace"]).unwrap_or_else(|e| panic!("{name} --trace: did not run: {e}"));
    let heads: Vec<String> = String::from_utf8(traced).unwrap().lines().filter_map(|l| l.strip_prefix("trace ").map(String::from)).collect();
    let bad: Vec<String> = r
        .heads
        .iter()
        .zip(&heads)
        .enumerate()
        .filter(|(_, (w, g))| w != g)
        .map(|(i, (w, g))| format!("log {}: want {w}, got {g}", i + 1))
        .collect();
    assert!(heads.len() == r.heads.len() && bad.is_empty(), "{name}: {} logs, want {}; {} D heads differ:\n{}", heads.len(), r.heads.len(), bad.len(), bad.iter().take(20).cloned().collect::<Vec<_>>().join("\n"));
    eprintln!(
        "{name}: agrees with the reference: {} observation lines equal line for line, {} bytes equal byte for byte, {} D heads equal",
        r.want.len(),
        got.len(),
        heads.len()
    );
    record_json(
        &format!("target-{name}.json"),
        serde_json::json!({
            "target": name,
            "outcome": "agrees",
            "tool": tool,
            "line_for_line": true,
            "byte_for_byte": true,
            "d_heads": true,
            "observation_lines": lines.len(),
            "stdout_bytes": got.len(),
            "stdout_sha256": sha(&text),
            "heads": heads.len(),
            "heads_sha256": sha(&(heads.join("\n") + "\n")),
        }),
    );
}

/// A target that cannot run here: printed and recorded, never a pass.
fn skip(name: &str, why: &str) {
    eprintln!("SKIPPED {name}: {why} (not a pass)");
    record_json(&format!("target-{name}.json"), serde_json::json!({"target": name, "outcome": "skipped", "reason": why}));
}

fn write(dir: &Path, name: &str, code: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, code).unwrap();
    p
}

#[test]
fn rust_target_agrees() {
    let rustc = tool("RCE_RUSTC", "rustc");
    let Some(v) = version(&rustc, "--version") else {
        skip("rust", &format!("{rustc} not found"));
        return;
    };
    eprintln!("rust oracle: {v}");
    let dir = scratch("rust");
    let src = write(&dir, "edges.rs", &emit(&reference().module, Lang::Rust).unwrap());
    let exe = dir.join("edges");
    out(Command::new(&rustc).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&src)).unwrap_or_else(|e| panic!("rustc: {e}"));
    agree("rust", &v, Box::new(move |a| out(Command::new(&exe).args(a))));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn python_target_agrees() {
    let py = tool("RCE_PYTHON", "python3");
    let Some(v) = version(&py, "--version") else {
        skip("python", &format!("{py} not found"));
        return;
    };
    eprintln!("python oracle: {v}");
    let dir = scratch("python");
    let src = write(&dir, "edges.py", &emit(&reference().module, Lang::Python).unwrap());
    agree("python", &v, Box::new(move |a| out(Command::new(&py).arg(&src).args(a))));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn js_target_agrees() {
    let node = tool("RCE_NODE", "node");
    let Some(v) = version(&node, "--version") else {
        skip("js", &format!("{node} not found"));
        return;
    };
    eprintln!("node oracle: {v}");
    let dir = scratch("js");
    let src = write(&dir, "edges.mjs", &emit(&reference().module, Lang::Js).unwrap());
    agree("js", &v, Box::new(move |a| out(Command::new(&node).arg(&src).args(a))));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn scala_target_agrees() {
    let scala = tool("RCE_SCALA", "scala-cli");
    let Some(v) = version(&scala, "version") else {
        skip("scala", &format!("{scala} not found"));
        return;
    };
    let Some(java) = scala_java() else {
        skip("scala", "no Java 19 or later for the Scala jar");
        return;
    };
    eprintln!("scala oracle: {v}");
    let dir = scratch("scala");
    let src = write(&dir, "edges.scala", &emit(&reference().module, Lang::Scala).unwrap());
    // package once into a jar (as tests/fuel.rs does), then run it per argument list
    let jar = dir.join("edges.jar");
    out(Command::new(&scala)
        .args(["--power", "package", "-q", "--server=false", "--scala", "3.10.0", "--scalac-option", "-nowarn", "--assembly", "-f", "-o"])
        .arg(&jar)
        .arg(&src))
    .unwrap_or_else(|e| panic!("scala-cli package: {e}"));
    agree("scala", &format!("{v}; {}", java.display()), Box::new(move |a| out(Command::new(&java).arg("-jar").arg(&jar).args(a))));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Rust output compiled for wasm32-wasip1 and run under a WASI runtime:
/// wasmtime when installed, else Node's `node:wasi`.
#[test]
fn rust_wasm32_wasip1_target_agrees() {
    let rustc = tool("RCE_RUSTC", "rustc");
    if version(&rustc, "--version").is_none() {
        skip("wasm32-wasip1", &format!("{rustc} not found"));
        return;
    }
    let sysroot = Command::new(&rustc).args(["--print", "sysroot"]).output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default();
    if !Path::new(&sysroot).join("lib/rustlib/wasm32-wasip1").exists() {
        skip("wasm32-wasip1", &format!("the target's standard library is not installed in {sysroot} (rustup target add wasm32-wasip1)"));
        return;
    }
    let wasmtime = tool("RCE_WASMTIME", "wasmtime");
    let node = tool("RCE_NODE", "node");
    let runtime = if let Some(v) = version(&wasmtime, "--version") {
        format!("wasmtime ({v})")
    } else if Command::new(&node).args(["-e", "require('node:wasi')"]).output().is_ok_and(|o| o.status.success()) {
        format!("node:wasi ({})", version(&node, "--version").unwrap_or_default())
    } else {
        skip("wasm32-wasip1", &format!("no WASI runtime (neither {wasmtime} nor {node} with node:wasi)"));
        return;
    };
    eprintln!("wasm32-wasip1 runtime: {runtime}");
    let dir = scratch("wasi");
    let src = write(&dir, "edges.rs", &emit(&reference().module, Lang::Rust).unwrap());
    let wasm = dir.join("edges.wasm");
    out(Command::new(&rustc).args(["--edition", "2024", "-O", "--target", "wasm32-wasip1", "-o"]).arg(&wasm).arg(&src)).unwrap_or_else(|e| panic!("rustc --target wasm32-wasip1: {e}"));
    let run: Runner = if runtime.starts_with("wasmtime") {
        Box::new(move |a| out(Command::new(&wasmtime).arg("run").arg(&wasm).args(a)))
    } else {
        let host = write(
            &dir,
            "wasi.mjs",
            "import { readFileSync } from 'node:fs';\nimport { WASI } from 'node:wasi';\nconst [file, ...args] = process.argv.slice(2);\nconst wasi = new WASI({ version: 'preview1', args: [file, ...args], env: {}, returnOnExit: true });\nconst m = await WebAssembly.compile(readFileSync(file));\nconst i = await WebAssembly.instantiate(m, wasi.getImportObject());\nprocess.exitCode = wasi.start(i);\n",
        );
        // Node 22.22.0 ends with SIGSEGV on a WASI call made after the wasm
        // memory has grown by tens of MB (a Rust program that builds a
        // 1,000,000-element Vec and then prints is enough) while V8 calls
        // the WASI functions through its fast API path. The flag turns that
        // path off; it changes no instruction of the program.
        Box::new(move |a| out(Command::new(&node).args(["--no-warnings", "--no-turbo-fast-api-calls"]).arg(&host).arg(&wasm).args(a)))
    };
    agree("wasm32-wasip1", &format!("{}; {runtime}", version(&rustc, "--version").unwrap_or_default()), run);
    let _ = std::fs::remove_dir_all(&dir);
}
