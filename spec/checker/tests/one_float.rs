//! The runtime functions of floats written in the core (src/one/rtf.rs),
//! against the reference interpreter's own functions on edge values and on
//! pseudo-random values: the display of f64 (`float_text`), its canonical
//! JSON (`float_json`), the display of f32 (Rust's `Display`, which the
//! reference uses), the exact integer a trap message names
//! (`exact_int_text`), and the rounding of an integer to f32.
use rce_spec_check::interp::{exact_int_text, float_json, float_text};
use rce_spec_check::one::{self, build::*, ir::*, rtf};

/// splitmix64, so that the values are the same on every run.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

fn f64_atom(x: f64) -> Atom {
    Atom::K(Ty::F64, Lit::F64(x.to_bits()))
}

/// Evaluate `g(f(a))` for each argument and return the lines printed;
/// the module is checked (W1 to W6) first.
fn run_each(make: impl FnOnce(&mut Fns) -> usize, then: Option<fn(&mut Fns) -> usize>, conv: Option<&str>, args: &[Atom]) -> Vec<String> {
    let mut f = Fns::default();
    let id = make(&mut f);
    let next = then.map(|g| g(&mut f));
    let entry = f.declare("entry", vec![], Ty::unit());
    let mut b = FB::new(vec![], Ty::unit());
    for a in args {
        let r0 = f.ret(id);
        let mut r = b.call(sym(id), r0, vec![a.clone()]);
        if let Some(p) = conv {
            r = b.p(p, vec![], vec![r]).unwrap();
        }
        if let Some(g) = next {
            r = b.call(sym(g), Ty::Bytes, vec![r]);
        }
        let l = b.cat(r, Atom::bytes(b"\n")).unwrap();
        b.out(l);
    }
    let u = b.unit().unwrap();
    b.ret(u);
    f.define(entry, b.finish());
    let m = f.finish(entry).unwrap();
    one::wf::check(&m).unwrap_or_else(|e| panic!("not well formed: {e}"));
    let r = one::eval::run(&m).unwrap_or_else(|e| panic!("{e}"));
    assert!(r.halt.is_none(), "halted: {:?}", r.halt);
    String::from_utf8(r.out).unwrap().lines().map(String::from).collect()
}

fn f64_values(n: usize) -> Vec<f64> {
    let mut v: Vec<f64> = vec![
        0.0, -0.0, 1.0, -1.0, 0.1, 0.2, 0.3, 0.1 + 0.2, 1e23, 9007199254740993.0, 5e-324, -5e-324, 1e-323,
        2.2250738585072014e-308, 2.225073858507201e-308, 1.7976931348623157e308, -1.7976931348623157e308, 1e16,
        9999999999999998.0, 1e-5, 9.999999999999999e-6, 1e-4, 9.999999999999999e-5, 0.5, 2.5, -2.5, 123456789.125,
        1e21, 1e22, 1e300, 1e-300, 4.35, 0.000123, 100.0, 1234.5, std::f64::consts::PI, std::f64::consts::E,
        f64::EPSILON, 4503599627370496.5, 4503599627370497.0, 0.30000000000000004,
    ];
    for k in -1074..=1023 {
        v.push(2f64.powi(k));
    }
    for k in -323..=308 {
        for d in ["1", "5"] {
            let x: f64 = format!("{d}e{k}").parse().unwrap();
            if x.is_finite() {
                v.push(x);
            }
        }
    }
    let mut r = Rng(1);
    while v.len() < n {
        let x = f64::from_bits(r.next());
        if x.is_finite() {
            v.push(x);
        }
        // short decimals, whose shortest form is short
        let d: f64 = format!("{}e{}", r.next() % 100000, (r.next() % 60) as i64 - 30).parse().unwrap();
        v.push(d);
    }
    v
}

#[test]
fn f64_display_and_json() {
    let xs = f64_values(3000);
    let args: Vec<Atom> = xs.iter().map(|x| f64_atom(*x)).collect();
    let got = run_each(rtf_text, None, None, &args);
    for (x, g) in xs.iter().zip(&got) {
        assert_eq!(*g, float_text(*x), "display of {x:e} ({:#x})", x.to_bits());
    }
    let got = run_each(rtf_json, None, None, &args);
    for (x, g) in xs.iter().zip(&got) {
        assert_eq!(*g, float_json(*x), "JSON of {x:e} ({:#x})", x.to_bits());
    }
}

fn rtf_text(f: &mut Fns) -> usize {
    rtf::text(f).unwrap()
}
fn rtf_json(f: &mut Fns) -> usize {
    rtf::json(f).unwrap()
}
fn rtf_text32(f: &mut Fns) -> usize {
    rtf::text32(f).unwrap()
}

#[test]
fn f32_display() {
    let mut xs: Vec<f32> = vec![0.0, -0.0, 1.0, 0.1, -0.1, 1e-7, 16777216.0, 16777217.0, 1e10, 3.4028235e38, f32::MIN_POSITIVE, 1e-45, 1.1, 0.3, 7.0e-10];
    for k in -149..=127 {
        xs.push(2f32.powi(k));
    }
    for k in -45..=38 {
        if let Ok(x) = format!("1e{k}").parse::<f32>() {
            xs.push(x);
        }
    }
    let mut r = Rng(2);
    while xs.len() < 2500 {
        let x = f32::from_bits(r.next() as u32);
        if x.is_finite() {
            xs.push(x);
        }
    }
    let args: Vec<Atom> = xs.iter().map(|x| f64_atom(*x as f64)).collect();
    let got = run_each(rtf_text32, None, None, &args);
    for (x, g) in xs.iter().zip(&got) {
        assert_eq!(*g, x.to_string(), "display of the f32 {x:e} ({:#x})", x.to_bits());
    }
}

fn rtf_int_text(f: &mut Fns) -> usize {
    rtf::int_text(f).unwrap()
}

#[test]
fn exact_integer_text() {
    let mut xs = vec![0.0, -0.0, -0.5, 0.5, 1e300, -1e300, 1.7976931348623157e308, 9007199254740992.0, -9007199254740993.0, 1.8446744073709552e19, 1.7014118346046923e38, 123.9, -123.9];
    let mut r = Rng(3);
    while xs.len() < 1500 {
        let x = f64::from_bits(r.next());
        if x.is_finite() {
            xs.push(x);
        }
    }
    let args: Vec<Atom> = xs.iter().map(|x| f64_atom(*x)).collect();
    let got = run_each(rtf_int_text, None, None, &args);
    for (x, g) in xs.iter().zip(&got) {
        // interp `trunc_chk`: the i128 when it holds the value, else `exact_int_text`
        let t = x.trunc();
        let want = if t.abs() < 1.7014118346046923e38 { (t as i128).to_string() } else { exact_int_text(t) };
        assert_eq!(*g, want, "trunc of {x:e}");
    }
}

fn rtf_i2f32(f: &mut Fns) -> usize {
    rtf::int_to_f32(f).unwrap()
}

#[test]
fn integers_rounded_once_to_f32() {
    let mut vs: Vec<i128> = vec![0, 1, -1, 16777216, 16777217, 16777219, (1 << 53) + 1, (1 << 54) + 3, i64::MIN as i128, i64::MAX as i128, u64::MAX as i128, (1 << 63) + (1 << 39), (1 << 63) + (1 << 39) + 1];
    let mut r = Rng(4);
    while vs.len() < 1500 {
        let x = r.next();
        let sh = r.next() % 64;
        let v = (x >> sh) as i128;
        vs.push(v);
        if v <= i64::MAX as i128 {
            vs.push(-v);
        }
    }
    let args: Vec<Atom> = vs.iter().map(|v| Atom::K(Ty::Int(*v, *v), Lit::Int(*v))).collect();
    let got = run_each(rtf_i2f32, Some(rtf_text), Some("f32.to_f64"), &args);
    for (v, g) in vs.iter().zip(&got) {
        assert_eq!(*g, float_text(*v as f32 as f64), "{v} as f32");
    }
}
