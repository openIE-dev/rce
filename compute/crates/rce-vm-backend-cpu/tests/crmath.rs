//! `crmath` against two oracles, bit for bit:
//!
//! - the MPFR-checked vectors of RCE's standard library
//!   (`stdlib/tests/math/math_checks_{exp,tanh,erf,sin,cos}_*.rce`, written by
//!   `scripts/gen/math_vec.py`, the worst cases of CORE-MATH among them),
//!   the same lines `math.rce` is checked on;
//! - `tests/crmath_oracle.txt`, a fixed sample with edge cases computed in
//!   exact decimal arithmetic by `tests/crmath_oracle.py`.
use rce_vm_backend_cpu::crmath;
use std::path::Path;

fn f(name: &str) -> fn(f64) -> f64 {
    match name {
        "exp" => crmath::exp,
        "tanh" => crmath::tanh,
        "erf" => crmath::erf,
        "sin" => crmath::sin,
        "cos" => crmath::cos,
        _ => panic!("{name}"),
    }
}

/// A number of a vector line: `X` or `(0.0 - X)`, evaluated as RCE does.
fn number(s: &str) -> f64 {
    let s = s.trim();
    match s.strip_prefix("(0.0 - ").and_then(|r| r.strip_suffix(')')) {
        Some(x) => 0.0 - x.parse::<f64>().unwrap(),
        None => s.parse::<f64>().unwrap(),
    }
}

#[test]
fn equals_the_mpfr_vectors_of_the_standard_library() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../stdlib/tests/math");
    let mut seen = std::collections::BTreeMap::new();
    let mut bad = vec![];
    for entry in std::fs::read_dir(&dir).unwrap() {
        let p = entry.unwrap().path();
        let file = p.file_name().unwrap().to_string_lossy().into_owned();
        let Some(name) = ["exp", "tanh", "erf", "sin", "cos"].into_iter().find(|n| file.starts_with(&format!("math_checks_{n}_"))) else {
            continue;
        };
        for line in std::fs::read_to_string(&p).unwrap().lines() {
            let Some(rest) = line.strip_prefix(&format!("check {name}(")) else { continue };
            let (arg, res) = rest.split_once(") == ").unwrap();
            let (x, want) = (number(arg), number(res));
            let got = f(name)(x);
            if got.to_bits() != want.to_bits() {
                bad.push(format!("{name}({x:e}) = {got:e}, MPFR {want:e}"));
            }
            *seen.entry(name).or_insert(0) += 1;
        }
    }
    assert!(bad.is_empty(), "{} differ: {:?}", bad.len(), &bad[..bad.len().min(10)]);
    assert!(seen["exp"] > 400 && seen["tanh"] > 2000 && seen["erf"] > 10000 && seen["sin"] > 7000 && seen["cos"] > 7000, "{seen:?}");
}

#[test]
fn equals_the_decimal_oracle_on_its_fixed_sample() {
    let text = include_str!("crmath_oracle.txt");
    let mut n = 0;
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let w: Vec<&str> = line.split(' ').collect();
        let x = f64::from_bits(u64::from_str_radix(w[1], 16).unwrap());
        let want = u64::from_str_radix(w[2], 16).unwrap();
        let got = f(w[0])(x);
        assert_eq!(got.to_bits(), want, "{}({x:e}) = {got:e}, oracle {:e}", w[0], f64::from_bits(want));
        n += 1;
    }
    assert_eq!(n, 1951);
}

#[test]
fn edge_cases() {
    // signed zeros, subnormals, the ends of each range, NaN and infinities
    assert_eq!(crmath::exp(-0.0), 1.0);
    assert_eq!(crmath::exp(709.782712893384), 1.7976931348622732e308);
    assert_eq!(crmath::exp(709.7827128933841), f64::INFINITY);
    assert_eq!(crmath::exp(-745.1332191019411), 5e-324);
    assert_eq!(crmath::exp(-745.1332191019412).to_bits(), 0);
    assert!(crmath::exp(f64::NAN).is_nan());
    assert_eq!(crmath::exp(f64::INFINITY), f64::INFINITY);
    assert_eq!(crmath::exp(f64::NEG_INFINITY), 0.0);
    for z in [0.0f64, -0.0] {
        assert_eq!(crmath::tanh(z).to_bits(), z.to_bits());
        assert_eq!(crmath::erf(z).to_bits(), z.to_bits());
    }
    assert_eq!(crmath::tanh(-5e-324), -5e-324);
    assert_eq!(crmath::tanh(1e300), 1.0);
    assert_eq!(crmath::tanh(f64::NEG_INFINITY), -1.0);
    assert!(crmath::tanh(f64::NAN).is_nan());
    assert_eq!(crmath::erf(-1e300), -1.0);
    assert_eq!(crmath::erf(f64::INFINITY), 1.0);
    assert!(crmath::erf(f64::NAN).is_nan());
    // erf of the least subnormal: 2/sqrt(pi) x rounds to 5e-324
    assert_eq!(crmath::erf(5e-324), 5e-324);
}
