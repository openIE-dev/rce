//! Cross-module references (spec 4.1): `use p.q [as r]` makes the `pub`
//! items of module `p.q` reachable as `r.x`; private items and unqualified
//! references require change. Checks and examples run on the linked program,
//! so a `check` that calls into another module is evaluated.
use rce_spec_check::check_program;

fn go(files: &[(&str, &str)]) -> Vec<Result<(), String>> {
    let fs: Vec<(String, String)> = files.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
    check_program(&fs, false).into_iter().map(|x| x.1).collect()
}

const PRICING: &str = "module shop.pricing

fn clamp(p: i64) -> i64 = if p > 100 then 100 else p

pub fn discount(total: i64, percent: i64) -> i64 = total - total * clamp(percent) / 100

pub struct Money
    cents: i64

pub fn double(m: Money) -> Money = Money(cents: m.cents * 2)

pub kind Tier
    Gold
    Basic

pub const VAT: i64 = 20
";

#[test]
fn pub_items_are_reachable_through_the_import() {
    let orders = "module shop.orders
use shop.pricing

pub fn final_price(total: i64) -> i64 = pricing.discount(total, 10) + pricing.VAT

fn cash() -> pricing.Money
    let m: pricing.Money = pricing.Money(cents: 7)
    pricing.double(m)

fn tier_name(t: pricing.Tier) -> text
    match t
        pricing.Gold -> \"gold\"
        pricing.Basic -> \"basic\"

check final_price(200) == 200
check cash().cents == 14
check pricing.discount(50, 500) == 0
";
    let r = go(&[("pricing.rce", PRICING), ("orders.rce", orders)]);
    assert!(r.iter().all(|x| x.is_ok()), "{r:?}");
    // the check is evaluated on the linked program: a wrong one requires change
    let bad = orders.replace("check final_price(200) == 200", "check final_price(200) == 199");
    let r = go(&[("pricing.rce", PRICING), ("orders.rce", &bad)]);
    assert!(r[1].as_ref().is_err_and(|e| e.contains("check")), "{r:?}");
}

#[test]
fn alias_and_transitive_imports() {
    let base = "module util.base\n\npub fn inc(x: i64) -> i64 = x + 1\n";
    let mid = "module util.mid\nuse util.base as b\n\npub fn inc2(x: i64) -> i64 = b.inc(b.inc(x))\n";
    let top = "module app\nuse util.mid as m\n\ncheck m.inc2(40) == 42\n";
    let r = go(&[("base.rce", base), ("mid.rce", mid), ("top.rce", top)]);
    assert!(r.iter().all(|x| x.is_ok()), "{r:?}");
}

#[test]
fn private_unqualified_unknown_and_clashing_names_require_change() {
    let cases = [
        ("module a\nuse shop.pricing\n\nfn f() -> i64 = pricing.clamp(3)\n", "W-VIS-1"),
        ("module a\nuse shop.pricing\n\nfn f() -> i64 = discount(10, 1)\n", "W-NAME-1"),
        ("module a\nuse shop.pricing\n\nfn f() -> i64 = pricing.nothing(1)\n", "W-NAME-1"),
        ("module a\nuse shop.pricing\n\nfn pricing(x: i64) -> i64 = x\n", "W-MOD-3"),
        ("module a\nuse shop.pricing\nuse other.pricing\n\nfn f() -> i64 = 1\n", "W-MOD-3"),
    ];
    for (src, rule) in cases {
        let r = go(&[("pricing.rce", PRICING), ("a.rce", src), ("other.rce", OTHER)]);
        assert!(r[0].is_ok(), "{r:?}");
        assert!(r[1].as_ref().is_err_and(|e| e.contains(rule)), "{rule}: {:?}\n{src}", r[1]);
    }
}

const OTHER: &str = "module other.pricing\n\npub fn discount(x: i64) -> i64 = x + 1000\n";

/// Items are reached only through their import (4.1): the importer, and
/// two imported modules, may each declare `discount` and `clamp`, `pub` or
/// not, and every reference reaches its own module's item (W-MOD-3).
#[test]
fn same_names_in_two_modules_and_the_importer() {
    let a = "module a
use shop.pricing
use other.pricing as o

fn clamp(p: i64) -> i64 = p + 1
fn discount(x: i64) -> i64 = clamp(x) * 2

check discount(1) == 4
check pricing.discount(200, 500) == 0
check pricing.discount(200, 10) == 180
check o.discount(1) == 1001
";
    let r = go(&[("pricing.rce", PRICING), ("other.rce", OTHER), ("a.rce", a)]);
    assert!(r.iter().all(|x| x.is_ok()), "{r:?}");
    // the checks run on the linked program: each name reaches its own item
    for bad in ["check discount(1) == 1001", "check o.discount(1) == 4", "check pricing.discount(200, 500) == 400"] {
        let src = a.replace("check discount(1) == 4", bad);
        let r = go(&[("pricing.rce", PRICING), ("other.rce", OTHER), ("a.rce", &src)]);
        assert!(r[2].as_ref().is_err_and(|e| e.contains("check")), "{bad}: {r:?}");
    }
    // a private item of an imported module stays private
    let src = a.replace("check discount(1) == 4", "check pricing.clamp(1) == 1");
    let r = go(&[("pricing.rce", PRICING), ("other.rce", OTHER), ("a.rce", &src)]);
    assert!(r[2].as_ref().is_err_and(|e| e.contains("W-VIS-1")), "{r:?}");
}

/// Every pair of standard-library modules can be imported into one module,
/// each under its own import name. A pair whose default import names are
/// equal (`use os.clock` and `use time.clock` would both be `clock`) needs
/// `as` (W-MOD-3); that pair is listed, and admitted with `as`.
#[test]
fn every_pair_of_stdlib_modules_imports_together() {
    let names = rce_spec_check::stdlib::names();
    let mut fail = vec![];
    let mut same_name = vec![];
    for (i, a) in names.iter().enumerate() {
        for b in &names[i + 1..] {
            let src = format!("module pair\nuse {a} as m1\nuse {b} as m2\n\nfn f() -> i64 = 1\n");
            // the module reads neither import, so W-UNUSED-1 on the two
            // `use` lines is expected; any other diagnostic is a failure
            if let Err(e) = rce_spec_check::check_one(&src) {
                let rest: Vec<&str> = e.split("; ").filter(|d| !d.contains("W-UNUSED-1 `use ")).collect();
                if !rest.is_empty() {
                    fail.push(format!("{a} + {b}: {}", rest.join("; ")));
                }
            }
            if a.rsplit('.').next() == b.rsplit('.').next() {
                same_name.push(format!("{a} + {b}"));
                let src = format!("module pair\nuse {a}\nuse {b}\n\nfn f() -> i64 = 1\n");
                let r = rce_spec_check::check_one(&src);
                assert!(r.as_ref().is_err_and(|e| e.contains("W-MOD-3")), "{a} + {b}: {r:?}");
            }
        }
    }
    let n = names.len();
    eprintln!("{} modules, {} pairs; {} pairs share an import name and need `as`: {}", n, n * (n - 1) / 2, same_name.len(), same_name.join(", "));
    assert!(fail.is_empty(), "{} of {} pairs require change:\n{}", fail.len(), n * (n - 1) / 2, fail.join("\n"));
}

#[test]
fn spec_example_26_still_checks() {
    let a = include_str!("../../examples/26_use_module/shop/orders.rce");
    let b = include_str!("../../examples/26_use_module/shop/pricing.rce");
    let r = go(&[("26.rce", a), ("shop_pricing.rce", b)]);
    assert!(r.iter().all(|x| x.is_ok()), "{r:?}");
}

/// Example 26 is a package directory: `rce check DIR` checks it alone.
#[test]
fn spec_example_26_checks_alone_as_a_package() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/26_use_module");
    let (mut out, mut err) = (vec![], vec![]);
    let code = rce_spec_check::cli::main(&["check".into(), dir.into()], &mut &b""[..], &mut out, &mut err);
    let out = String::from_utf8(out).unwrap();
    assert_eq!(code, 0, "{out}{}", String::from_utf8_lossy(&err));
    assert_eq!(out.lines().filter(|l| l.starts_with("ok ")).count(), 2, "{out}");
}
