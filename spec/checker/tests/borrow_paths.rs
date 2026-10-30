//! Borrow paths (spec 5.6): two borrows conflict when their paths overlap,
//! that is one is a prefix of the other (`f.a` and `f.b` do not, `f` and
//! `f.a` do, `x[i]` and `x[j]` always do), and a `let` borrow is live until
//! the last use of its name in its block, so a conflicting use before then
//! requires change under W-OWN-2 (Lean `RCE.Borrow`, theorem T6).

fn verdict(src: &str) -> Result<(), String> {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    d[0].1.clone()
}

fn requires_own2(src: &str, has: &str) {
    let e = verdict(src).expect_err("requires change");
    assert!(e.contains("W-OWN-2") && e.contains(has), "{e}");
}

const FOO: &str = "struct Foo
    a: i64
    b: i64

fn bump(x: &mut i64)
    x = x + 1

fn two(x: &mut i64, y: &mut i64)
    x = x + 1
    y = y + 1

fn whole(f: &mut Foo)
    f.b = f.b + 1

fn see(f: &Foo) -> i64 = f.a
";

fn with(body: &str) -> String {
    format!("{FOO}\nfn main()\n{body}")
}

#[test]
fn disjoint_fields_in_one_call_comply() {
    assert_eq!(verdict(&with("    var f: Foo = Foo(a: 0, b: 0)\n    two(&mut f.a, &mut f.b)\n    emit f.a\n")), Ok(()));
}

#[test]
fn overlapping_paths_in_one_call_require_change() {
    requires_own2(&with("    var f: Foo = Foo(a: 0, b: 0)\n    two(&mut f.a, &mut f.a)\n    emit f.a\n"), "`f.a` is borrowed mutably and again in one call");
    let src = "fn two(x: &mut i64, g: &mut List[i64])\n    x = x + 1\n\nfn main()\n    var xs: List[i64] = [1, 2]\n    two(&mut xs[0], &mut xs)\n    emit xs\n";
    requires_own2(src, "overlap");
    let src = "fn two(x: &mut i64, y: &mut i64)\n    x = x + y\n\nfn main()\n    var xs: List[i64] = [1, 2]\n    two(&mut xs[0], &mut xs[1])\n    emit xs\n";
    requires_own2(src, "`xs[_]`");
    // a shared and a mutable borrow of overlapping paths also conflict
    let src = "struct Foo\n    a: i64\n    b: i64\n\nfn g(x: &mut i64, f: &Foo)\n    x = x + f.b\n\nfn main()\n    var f: Foo = Foo(a: 0, b: 0)\n    g(&mut f.a, &f)\n    emit f.a\n";
    requires_own2(src, "overlap");
}

#[test]
fn a_write_while_a_let_borrow_is_live_requires_change() {
    requires_own2(&with("    var n: i64 = 0\n    let a = &mut n\n    n = 5\n    bump(a)\n    emit n\n"), "[No-Alias-Write]");
    requires_own2(&with("    var f: Foo = Foo(a: 0, b: 0)\n    let a = &mut f.a\n    whole(&mut f)\n    bump(a)\n    emit f.a\n"), "[Borrow-Mut]");
    requires_own2(&with("    var f: Foo = Foo(a: 0, b: 0)\n    let a = &mut f.a\n    emit see(&f)\n    bump(a)\n"), "[Borrow-Shr]");
    requires_own2(&with("    var f: Foo = Foo(a: 0, b: 0)\n    let a = &mut f\n    emit f.b\n    whole(a)\n"), "[No-Mut-Read]");
    requires_own2(&with("    var n: i64 = 1\n    let a = &n\n    n = 2\n    emit a + 0\n"), "[No-Alias-Write]");
    // a use inside a nested block or a loop keeps the borrow live
    requires_own2(&with("    var n: i64 = 0\n    let a = &mut n\n    n = 5\n    for _k in 0..2\n        bump(a)\n    emit n\n"), "[No-Alias-Write]");
    // a mutating method of the root
    requires_own2("fn main()\n    var xs: List[i64] = [1]\n    let a = &xs\n    xs.append(2)\n    emit a.len()\n", "[No-Alias-Write]");
    // a move of the root under a shared borrow
    requires_own2("fn count(xs: &List[i64]) -> usize = xs.len()\n\nfn main()\n    var xs: List[i64] = [1, 2]\n    let a = &xs\n    let ys = xs\n    emit count(a)\n    emit ys.len()\n", "moved");
}

#[test]
fn uses_after_the_last_use_or_of_other_paths_comply() {
    // after its last use the borrow has ended
    assert_eq!(verdict(&with("    var n: i64 = 0\n    let a = &mut n\n    bump(a)\n    n = 5\n    emit n\n")), Ok(()));
    // a disjoint field while a field is borrowed
    assert_eq!(verdict(&with("    var f: Foo = Foo(a: 0, b: 0)\n    let a = &mut f.a\n    bump(&mut f.b)\n    f.b = f.b + 1\n    bump(a)\n    emit f.a\n")), Ok(()));
    // reads under a shared borrow
    assert_eq!(verdict(&with("    var f: Foo = Foo(a: 0, b: 0)\n    let a = &f\n    emit f.a + see(&f)\n    emit see(a)\n")), Ok(()));
    // a borrow made again in each iteration
    assert_eq!(verdict(&with("    var n: i64 = 0\n    for _k in 0..3\n        let q = &mut n\n        bump(q)\n        n = n + 1\n    emit n\n")), Ok(()));
}
