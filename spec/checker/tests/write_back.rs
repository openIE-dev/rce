//! Spec 6.2 `&mut` write-back with the F.4.4 restore: a callee that fails
//! leaves the caller's `&mut` arguments as they were before the call, and a
//! callee that returns writes them back. The program emitters implement it for a
//! write-back call alone, under `catch` and under `try` (T9-L); deeper nesting is
//! first flattened by the shared order-preserving ANF pass (T34). All
//! agree with the reference evaluator on every target.
use rce_spec_check::cli;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

fn rce(args: &[String]) -> (i32, String, String) {
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(args, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

const PROG: &str = "error Boom

fn bump(x: &mut i64) -> i64 ! Boom
    x += 1
    fail Boom

fn wrap(n: &mut i64) -> i64 ! Boom
    let _ = try bump(&mut n)
    let k: i64 = 0
    return k

fn main()
    var n: i64 = 10
    let v = wrap(&mut n) catch -1
    emit n
    emit v
";

fn file(stem: &str, src: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-wb-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.rce"));
    std::fs::write(&f, src).unwrap();
    f.to_string_lossy().into_owned()
}

/// stdout of the emitted program, or None when this run does not cover the target
fn run_target(f: &str, t: &str) -> Option<String> {
    let (code, out, err) = rce(&["emit".into(), t.into(), f.into()]);
    assert_eq!(code, 0, "{t}: {err}");
    let (ext, tool) = match t {
        "python" => ("py", "python3"),
        "js" => ("mjs", "node"),
        _ => ("rs", "rustc"),
    };
    if !targets::covers(t, tool) {
        return None;
    }
    let p = format!("{}_{t}.{ext}", f.trim_end_matches(".rce"));
    std::fs::write(&p, out).unwrap();
    let o = if t == "rust" {
        let exe = format!("{f}.bin");
        let c = Command::new("rustc").args(["--edition", "2024", "-o", &exe, &p]).output().unwrap();
        assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
        Command::new(&exe).arg("--run").output().unwrap()
    } else {
        Command::new(tool).arg(&p).arg("--run").output().unwrap()
    };
    assert!(o.status.success(), "{t}: {}", String::from_utf8_lossy(&o.stderr));
    let s = String::from_utf8(o.stdout).unwrap();
    Some(
        s.lines()
            .filter_map(|l| if l.starts_with("emit ") { Some(l.to_string()) } else { l.strip_prefix("outcome Stopped: ").map(|r| format!("stopped: {r}")) })
            .map(|l| format!("{l}\n"))
            .collect(),
    )
}

fn agree(stem: &str, src: &str, want: &str) {
    let f = file(stem, src);
    let (code, out, err) = rce(&["run".into(), f.clone()]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(out, want, "reference");
    for t in ["rust", "python", "js"] {
        if let Some(got) = run_target(&f, t) {
            assert_eq!(got, want, "{t} differs from the reference on {stem}");
        }
    }
}

#[test]
fn a_failed_call_restores_its_mut_arguments_on_every_target() {
    agree("wb", PROG, "emit 10\nemit -1\n");
}

#[test]
fn receivers_several_args_try_and_catch_agree() {
    let src = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/write_back_methods.rce")).unwrap();
    agree(
        "wbm",
        &src,
        "emit 10\nemit 0\nemit 12\nemit 11\nemit 110\nemit 3\nemit 3\nemit 1200\nemit 3\n",
    );
}

const NESTED: &str = "error Boom

fn bump(x: &mut i64) -> i64 ! Boom
    x += 1
    if x > 12
        fail Boom
    return x + 0

fn note(v: i64) -> i64
    return v * 2

struct Acc
    n: i64

impl Acc
    fn add(&mut self, k: i64) -> i64
        self.n += k
        return self.n

fn main()
    var x: i64 = 10
    let a = 1 + (bump(&mut x) catch 0)
    emit a
    let b = x + (bump(&mut x) catch -1) + x
    emit b
    emit [x, bump(&mut x) catch -5, x]
    let c = note(x) - (bump(&mut x) catch 100) * note(x)
    emit c
    var acc = Acc(n: 1)
    emit acc.n + acc.add(5) + acc.n
    var xs: List[i64] = [3, 1]
    emit [xs.len(), xs.pop().or(0) as usize, xs.len()]
    with budget BUDGET
        emit x
        let d = x * 2 + (bump(&mut x) catch 7) + note(x)
        emit d
    emit x
";

/// T34: nested write-back calls (under operators, in list literals and
/// arguments, on receivers) with reads of the written variable before
/// and after: every target matches the reference evaluator.
#[test]
fn nested_write_back_calls_keep_left_to_right_order_on_every_target() {
    agree(
        "wbnest",
        &NESTED.replace("BUDGET", "1000"),
        "emit 12\nemit 35\nemit [12, -5, 12]\nemit -2376\nemit 13\nemit [2, 1, 1]\nemit 12\nemit 55\nemit 12\n",
    );
}

/// T34: budget charges keep their order. Sweeping the budget across the
/// nested statement moves the trap point identically on every target,
/// with emits interleaved before and after it.
#[test]
fn nested_write_back_budget_traps_at_the_same_point_on_every_target() {
    let mut seen = std::collections::BTreeSet::new();
    for b in [0, 2, 4, 6, 7, 8, 10, 14] {
        let src = NESTED.replace("BUDGET", &b.to_string());
        let f = file(&format!("wbbud{b}"), &src);
        let (_, want, err) = rce(&["run".into(), f.clone()]);
        assert!(want.contains("emit "), "{err}");
        seen.insert(want.lines().count());
        for t in ["rust", "python", "js"] {
            if let Some(got) = run_target(&f, t) {
                assert_eq!(got, want, "{t} differs from the reference at budget {b}");
            }
        }
    }
    assert!(seen.len() >= 2, "the sweep should cross the trap point");
}

/// Conditional positions are not flattened (that would evaluate a
/// branch eagerly); a write-back call there still requires change.
#[test]
fn a_write_back_call_in_a_conditional_position_requires_change() {
    let src = PROG.replace("let v = wrap(&mut n) catch -1", "let v = n > 0 and (wrap(&mut n) catch -1) > 0");
    let f = file("wbcond", &src);
    for t in ["rust", "python", "js"] {
        let (code, _, err) = rce(&["emit".into(), t.into(), f.clone()]);
        assert_ne!(code, 0, "{t}");
        assert!(err.contains("writes back"), "{t}: {err}");
    }
}

/// 5.6: a write through a `let` `&mut` borrow is a write to the borrowed
/// place (`writeVia` of the Lean model of T6), on every target: the whole
/// variable, a field, a list element whose index variable changes after the
/// `let`, a method call through the borrow, and a borrow made in a loop body.
#[test]
fn writes_through_a_let_borrow_land_in_the_place_on_every_target() {
    let src = "struct Foo
    a: i64
    b: i64

fn bump(x: &mut i64)
    x = x + 1

fn main()
    var n: i64 = 0
    let a = &mut n
    bump(a)
    a += 5
    emit n
    var f: Foo = Foo(a: 0, b: 0)
    let fa = &mut f.a
    bump(&mut f.b)
    bump(fa)
    emit [f.a, f.b]
    var xs: List[i64] = [1, 2, 3]
    var i: usize = 1
    let e = &mut xs[i]
    i = 2
    e = e * 100
    bump(e)
    emit xs
    emit i
    var ys: List[i64] = []
    let l = &mut ys
    l.append(7)
    emit ys.len()
    var m: i64 = 0
    for _k in 0..3
        let q = &mut m
        bump(q)
    emit m
";
    agree("letmut", src, "emit 6\nemit [1, 1]\nemit [1, 201, 3]\nemit 2\nemit 1\nemit 3\n");
}
