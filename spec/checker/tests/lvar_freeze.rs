//! LVars with a frozen bit, threshold reads in the program's own Merge
//! order, pairwise incompatible threshold sets, and Merge laws as
//! obligations (C.11.5, C.11.8; model RCE/LVarQD.lean, T29q). Each runnable
//! program is emitted to Rust, Python and JavaScript and compared line for
//! line with the reference evaluator.
#![cfg(feature = "conc06")]
use rce_spec_check::check_program;
use rce_spec_check::emit::{Lang, emit, observe, with_probes};
use rce_spec_check::parse::parse;
use std::process::Command;

fn tool(lang: Lang) -> &'static str {
    match lang {
        Lang::Rust => "rustc",
        Lang::Python => "python3",
        Lang::Js => "node",
        Lang::Scala => "scala-cli",
    }
}

fn run(lang: Lang, tag: &str, code: &str) -> Result<String, String> {
    let dir = std::env::temp_dir().join(format!("rce-emit06b-{tag}-{lang:?}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join(format!("p.{}", lang.ext()));
    std::fs::write(&src, code).unwrap();
    let out = match lang {
        Lang::Rust => {
            let exe = dir.join("p");
            let c = Command::new("rustc")
                .args(["--edition", "2024", "-O", "-o"])
                .arg(&exe)
                .arg(&src)
                .output()
                .map_err(|e| e.to_string())?;
            if !c.status.success() {
                return Err(String::from_utf8_lossy(&c.stderr).to_string());
            }
            Command::new(&exe).output()
        }
        _ => Command::new(tool(lang)).arg(&src).output(),
    }
    .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_dir_all(&dir);
    if !out.status.success() {
        return Err(format!("{}\n{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)));
    }
    Ok(String::from_utf8(out.stdout).unwrap())
}

/// admitted by rce-check, then every available target prints the evaluator's lines
fn agree(tag: &str, src: &str) -> Vec<String> {
    let d = rce_spec_check::check_program(&[("e.rce".into(), src.to_string())], false);
    assert!(d[0].1.is_ok(), "{tag}: {:?}\n{src}", d[0].1);
    let m = with_probes(&parse(src).unwrap());
    let want = observe(&m);
    for lang in [Lang::Rust, Lang::Python, Lang::Js] {
        if Command::new(tool(lang)).arg("--version").output().is_err() {
            eprintln!("SKIPPED {lang:?}: {} not found (not a pass)", tool(lang));
            continue;
        }
        let code = emit(&m, lang).unwrap_or_else(|e| panic!("{tag} {lang:?}: not emitted: {e}"));
        let got = run(lang, tag, &code).unwrap_or_else(|e| panic!("{tag} {lang:?} did not run: {e}"));
        let got: Vec<&str> = got.lines().collect();
        if got != want {
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                if g != w {
                    panic!("{tag} {lang:?}: line {i}: got {g:?}, want {w:?}");
                }
            }
            panic!("{tag} {lang:?}: {} lines, want {}", got.len(), want.len());
        }
    }
    want
}


fn refused(src: &str, rule: &str) -> String {
    let r = check_program(&[("t.rce".into(), src.to_string())], false);
    match &r[0].1 {
        Ok(()) => panic!("accepted, expected {rule}:\n{src}"),
        Err(e) => {
            assert!(e.contains(rule), "expected {rule}, got {e}");
            e.clone()
        }
    }
}

fn admitted(src: &str) {
    let r = check_program(&[("t.rce".into(), src.to_string())], false);
    assert!(r[0].1.is_ok(), "{:?}\n{src}", r[0].1);
}

const LVL: &str = "kind Lvl
    Lo
    Mid
    Hi

fn rank(x: Lvl) -> i64
    match x
        Lo -> 0
        Mid -> 1
        Hi -> 2

impl Merge for Lvl
    fn merge(self, other: Lvl) -> Lvl
        if rank(other) > rank(self)
            return other
        self
    fn bottom() -> Lvl
        Lo

kind IV
    Empty
    A
    B
    Bad

fn code(x: IV) -> i64
    match x
        Empty -> 0
        A -> 1
        B -> 2
        Bad -> 3

impl Merge for IV
    fn merge(self, other: IV) -> IV
        let a = code(self)
        let b = code(other)
        if a == 0
            return other
        if b == 0
            return self
        if a == b
            return self
        Bad
    fn bottom() -> IV
        Empty
";

#[test]
fn frozen_bit_agrees() {
    let src = "fn put_after() -> i64
    let x = lvar_new(0)
    x.put(3)
    let a = x.freeze()
    x.put(5)
    a

fn same_after() -> i64
    let x = lvar_new(0)
    x.put(3)
    let a = x.freeze()
    x.put(3)
    x.put(1)
    a + x.freeze()

fn wait_after() -> i64
    let x = lvar_new(0)
    x.put(4)
    let a = x.freeze()
    a + x.wait([2])

fn par_then_freeze() -> i64
    let x = lvar_new(0)
    par
        branch a
            x.put(3)
        branch b
            x.put(8)
    let f = x.freeze()
    x.put(2)
    f

fn fresh_in_branch() -> i64
    let x = lvar_new(0)
    var r: i64 = 0
    par
        branch a
            x.put(3)
        branch b
            let y = lvar_new(0)
            y.put(6)
            r = y.freeze()
    r + x.freeze()
";
    let l = agree("frozen", src);
    for want in [
        "fn put_after trap: lvar put after freeze",
        "fn same_after = 6",
        "fn wait_after = 6",
        "fn par_then_freeze = 8",
        "fn fresh_in_branch = 9",
    ] {
        assert!(l.iter().any(|x| x == want), "missing {want:?} in {l:?}");
    }
}

#[test]
fn thresholds_use_the_program_merge() {
    let src = format!("{LVL}
fn user_order() -> i64
    let x = lvar_new(Lo)
    x.put(Hi)
    rank(x.wait([Mid]))

fn user_order_par() -> i64
    let x = lvar_new(Lo)
    var r: i64 = 0
    par
        branch w
            x.put(Mid)
            x.put(Hi)
        branch g
            r = rank(x.wait([Hi]))
    r

fn either() -> i64
    let x = lvar_new(Empty, Bad)
    var r: i64 = 0
    par
        branch w
            x.put(B)
        branch g
            r = code(x.wait([A, B]))
    r

fn either_conflict() -> i64
    let x = lvar_new(Empty, Bad)
    x.put(A)
    x.put(B)
    0

fn not_incomp() -> i64
    let x = lvar_new(0, 100)
    let t = 3
    x.put(5)
    x.wait([t, 4])

fn no_top() -> i64
    let x = lvar_new(Empty)
    x.put(A)
    code(x.wait([A, B]))
");
    let l = agree("thresh", &src);
    for want in [
        "fn user_order = 1",
        "fn user_order_par = 2",
        "fn either = 2",
        "fn either_conflict trap: lvar conflict",
        "fn not_incomp trap: lvar thresholds are not pairwise incompatible",
        "fn no_top trap: lvar thresholds are not pairwise incompatible",
    ] {
        assert!(l.iter().any(|x| x == want), "missing {want:?} in {l:?}");
    }
}

#[test]
fn w_lvar_2_decides_literal_sets() {
    refused(
        "fn f() -> i64
    let x = lvar_new(0, 100)
    x.put(5)
    x.wait([5, 3])
",
        "W-LVAR-2",
    );
    let e = refused(
        &format!("{LVL}
fn g() -> i64
    let x = lvar_new(Lo)
    x.put(Hi)
    rank(x.wait([Mid, Hi]))
"),
        "W-LVAR-2",
    );
    assert!(e.contains("compatible"), "{e}");
}

#[test]
fn w_lvar_1_is_closed_under_aliasing() {
    // an alias made before the par
    refused(
        "fn f() -> i64
    let x = lvar_new(0)
    let y = x
    var r: i64 = 0
    par
        branch a
            x.put(3)
        branch b
            r = y.freeze()
    r
",
        "W-LVAR-1",
    );
    // a helper that freezes its parameter
    refused(
        "fn peek(z: lvar i64) -> i64 = z.freeze()

fn f() -> i64
    let x = lvar_new(0)
    var r: i64 = 0
    par
        branch a
            x.put(3)
        branch b
            r = peek(x)
    r
",
        "W-LVAR-1",
    );
    // a helper that puts, two calls deep, and a freeze through a parameter name
    refused(
        "fn add(z: lvar i64, v: i64)
    z.put(v)

fn add2(z: lvar i64)
    add(z, 2)

fn f(x: lvar i64) -> i64
    var r: i64 = 0
    par
        branch a
            add2(x)
        branch b
            r = x.freeze()
    r
",
        "W-LVAR-1",
    );
    // a name rebound from a fresh lvar to an outside one
    refused(
        "fn f() -> i64
    let x = lvar_new(0)
    var r: i64 = 0
    par
        branch a
            x.put(3)
        branch b
            var y = lvar_new(0)
            y = x
            r = y.freeze()
    r
",
        "W-LVAR-1",
    );
    // for ... par: a freeze of the outside lvar in the body
    refused(
        "fn f() -> i64
    let x = lvar_new(0)
    var s: i64 = 0
    for i in 0..4 par
        x.put(i)
        s += x.freeze()
    s
",
        "W-LVAR-1",
    );
    // freeze after the join, and freezes of a branch's own lvar, are admitted
    admitted(
        "fn f() -> i64
    let x = lvar_new(0)
    var r: i64 = 0
    par
        branch a
            x.put(3)
        branch b
            let y = lvar_new(0)
            y.put(1)
            r = y.freeze()
    r + x.freeze()
",
    );
}

#[test]
fn merge_laws_are_obligations() {
    // componentwise from max and or: discharged by MergeLaws.prodSL
    admitted(
        "struct Hw
    hi: i64
    seen: bool

impl Merge for Hw
    fn merge(self, other: Hw) -> Hw
        Hw(hi: self.hi.max(other.hi), seen: self.seen or other.seen)
    fn bottom() -> Hw
        Hw(hi: 0, seen: false)

fn main()
    let x = lvar_new(Hw(hi: 0, seen: false))
    x.put(Hw(hi: 4, seen: false))
    x.put(Hw(hi: 2, seen: true))
    emit x.freeze().hi
",
    );
    // a finite kind: decided by enumeration
    admitted(&format!("{LVL}\nfn main()\n    emit 1\n"));
    // lawful but not in a decidable form: the obligation is not discharged
    let e = refused(
        "struct Hi
    v: i64

impl Merge for Hi
    fn merge(self, other: Hi) -> Hi
        if other.v > self.v
            return other
        self
    fn bottom() -> Hi
        Hi(v: 0)
",
        "W-CRDT-1",
    );
    assert!(e.contains("proof obligation"), "{e}");
    // unlawful: refuted with a counterexample
    let e = refused(
        "kind Ab
    P
    Q

impl Merge for Ab
    fn merge(self, other: Ab) -> Ab
        other
    fn bottom() -> Ab
        P
",
        "W-CRDT-1",
    );
    assert!(e.contains("not "), "{e}");
}
