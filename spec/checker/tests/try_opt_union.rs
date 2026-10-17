//! Regression: `try e` where `e : T? ! E` succeeds with the option itself
//! ([E-Try-V], RCE-SPEC 6.x): `some(v)` stays `some(v)` and `none` stays
//! `none`. The comptime evaluator (admission `check` items) and the Rust,
//! Python, JS and Scala runtimes unwrapped `some` and trapped "try on none".
use rce_spec_check::emit::{Lang, emit, observe};
use rce_spec_check::parse::parse;
use std::process::Command;

const SRC: &str = "error Bad(why: text)

kind Val
    VStr(t: text)
    VInt(v: i64)

kind J
    JNull
    JStr(t: text)
    JNum(n: i64)

fn conv(j: &J) -> Val ! Bad
    match j
        JStr(t: t) -> return VStr(t: copy(t))
        JNum(n: n) -> return VInt(v: n)
        _ -> fail Bad(why: \"x\")

fn opt(j: &J) -> Val? ! Bad
    match j
        JNull -> return none
        _ -> return some(try conv(j))

fn find(x: i64) -> i64? ! Bad
    if x < 0
        fail Bad(why: \"neg\")
    if x == 0
        return none
    return some(x * 2)

fn kindof(j: J) -> i64 ! Bad
    let r = try opt(&j)
    match r
        some(v) -> return 1
        none -> return 0

fn twice(x: i64) -> i64 ! Bad
    let r = try find(x)
    match r
        some(v) -> return v
        none -> return 0 - 1000

check (kindof(JNull) catch -1) == 0
check (kindof(JNum(n: 4)) catch -1) == 1
check (kindof(JStr(t: \"a\")) catch -1) == 1
check (twice(3) catch -1) == 6
check (twice(0) catch -1) == -1000
check (twice(-2) catch -1) == -1

fn main()
    emit kindof(JNull) catch -1
    emit kindof(JNum(n: 4)) catch -1
    emit twice(3) catch -1
    emit twice(0) catch -1
    emit twice(-2) catch -1
";

#[test]
fn admitted_with_checks() {
    let d = rce_spec_check::check_program(&[("t.rce".into(), SRC.to_string())], false);
    assert!(d[0].1.is_ok(), "{:?}", d[0].1);
}

#[test]
fn emitted_targets_agree() {
    let m = parse(SRC).unwrap();
    let want = observe(&m);
    assert!(want.iter().any(|l| l.contains("-1000")), "{want:?}");
    let d = std::env::temp_dir().join(format!("rce-tryopt-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    for lang in [Lang::Rust, Lang::Python, Lang::Js, Lang::Scala] {
        let code = emit(&m, lang).unwrap();
        let f = d.join(format!("t.{}", lang.ext()));
        std::fs::write(&f, code).unwrap();
        let out = match lang {
            Lang::Rust => {
                let exe = d.join("t");
                let Ok(c) = Command::new(std::env::var("RCE_RUSTC").unwrap_or("rustc".into())).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&f).output() else { eprintln!("SKIPPED rustc"); continue };
                assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
                Command::new(&exe).output().unwrap()
            }
            Lang::Python => match Command::new(std::env::var("RCE_PYTHON").unwrap_or("python3".into())).arg(&f).output() { Ok(o) => o, Err(_) => { eprintln!("SKIPPED python"); continue } },
            Lang::Js => match Command::new(std::env::var("RCE_NODE").unwrap_or("node".into())).arg(&f).output() { Ok(o) => o, Err(_) => { eprintln!("SKIPPED node"); continue } },
            Lang::Scala => match Command::new(std::env::var("RCE_SCALA").unwrap_or("scala-cli".into())).args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"]).arg(&f).output() { Ok(o) => o, Err(_) => { eprintln!("SKIPPED scala-cli"); continue } },
        };
        let got: Vec<String> = String::from_utf8(out.stdout).unwrap().lines().map(String::from).collect();
        assert_eq!(got, want, "{lang:?}");
    }
}
