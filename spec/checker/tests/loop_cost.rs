//! `loop bound N` is the one spelling of an unconditional loop (`while true` is
//! refused, W-UNUSED-2). Its cost is the body's alone: no condition is
//! evaluated per iteration. The budget boundary is the same operation in the
//! reference evaluator and in every emitted runtime.
use rce_spec_check::cli;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

fn prog(b: i128) -> String {
    format!("fn f() -> i64\n    var i = 0\n    loop bound 8\n        i += 1\n        if i == 5\n            break\n    return i\n\nfn main()\n    with budget {b}\n        emit f()\n")
}

fn trapped(out: &str) -> bool {
    out.contains("budget")
}

fn reference(src: &str, stem: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-loopcost-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.rce"));
    std::fs::write(&f, src).unwrap();
    let (mut out, mut err) = (vec![], vec![]);
    cli::main(&["run".into(), f.to_string_lossy().into_owned()], &mut &b""[..], &mut out, &mut err);
    String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap()
}

fn emitted(src: &str, stem: &str, target: &str) -> Option<String> {
    let d = std::env::temp_dir().join(format!("rce-loopcost-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("{stem}.rce"));
    std::fs::write(&f, src).unwrap();
    let (mut out, mut err) = (vec![], vec![]);
    let a = vec!["emit".to_string(), target.to_string(), f.to_string_lossy().into_owned()];
    assert_eq!(cli::main(&a, &mut &b""[..], &mut out, &mut err), 0, "{}", String::from_utf8_lossy(&err));
    let (ext, tool) = match target {
        "python" => ("py", "python3"),
        "js" => ("mjs", "node"),
        _ => ("rs", "rustc"),
    };
    if !targets::covers(target, tool) {
        return None;
    }
    let p = d.join(format!("{stem}.{ext}"));
    std::fs::write(&p, out).unwrap();
    let o = if target == "rust" {
        let exe = d.join(stem);
        let c = Command::new("rustc").args(["--edition", "2024", "-o"]).arg(&exe).arg(&p).output().unwrap();
        assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
        Command::new(&exe).arg("--run").output().unwrap()
    } else {
        Command::new(tool).arg(&p).arg("--run").output().unwrap()
    };
    Some(String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr))
}

#[test]
fn the_loop_boundary_is_found_and_is_the_same_in_every_runtime() {
    // the smallest budget that does not trap, by the reference
    let b = (0..200).find(|b| !trapped(&reference(&prog(*b), &format!("r{b}")))).expect("some budget runs it");
    assert!(b > 0, "a loop of five iterations is not free");
    assert!(trapped(&reference(&prog(b - 1), "below")));
    for (budget, trap) in [(b - 1, true), (b, false)] {
        for t in ["rust", "python", "js"] {
            if let Some(out) = emitted(&prog(budget), &format!("e{budget}_{t}"), t) {
                assert_eq!(trapped(&out), trap, "{t}: budget {budget} (boundary {b}): {out}");
            }
        }
    }
}

#[test]
fn while_true_is_refused() {
    let src = "fn main()\n    var i = 0\n    while true bound 8\n        i += 1\n        if i == 5\n            break\n    emit i\n";
    assert!(rce_spec_check::check_one(src).unwrap_err().contains("W-UNUSED-2"));
}
