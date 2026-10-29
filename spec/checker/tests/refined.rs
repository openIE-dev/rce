//! 5.1 refined types (W-REF-1; CORE-PROPERTIES.md P2; Lean RCE/Refine.lean).
//! A refined value is checked where its base type's range is checked: the
//! reference evaluator and the emitted Rust, Python and JavaScript runtimes
//! accept the same values and trap with the same text on the same value.
use rce_spec_check::cli;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

const PROG: &str = "type Pct = u8 where self <= 100
type Pin = text where self.isdigit() and self.len() == 4
type Tag = List[i64] where self.len() >= 1 and not (self.len() > 3)
fn ok(p: Pct) -> Pct
    p
fn main()
    let a: Pct = 100
    emit a
    let b: Pin = \"0042\"
    emit b
    let t: Tag = [1, 2]
    emit t.len()
    emit ok(7)
    let n: u8 = 101
    let c: Pct = n
    emit c
";

fn dir() -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rce-refined-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let c = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (c, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

const WANT: [&str; 5] = ["100", "0042", "2", "7", "refinement: value is outside Pct"];

fn agree(s: &str) {
    for w in WANT {
        assert!(s.contains(w), "missing {w:?} in {s}");
    }
}

#[test]
fn reference_accepts_inside_and_traps_outside() {
    let f = dir().join("r.rce");
    std::fs::write(&f, PROG).unwrap();
    let (c, o, e) = run(&["run", f.to_str().unwrap()]);
    assert_eq!(c, 1, "{o}{e}");
    agree(&o);
}

#[test]
fn every_emitted_runtime_agrees() {
    let d = dir();
    let f = d.join("e.rce");
    std::fs::write(&f, PROG).unwrap();
    for (target, ext, tool) in [("rust", "rs", "rustc"), ("python", "py", "python3"), ("js", "mjs", "node")] {
        if !targets::covers(target, tool) {
            continue;
        }
        let (c, out, err) = run(&["emit", target, f.to_str().unwrap()]);
        assert_eq!(c, 0, "{err}");
        let p = d.join(format!("r_{target}.{ext}"));
        std::fs::write(&p, out).unwrap();
        let o = if target == "rust" {
            let exe = d.join("r_rust");
            let c = Command::new("rustc").args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&p).output().unwrap();
            assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
            Command::new(&exe).arg("--run").output().unwrap()
        } else {
            Command::new(tool).arg(&p).arg("--run").output().unwrap()
        };
        let s = format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr));
        agree(&s);
    }
}

#[test]
fn w_ref_1_requires_change_outside_the_fragment() {
    for bad in [
        "type B = f64 where self < 1.0\n",
        "type B = i64 where self * 2 < 1\n",
        "type B = text where self < 3\n",
        "type B = i64 where self.len() < 3\n",
        "type B = text where self.contains(\"a\")\n",
        "type B[T] = List[T] where self.len() < 3\n",
    ] {
        let f = dir().join("bad.rce");
        std::fs::write(&f, format!("{bad}fn main()\n    emit 1\n")).unwrap();
        let (c, o, e) = run(&["check", f.to_str().unwrap()]);
        assert_ne!(c, 0, "{bad}");
        assert!(format!("{o}{e}").contains("W-REF-1"), "{bad}: {o}{e}");
    }
}

#[test]
fn a_state_write_outside_the_refinement_keeps_the_state() {
    let f = dir().join("s.rce");
    std::fs::write(
        &f,
        "type Pct = u8 where self <= 100\nstate V\n    level: Pct = 0\nevent Up(by: u8)\nstep V on Up(by: b) -> level + b\n",
    )
    .unwrap();
    let (c, o, _) = run(&["run", f.to_str().unwrap(), "Up(by: 60), Up(by: 60)"]);
    assert_eq!(c, 1);
    assert!(o.contains("1 Up(by: 60) V { level = 60 }"), "{o}");
    assert!(o.contains("kept: trap(refinement: value is outside Pct)"), "{o}");
}
