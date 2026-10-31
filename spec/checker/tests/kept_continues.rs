//! A kept step does not end a sequence of events (6.9, 10.3, 10.9): the
//! step keeps the state (T8), it is printed or logged as kept, and the next
//! event is applied to the unchanged state, as on the host (HC-21). The
//! example `Dec, Inc -> 1` of a counter at 0 holds: `Dec` is kept with
//! `Underflow` and `Inc` applies. Checked in the reference (admission,
//! `rce run`, `rce trace`), in the core evaluator (`rce run-core`) and in
//! every covered emit target.
use rce_spec_check::cli::{run, trace};
use rce_spec_check::emit::{Lang, emit, observe_trace};
use rce_spec_check::parse::parse;
use std::process::Command;

#[path = "common/targets.rs"]
mod targets;

const COUNTER: &str = "intent \"a counter that never goes negative\"
    example Dec, Inc -> 1
    example Inc, Dec, Dec, Inc, Inc -> 2
    example Dec, Inc, Dec, Dec -> kept: Underflow

state Counter
    n: u64 = 0

event Inc
event Dec

error Underflow

step Counter on Inc -> n + 1
step Counter on Dec -> try n.checked_sub(1) else Underflow
";

#[test]
fn examples_with_an_earlier_kept_step_hold() {
    let d = rce_spec_check::check_program(&[("t.rce".into(), COUNTER.to_string())], false);
    assert!(d[0].1.is_ok(), "{:?}", d[0].1);
    // the last step's outcome is still what an example states
    let wrong = COUNTER.replace("example Dec, Inc -> 1", "example Inc, Dec -> kept: Underflow");
    let d = rce_spec_check::check_program(&[("t.rce".into(), wrong)], false);
    let e = d[0].1.clone().unwrap_err();
    assert!(format!("{e:?}").contains("expected kept: Underflow, but Dec was applied"), "{e:?}");
}

#[test]
fn run_applies_the_events_after_a_kept_step() {
    let r = run(COUNTER, Some("Dec, Inc, Dec, Dec")).unwrap();
    assert_eq!(
        r.lines,
        [
            "0 (initial) Counter { n = 0 }",
            "1 Dec kept: Underflow",
            "2 Inc Counter { n = 1 }",
            "3 Dec Counter { n = 0 }",
            "4 Dec kept: Underflow",
        ]
    );
    assert!(r.kept && r.last_kept);
    let r = run(COUNTER, Some("Dec, Inc")).unwrap();
    assert_eq!(r.lines.last().unwrap(), "2 Inc Counter { n = 1 }");
    assert!(r.kept && !r.last_kept, "a run with a kept step is status 1 (10.3)");
}

#[test]
fn trace_has_the_kept_step_and_the_steps_after_it() {
    let t: serde_json::Value = serde_json::from_str(&trace("t.rce", COUNTER, Some("Dec, Inc")).unwrap()).unwrap();
    let st = t["traces"][0]["steps"].as_array().unwrap();
    assert_eq!(st.len(), 2);
    assert_eq!(st[0]["outcome"]["kept"], "Underflow");
    let hash = |evs: &str, i: usize| -> serde_json::Value {
        let t: serde_json::Value = serde_json::from_str(&trace("t.rce", COUNTER, Some(evs)).unwrap()).unwrap();
        t["traces"][0]["steps"][i]["state_hash"].clone()
    };
    assert_eq!(st[0]["state_hash"], hash("Inc, Dec", 1), "a kept step has the unchanged state's hash (T8)");
    assert_eq!(st[1]["outcome"], "applied");
    assert_eq!(st[1]["prev_hash"], st[0]["hash"]);
    assert_eq!(st[1]["state_hash"], hash("Inc", 0), "the next event is applied to the unchanged state");
    // the traces of the module's examples go on past a kept step too
    let t: serde_json::Value = serde_json::from_str(&trace("t.rce", COUNTER, None).unwrap()).unwrap();
    let lens: Vec<usize> = t["traces"].as_array().unwrap().iter().map(|x| x["steps"].as_array().unwrap().len()).collect();
    assert_eq!(lens, [2, 5, 4]);
}

#[test]
fn core_evaluator_applies_the_events_after_a_kept_step() {
    let m = rce_spec_check::program(COUNTER).unwrap();
    let want = rce_spec_check::one::trace::reference(&m);
    assert!(want.contains("1 Dec kept: Underflow\n2 Inc Counter { n = 1 }\n"), "{want}");
    assert_eq!(rce_spec_check::cli::run_core(COUNTER).unwrap(), want);
}

#[test]
fn emitted_targets_apply_the_events_after_a_kept_step() {
    let m = parse(COUNTER).unwrap();
    let want = observe_trace(&m);
    assert!(want.iter().any(|l| l == "2 Inc Counter { n = 1 }"), "{want:?}");
    let d = std::env::temp_dir().join(format!("rce-kept-continues-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let args = ["--trace", "--fuel-trace"];
    for lang in targets::covered(&targets::ALL, targets::oracle) {
        let code = emit(&m, lang).unwrap_or_else(|e| panic!("{lang:?}: {e}"));
        let f = d.join(format!("counter.{}", lang.ext()));
        std::fs::write(&f, code).unwrap();
        let out = match lang {
            Lang::Rust => {
                let exe = d.join("counter");
                let c = Command::new(targets::oracle(Lang::Rust)).args(["--edition", "2024", "-O", "-o"]).arg(&exe).arg(&f).output();
                let c = c.unwrap_or_else(|e| panic!("rustc does not run: {e}"));
                assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
                Command::new(&exe).args(args).output().unwrap()
            }
            Lang::Python => Command::new(targets::oracle(Lang::Python)).arg(&f).args(args).output().expect("python does not run"),
            Lang::Js => Command::new(targets::oracle(Lang::Js)).arg(&f).args(args).output().expect("node does not run"),
            Lang::Scala => Command::new(targets::oracle(Lang::Scala))
                .args(["run", "-q", "--server=false", "--jvm", "21", "--scala", "3.10.0", "--scalac-option", "-nowarn"])
                .arg(&f)
                .arg("--")
                .args(args)
                .output()
                .expect("scala-cli does not run"),
        };
        let got: Vec<String> = String::from_utf8(out.stdout).unwrap().lines().map(String::from).collect();
        assert_eq!(got, want, "{lang:?}");
    }
    let _ = std::fs::remove_dir_all(&d);
}
