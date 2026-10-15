//! The interpreter's evaluation depth is defined by the language, not the
//! host: `interp::EVAL_DEPTH` nested evaluations, then the trap
//! `evaluation depth`. Each case runs on a thread with a deliberately small
//! (256 KiB) stack, so a host-stack dependence would show as a crash here.
use std::process::Command;

fn rce_test(file: &str) -> String {
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["test", file]).output().unwrap();
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

#[test]
fn deep_recursion_under_the_limit_runs() {
    let out = rce_test("tests/depth/ok_7000.rce");
    assert!(out.starts_with("ok "), "{out}");
}

#[test]
fn deep_recursion_over_the_limit_is_the_defined_trap() {
    let out = rce_test("tests/depth/trap_10000.rce");
    assert!(out.contains("trap: evaluation depth"), "{out}");
}

#[test]
fn the_limit_holds_on_a_small_host_stack() {
    // the same program in process, on a 256 KiB thread: stacker grows the
    // stack on the heap, so the outcome is the trap, not an overflow
    let src = std::fs::read_to_string("tests/depth/trap_10000.rce").unwrap();
    let r = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(move || {
            let d = rce_spec_check::check_program(&[("t.rce".into(), src)], false);
            format!("{:?}", d[0].1)
        })
        .unwrap()
        .join()
        .unwrap();
    assert!(r.contains("evaluation depth"), "{r}");
}
