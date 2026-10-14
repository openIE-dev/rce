//! A type alias is the type it names (5.1): a parameter whose type is an
//! alias of a channel type makes its function a task function (5.2), so it
//! may send and be spawned; the same function with a non-channel alias is
//! pure and `spawn` of it is W-SPAWN-1.

fn verdict(src: &str) -> Result<(), String> {
    let d = rce_spec_check::check_program(&[("t.rce".into(), src.to_string())], false);
    d[0].1.clone()
}

const TASK: &str = "type Pipe = chan (u64, text)

fn feed(p: Pipe, n: u64)
    for i in 0..n
        p <- (i, \"job\")
    p.close()

fn main()
    let p: Pipe = chan[(u64, text)](2)
    spawn feed(p, 2)
";

#[test]
fn channel_alias_parameter_is_a_task_function() {
    assert_eq!(verdict(TASK), Ok(()));
    let nested = TASK.replace("type Pipe = chan (u64, text)", "type Inner = chan (u64, text)\ntype Pipe = Inner");
    assert_eq!(verdict(&nested), Ok(()));
}

#[test]
fn non_channel_alias_stays_pure() {
    let src = "type Num = u64

fn f(x: Num) -> u64
    return x

fn main()
    spawn f(1)
";
    let e = verdict(src).unwrap_err();
    assert!(e.contains("W-SPAWN-1"), "{e}");
}
