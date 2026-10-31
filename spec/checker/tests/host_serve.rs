//! W1d: the host loop (`rce host serve`): stdin lines become inputs (event
//! syntax, else `Input(data)`), `Tick` is admitted every frame, `Display`
//! goes to stdout and `Exit` ends the loop. The terminal app v0
//! (crates/rce/apps/terminal.rce) runs a real shell on a PTY this way, and
//! its log replays with every adapter disconnected.
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

#[cfg(unix)]
#[test]
fn terminal_app_runs_a_shell_and_replays() {
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/terminal.rce");
    let log = std::env::temp_dir().join(format!("rce-serve-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve", "pty"])
        .arg(&app)
        .arg(&log)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut i = c.stdin.take().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    writeln!(i, "echo rce-$((40+2))").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(500));
    writeln!(i, "exit").unwrap();
    // the app emits Exit when the shell closes the terminal; stdin stays open
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if c.try_wait().unwrap().is_some() {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "the terminal app did not exit");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(i);
    let out = String::from_utf8_lossy(&c.wait_with_output().unwrap().stdout).into_owned();
    assert!(out.contains("rce-42"), "{out}");
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(&app).arg(&log).output().unwrap();
    let r = String::from_utf8_lossy(&r.stdout).into_owned();
    assert!(r.starts_with("replay ok") && r.contains("closed = true"), "{r}");
    let _ = std::fs::remove_file(&log);
}

/// Terminal v1: `Resize` reaches the PTY (the shell sees the new size) and
/// `Interrupt` sends ^C to the foreground job; the session still replays.
#[cfg(unix)]
#[test]
fn terminal_resize_and_interrupt() {
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/terminal.rce");
    let log = std::env::temp_dir().join(format!("rce-serve-v1-{}.log", std::process::id()));
    let _ = std::fs::remove_file(&log);
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve", "pty"])
        .arg(&app)
        .arg(&log)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut i = c.stdin.take().unwrap();
    // Each step waits for the output of the previous one instead of sleeping a
    // fixed time: a fixed pause lets `Interrupt` arrive before the shell has
    // started the job it is meant to interrupt, and the later lines then queue
    // behind `sleep 30`. The markers are computed by the shell (`$((1+1))`), so
    // the echo of the command line itself never matches them.
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<u8>::new()));
    let reader = {
        let (seen, mut out) = (seen.clone(), c.stdout.take().unwrap());
        std::thread::spawn(move || {
            use std::io::Read;
            let mut b = [0u8; 4096];
            while let Ok(n) = out.read(&mut b) {
                if n == 0 {
                    break;
                }
                seen.lock().unwrap().extend_from_slice(&b[..n]);
            }
        })
    };
    let wait_for = |marker: &str| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !String::from_utf8_lossy(&seen.lock().unwrap()).contains(marker) {
            assert!(std::time::Instant::now() < deadline, "no `{marker}` in the output: {}", String::from_utf8_lossy(&seen.lock().unwrap()));
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    };
    writeln!(i, "echo ready-$((1+1))").unwrap();
    wait_for("ready-2");
    writeln!(i, "Resize(rows: 40, cols: 100)").unwrap();
    writeln!(i, "echo size=$(stty size)").unwrap();
    wait_for("size=40 100");
    // `Interrupt` is ^C to the *foreground* job. The shell hands the terminal to a
    // job a moment after forking it, so a ^C sent before that goes to the shell
    // itself, which ignores it, and the job then runs on (seen: the job still
    // sleeping, in the foreground, 20 s later, under CPU load). The job therefore
    // prints its marker only once its process group is the terminal's foreground
    // group (`ps -o tpgid` equals `ps -o pgid`).
    writeln!(i, "sh -c 'until [ \"$(ps -o tpgid= -p $$)\" -eq \"$(ps -o pgid= -p $$)\" ]; do :; done; echo go-$((1+1)); exec sleep 30' && echo not-interrupted").unwrap();
    wait_for("go-2");
    writeln!(i, "Interrupt").unwrap();
    writeln!(i, "echo after-$((6*7))").unwrap();
    wait_for("after-42");
    writeln!(i, "exit").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while c.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "the terminal app did not exit");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(i);
    reader.join().unwrap();
    let out = String::from_utf8_lossy(&seen.lock().unwrap()).into_owned();
    assert!(out.contains("size=40 100"), "{out}");
    assert!(out.contains("after-42") && !out.contains("\nnot-interrupted"), "{out}");
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(&app).arg(&log).output().unwrap();
    let r = String::from_utf8_lossy(&r.stdout).into_owned();
    assert!(r.starts_with("replay ok") && r.contains("cols = 100"), "{r}");
    let _ = std::fs::remove_file(&log);
}
