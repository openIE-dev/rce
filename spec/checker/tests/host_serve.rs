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
        .args(["host", "serve"])
        .arg(&app)
        .arg("--log")
        .arg(&log)
        .arg("--pty")
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
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(&app).arg("--log").arg(&log).output().unwrap();
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
        .args(["host", "serve"])
        .arg(&app)
        .arg("--log")
        .arg(&log)
        .arg("--pty")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut i = c.stdin.take().unwrap();
    let pause = |ms| std::thread::sleep(std::time::Duration::from_millis(ms));
    pause(500);
    writeln!(i, "Resize(rows: 40, cols: 100)").unwrap();
    pause(200);
    writeln!(i, "echo size=$(stty size)").unwrap();
    pause(300);
    writeln!(i, "sleep 30 && echo not-interrupted").unwrap();
    pause(300);
    writeln!(i, "Interrupt").unwrap();
    pause(300);
    writeln!(i, "echo after-$((6*7))").unwrap();
    pause(300);
    writeln!(i, "exit").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while c.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline, "the terminal app did not exit");
        pause(50);
    }
    drop(i);
    let out = String::from_utf8_lossy(&c.wait_with_output().unwrap().stdout).into_owned();
    assert!(out.contains("size=40 100"), "{out}");
    assert!(out.contains("after-42") && !out.contains("\nnot-interrupted"), "{out}");
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(&app).arg("--log").arg(&log).output().unwrap();
    let r = String::from_utf8_lossy(&r.stdout).into_owned();
    assert!(r.starts_with("replay ok") && r.contains("cols = 100"), "{r}");
    let _ = std::fs::remove_file(&log);
}
