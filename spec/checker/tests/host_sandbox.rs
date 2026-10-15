//! W1h / HC-5: `rce host run` enters the sandbox's OS enforcement target after opening D;
//! the granted file root and the log still work, and the run's D replays.
use std::path::Path;
use std::process::Command;

#[test]
fn host_run_enters_sandbox_and_granted_authority_still_works() {
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/files.rce");
    let dir = std::env::temp_dir().join(format!("rce-sbx-{}", std::process::id()));
    let root = dir.join("root");
    std::fs::create_dir_all(&root).unwrap();
    let log = dir.join("d.log");
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "run"]).arg(&app).arg("--log").arg(&log).arg("--fs").arg(&root).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    let out = String::from_utf8_lossy(&o.stdout);
    if cfg!(target_os = "macos") {
        assert!(err.contains("sandbox, macOS target (Seatbelt)"), "{err}");
    } else if cfg!(target_os = "linux") {
        assert!(err.contains("sandbox, Linux target"), "{err}");
    } else {
        assert!(err.contains("sandbox OS target not entered"), "{err}");
    }
    assert!(out.contains("durable"), "{out}\n{err}");
    // P5: every run validates its own D by replay before it reports
    assert!(err.contains("validated") && o.status.success(), "{err}");
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(&app).arg("--log").arg(&log).output().unwrap();
    assert!(String::from_utf8_lossy(&r.stdout).starts_with("replay ok"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profile_denies_network_and_exec_without_pty() {
    let p = rce_spec_check::sandbox::profile(Path::new("/tmp"), &[], false, false);
    assert!(p.contains("(deny network*)") && p.contains("(deny process-exec*)"));
    let q = rce_spec_check::sandbox::profile(Path::new("/tmp"), &[Path::new("/tmp")], true, true);
    assert!(!q.contains("process-exec") && !q.contains("network"));
}

/// Linux: the shell under `--pty` inherits Landlock: it can write under the
/// log directory but not elsewhere, and no TCP connection is allowed.
#[cfg(target_os = "linux")]
#[test]
fn landlock_confines_the_terminal_shell() {
    use std::io::Write;
    use std::process::Stdio;
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/terminal.rce");
    let dir = std::env::temp_dir().join(format!("rce-ll-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let outside = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("target/rce-ll-probe-{}", std::process::id()));
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve"])
        .arg(&app)
        .arg("--log")
        .arg(dir.join("d.log"))
        .arg("--pty")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut i = c.stdin.take().unwrap();
    let pause = |ms| std::thread::sleep(std::time::Duration::from_millis(ms));
    pause(500);
    writeln!(i, "touch {} && echo IN-OK", dir.join("inside").display()).unwrap();
    pause(300);
    writeln!(i, "touch {} 2>/dev/null && echo OUT-WROTE || echo OUT-DENIED", outside.display()).unwrap();
    pause(300);
    writeln!(i, "exit").unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while c.try_wait().unwrap().is_none() {
        assert!(std::time::Instant::now() < deadline);
        pause(50);
    }
    drop(i);
    let o = c.wait_with_output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let err = String::from_utf8_lossy(&o.stderr);
    if err.contains("Landlock unavailable") {
        eprintln!("Landlock unavailable on this kernel: {err}");
        return;
    }
    assert!(err.contains("Linux Landlock"), "{err}");
    assert!(out.contains("\nIN-OK") && out.contains("\nOUT-DENIED"), "{out}");
    assert!(!outside.exists());
    let _ = std::fs::remove_dir_all(&dir);
}
