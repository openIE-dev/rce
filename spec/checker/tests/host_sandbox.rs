//! W1h / HC-5: `rce host run` enters the sandbox's OS enforcement target after opening D;
//! the granted file root and the log still work, and the run's D replays.
use rce_spec_check::sandbox::{GrantKind, NetGrant, Rule, landlock_description, needs_ip_sockets, profile, seatbelt_description, seatbelt_rules};
use std::path::Path;
use std::process::Command;

#[test]
fn host_run_enters_sandbox_and_granted_authority_still_works() {
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/files.rce");
    let dir = std::env::temp_dir().join(format!("rce-sbx-{}", std::process::id()));
    let root = dir.join("root");
    std::fs::create_dir_all(&root).unwrap();
    let log = dir.join("d.log");
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "run", "fs"]).arg(&root).arg(&app).arg(&log).output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    let out = String::from_utf8_lossy(&o.stdout);
    if cfg!(target_os = "macos") {
        assert!(err.contains("sandbox, macOS target (Seatbelt)"), "{err}");
    } else if cfg!(target_os = "linux") {
        assert!(err.contains("sandbox, Linux target"), "{err}");
    } else {
        assert!(err.contains("sandbox OS target not entered"), "{err}");
    }
    // HC-28: the host's closing status line is on stderr, not stdout
    assert!(err.contains("durable") && !out.contains("durable"), "{out}\n{err}");
    // P5: every run validates its own D by replay before it reports
    assert!(err.contains("validated") && o.status.success(), "{err}");
    let r = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "replay"]).arg(&app).arg(&log).output().unwrap();
    assert!(String::from_utf8_lossy(&r.stdout).starts_with("replay ok"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn profile_denies_network_and_exec_without_pty() {
    let p = rce_spec_check::sandbox::profile(Path::new("/tmp"), &[], false, &[]);
    assert!(p.contains("(deny network*)") && p.contains("(deny process-exec*)"));
    assert!(!p.contains("(allow network"), "{p}");
    let q = rce_spec_check::sandbox::profile(Path::new("/tmp"), &[Path::new("/tmp")], true, &[]);
    assert!(!q.contains("process-exec") && q.contains("(deny network*)"));
}

fn grant(a: &str, k: GrantKind) -> NetGrant {
    NetGrant::parse(a, k).unwrap()
}

/// HC-5a: with network grants the Seatbelt profile still denies all network
/// use, then allows exactly the granted ports, at localhost for a loopback
/// grant and at any address otherwise (Seatbelt names no other host).
/// Before, a net grant removed the network rule altogether.
#[test]
fn profile_restricts_network_to_granted_ports() {
    let gs = [grant("127.0.0.1:8080", GrantKind::Prebound), grant("10.0.0.5:9000", GrantKind::Connect)];
    let p = profile(Path::new("/tmp"), &[], false, &gs);
    assert!(p.contains("(deny network*)\n"), "{p}");
    assert!(p.contains("(allow network-bind network-inbound (local ip \"localhost:8080\"))"), "{p}");
    assert!(p.contains("(allow network-outbound (remote ip \"*:9000\"))"), "{p}");
    assert_eq!(p.matches("(allow network").count(), 2, "nothing else is allowed: {p}");
    assert!(p.find("(deny network*)").unwrap() < p.find("(allow network").unwrap(), "the allows follow the deny");
    // a grant that names a host by name also needs the system resolver
    let named = profile(Path::new("/tmp"), &[], false, &[grant("example.org:443", GrantKind::Connect)]);
    assert!(named.contains("mDNSResponder") && named.contains("\"*:443\""), "{named}");
}

/// HC-5a: the printed line is made from the rule list that is installed,
/// so it cannot name more than the target enforces.
#[test]
fn description_is_generated_from_rules() {
    let gs = [grant("127.0.0.1:8080", GrantKind::Listen), grant("127.0.0.1:8080", GrantKind::Connect), grant("10.0.0.5:9000", GrantKind::Connect)];
    let rules = seatbelt_rules(&gs);
    let d = seatbelt_description(&rules);
    assert_eq!(d, "network only listening on ports [8080] at localhost and connecting to ports [8080] at localhost and connecting to ports [9000] at any address (TCP and UDP; Seatbelt cannot name other addresses; the net adapter checks the exact address)");
    // every port the line names is a rule of the profile, and the reverse
    let p = profile(Path::new("/tmp"), &[], false, &gs);
    for r in &rules {
        let port = match r {
            Rule::Inbound { port, .. } | Rule::Outbound { port, .. } => *port,
            Rule::Resolver => continue,
        };
        assert!(p.contains(&format!(":{port}\"")) && d.contains(&format!("{port}")), "{r:?}");
    }
    assert_eq!(seatbelt_description(&[]), "no network");
    // the Linux line, from what Landlock added
    let none = rce_spec_check::sandbox::Reads::default();
    let l = landlock_description(7, &[(8080, true, true)], &[grant("127.0.0.1:8080", GrantKind::Listen), grant("127.0.0.1:8080", GrantKind::Connect)], true, false, Ok(&none), false);
    assert_eq!(l, "Landlock ABI 7: TCP bind and connect only on ports [8080] (by port: any address on those ports; the net adapter allows only 127.0.0.1:8080), UDP not restricted, writes only under the log directory, reads only under the log directory, /dev (lstat and readlink of a path are not reads: Landlock has no right for them), no exec");
    let h = landlock_description(7, &[], &[grant("127.0.0.1:8080", GrantKind::Prebound)], false, false, Err("the terminal's programs read the system"), true);
    assert_eq!(h, "Landlock ABI 7: no TCP bind or connect (the listener 127.0.0.1:8080 was opened before entering), writes only under the log directory, reads not restricted (the terminal's programs read the system)");
    let db = [Path::new("/etc/passwd"), Path::new("/etc/group")];
    let names = rce_spec_check::sandbox::Reads { roots: vec![], files: db.to_vec() };
    let n = landlock_description(7, &[], &[], false, true, Ok(&names), false);
    assert!(n.contains("writes only under the log directory and the granted roots, reads only under the log directory, /dev and the granted roots, and the files /etc/passwd and /etc/group (lstat"), "{n}");
    // reads cannot be restricted under a terminal or a resolver
    assert!(rce_spec_check::sandbox::reads_open(true, &[]).is_some());
    assert!(rce_spec_check::sandbox::reads_open(false, &[grant("example.org:443", GrantKind::Connect)]).is_some());
    assert!(rce_spec_check::sandbox::reads_open(false, &[grant("127.0.0.1:443", GrantKind::Connect), grant("localhost:80", GrantKind::Prebound)]).is_none());
    let old = landlock_description(3, &[], &[grant("127.0.0.1:1", GrantKind::Connect)], true, true, Ok(&none), true);
    assert!(old.starts_with("Landlock ABI 3: network not restricted by Landlock (ABI 3), UDP not restricted"), "{old}");
    assert!(!needs_ip_sockets(&[grant("127.0.0.1:8080", GrantKind::Prebound)]));
    assert!(needs_ip_sockets(&[grant("127.0.0.1:8080", GrantKind::Listen)]));
}

/// HC-5a on Linux: Landlock restricts TCP by port, not by address, and not
/// UDP; the line the host prints says so.
#[cfg(target_os = "linux")]
#[test]
fn linux_description_says_by_port_and_udp() {
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/net.rce");
    let dir = std::env::temp_dir().join(format!("rce-sbx-net-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "run", "net"]).arg(format!("127.0.0.1:{port}")).arg(&app).arg(dir.join("d.log")).arg("Roll(n: 2)").output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    if err.contains("Landlock unavailable") {
        eprintln!("Landlock unavailable on this kernel: {err}");
        return;
    }
    if err.contains("Landlock ABI 1:") || err.contains("Landlock ABI 2:") || err.contains("Landlock ABI 3:") {
        assert!(err.contains("network not restricted by Landlock") && err.contains("UDP not restricted"), "{err}");
    } else {
        assert!(err.contains(&format!("TCP bind and connect only on ports [{port}] (by port: any address on those ports; the net adapter allows only 127.0.0.1:{port}), UDP not restricted")), "{err}");
    }
    assert!(!err.contains("no new IP sockets"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// What a shell under `pty` prints when it tries to open a UDP socket,
/// with these extra grants.
#[cfg(target_os = "linux")]
fn shell_udp(extra: &[String]) -> Option<String> {
    use std::io::{Read, Write};
    use std::process::Stdio;
    if Command::new("python3").args(["-c", "pass"]).output().map(|o| !o.status.success()).unwrap_or(true) {
        eprintln!("no python3 here: skipped");
        return None;
    }
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/pty_http.rce");
    let dir = std::env::temp_dir().join(format!("rce-sbx-udp-{}-{}", std::process::id(), extra.len()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "serve", "pty", "http"])
        .arg(format!("127.0.0.1:{port}"))
        .args(extra)
        .arg(&app)
        .arg(dir.join("d.log"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut i = c.stdin.take().unwrap();
    writeln!(i, "python3 -c 'import socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM); print(\"UDP-\" + \"OPEN\")' 2>/dev/null || echo UDP-$((1 + 1))").unwrap();
    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let (s2, mut out) = (seen.clone(), c.stdout.take().unwrap());
    std::thread::spawn(move || {
        let mut b = [0u8; 4096];
        while let Ok(n) = out.read(&mut b) {
            if n == 0 {
                break;
            }
            s2.lock().unwrap().push_str(&String::from_utf8_lossy(&b[..n]));
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let got = loop {
        let s = seen.lock().unwrap().clone();
        if s.contains("UDP-OPEN") || s.contains("UDP-2") {
            break s;
        }
        assert!(std::time::Instant::now() < deadline, "no answer from the shell: {s}");
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    // SAFETY: a signal to our own child process.
    unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    drop(i);
    let o = c.wait_with_output().unwrap();
    let err = String::from_utf8_lossy(&o.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&dir);
    Some(format!("{got}\n{err}"))
}

/// HC-5a: with `http` alone the listener was bound before entering, so
/// seccomp fails every new IP socket: the shell cannot open UDP. With a
/// `net` grant it can, and the line says UDP is not restricted.
#[cfg(target_os = "linux")]
#[test]
fn http_only_has_no_new_ip_sockets() {
    let Some(only) = shell_udp(&[]) else { return };
    if only.contains("seccomp not entered") {
        eprintln!("seccomp not entered here: {only}");
        return;
    }
    assert!(only.contains("UDP-2") && !only.contains("UDP-OPEN"), "{only}");
    assert!(only.contains("no new IP sockets") && only.contains("was opened before entering"), "{only}");
    assert!(only.contains("validated"), "{only}");
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let Some(net) = shell_udp(&["net".into(), format!("127.0.0.1:{port}")]) else { return };
    assert!(net.contains("UDP-OPEN"), "{net}");
    assert!(net.contains("UDP not restricted") && !net.contains("no new IP sockets"), "{net}");
}

/// macOS runner: under Seatbelt, a port that was not granted is not opened.
#[cfg(target_os = "macos")]
#[test]
fn seatbelt_blocks_an_ungranted_port() {
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/net.rce");
    let dir = std::env::temp_dir().join(format!("rce-sbx-mac-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let granted = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let other = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let oport = other.local_addr().unwrap().port();
    let o = Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "run", "net"])
        .arg(format!("127.0.0.1:{granted}"))
        .arg("net")
        .arg(format!("127.0.0.1:{oport}"))
        .arg(&app)
        .arg(dir.join("d.log"))
        .arg(format!("Listen(addr: \"127.0.0.1:{granted}\")"))
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(err.contains("sandbox, macOS target (Seatbelt): network only"), "{err}");
    assert!(err.contains(&format!("{granted}")), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// macOS runner: a request is answered under Seatbelt.
#[cfg(target_os = "macos")]
#[test]
fn serve_under_seatbelt_answers_a_request() {
    use std::io::{Read, Write};
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../programs/network/notes_http.rce");
    let dir = std::env::temp_dir().join(format!("rce-sbx-mac-serve-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "serve", "http"]).arg(format!("127.0.0.1:{port}")).arg(&app).arg(dir.join("d.log")).stdin(std::process::Stdio::null()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    let answer = loop {
        if let Ok(mut s) = std::net::TcpStream::connect(("127.0.0.1", port)) {
            s.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").unwrap();
            let mut out = String::new();
            let _ = s.read_to_string(&mut out);
            break out;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    // SAFETY: a signal to our own child process.
    unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    let _ = c.wait();
    assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
    let _ = std::fs::remove_dir_all(&dir);
}
/// Linux: the shell under `pty` inherits Landlock: it can write under the
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
        .args(["host", "serve", "pty"])
        .arg(&app)
        .arg(dir.join("d.log"))
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
    assert!(err.contains("Landlock ABI"), "{err}");
    if err.contains("sandbox,") {
        assert!(err.contains("Linux target"), "{err}");
    }
    assert!(out.contains("\nIN-OK") && out.contains("\nOUT-DENIED"), "{out}");
    assert!(!outside.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The child half of `linux_reads_are_restricted_to_the_grants`: run only
/// when that test starts this binary again with RCE_SBX_CHILD set. It
/// enters the Linux target with the reads the variables name, then reports
/// what it could read.
#[cfg(target_os = "linux")]
#[test]
fn sandboxed_child_reads() {
    let Ok(dir) = std::env::var("RCE_SBX_CHILD") else { return };
    let dir = Path::new(&dir);
    let root = dir.join("root");
    let db = Path::new("/etc/passwd");
    let files: Vec<&Path> = if std::env::var("RCE_SBX_NAMES").is_ok() { vec![db] } else { vec![] };
    let reads = rce_spec_check::sandbox::Reads { roots: vec![], files };
    let line = rce_spec_check::sandbox::enter(&dir.join("log"), &[root.as_path()], &reads, false, &[]).unwrap_or_else(|e| format!("NOT ENTERED {e}"));
    println!("LINE {line}");
    let say = |what: &str, ok: bool| println!("{what} {}", if ok { "OK" } else { "DENIED" });
    say("PASSWD", std::fs::read(db).is_ok());
    say("ROOT-LIST", std::fs::read_dir(&root).is_ok());
    say("ROOT-READ", std::fs::read(root.join("f")).is_ok());
    say("OUTSIDE-LIST", std::fs::read_dir("/etc").is_ok());
    say("OUTSIDE-LSTAT", std::fs::symlink_metadata("/etc/passwd").is_ok());
    say("LINK-READLINK", std::fs::read_link(root.join("l")).is_ok());
}

/// HC-5a, HC-33: on Linux the target allows reading only under the log
/// directory, /dev and the granted roots, and the account database only
/// when it is granted; lstat and readlink stay possible, and the line says
/// what was installed.
#[cfg(target_os = "linux")]
#[test]
fn linux_reads_are_restricted_to_the_grants() {
    let dir = std::env::temp_dir().join(format!("rce-sbx-reads-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("root")).unwrap();
    std::fs::create_dir_all(dir.join("log")).unwrap();
    std::fs::write(dir.join("root/f"), "x").unwrap();
    std::os::unix::fs::symlink("/etc/passwd", dir.join("root/l")).unwrap();
    let child = |names: bool| {
        let mut c = Command::new(std::env::current_exe().unwrap());
        // libtest prints a test's output only when it is not captured
        c.arg("sandboxed_child_reads").env("RCE_SBX_CHILD", &dir).env("RUST_TEST_NOCAPTURE", "1");
        if names {
            c.env("RCE_SBX_NAMES", "1");
        }
        String::from_utf8_lossy(&c.output().unwrap().stdout).into_owned()
    };
    let without = child(false);
    if without.contains("Landlock unavailable") || without.contains("NOT ENTERED") {
        eprintln!("Landlock not entered here: {without}");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(without.contains("reads only under the log directory, /dev and the granted roots (lstat and readlink of a path are not reads"), "{without}");
    assert!(!without.contains("/etc/passwd"), "{without}");
    for (k, v) in [("PASSWD", "DENIED"), ("ROOT-LIST", "OK"), ("ROOT-READ", "OK"), ("OUTSIDE-LIST", "DENIED"), ("OUTSIDE-LSTAT", "OK"), ("LINK-READLINK", "OK")] {
        assert!(without.contains(&format!("{k} {v}")), "{k}: {without}");
    }
    let with = child(true);
    assert!(with.contains("and the files /etc/passwd"), "{with}");
    assert!(with.contains("PASSWD OK") && with.contains("OUTSIDE-LIST DENIED"), "{with}");
    let _ = std::fs::remove_dir_all(&dir);
}
