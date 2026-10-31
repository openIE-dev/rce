//! HC-28: the host loop reports its status to its terminal as OSC 7501
//! records on stderr (`idle`, `working`, `done`, `error`), only when stderr
//! is a terminal or `RCE_STATUS=on`. The report is not an effect: D is the
//! same byte for byte with and without it.
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const COUNTER: &str = "module counter\n\nevent Inc\n\nstate C\n    n: i64 = 0\n\nstep C on Inc\n    n += 1\n";
const OTHER: &str = "module counter\n\nevent Inc\n\nstate C\n    n: i64 = 0\n\nstep C on Inc\n    n += 2\n";

struct Out {
    code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// `rce ARGS` with `stdin` as its input and stdout and stderr piped (so
/// neither is a terminal); `status` sets RCE_STATUS, or removes it.
fn rce(args: &[&str], stdin: &str, status: Option<&str>) -> Out {
    let mut c = Command::new(env!("CARGO_BIN_EXE_rce"));
    c.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    match status {
        Some(v) => c.env("RCE_STATUS", v),
        None => c.env_remove("RCE_STATUS"),
    };
    let mut p = c.spawn().unwrap();
    p.stdin.take().unwrap().write_all(stdin.as_bytes()).unwrap();
    let o = p.wait_with_output().unwrap();
    Out { code: o.status.code().unwrap_or(-1), stdout: o.stdout, stderr: o.stderr }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-hc28-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write(dir: &Path, name: &str, text: &str) -> String {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

fn unbase64(s: &str) -> String {
    const TABLE: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let (mut bits, mut n, mut out) = (0u32, 0u32, vec![]);
    for c in s.bytes().filter(|c| *c != b'=') {
        bits = bits << 6 | TABLE.iter().position(|t| *t == c).expect("base64") as u32;
        n += 6;
        if n >= 8 {
            n -= 8;
            out.push((bits >> n) as u8);
        }
    }
    String::from_utf8(out).unwrap()
}

/// The records of OSC 7501 in `bytes`, each as its (key, value) pairs with
/// `title` and `msg` decoded.
fn records(bytes: &[u8]) -> Vec<Vec<(String, String)>> {
    let text = String::from_utf8_lossy(bytes);
    let mut out = vec![];
    for part in text.split("\x1b]7501;").skip(1) {
        let body = part.split_once("\x1b\\").expect("a record ends with ESC \\").0;
        out.push(
            body.split(':')
                .map(|kv| {
                    let (k, v) = kv.split_once('=').unwrap();
                    let v = if k == "title" || k == "msg" { unbase64(v) } else { v.to_string() };
                    (k.to_string(), v)
                })
                .collect(),
        );
    }
    out
}

fn field<'a>(r: &'a [(String, String)], k: &str) -> &'a str {
    r.iter().find(|(x, _)| x == k).map_or("", |(_, v)| v.as_str())
}

fn states(rs: &[Vec<(String, String)>]) -> Vec<&str> {
    rs.iter().map(|r| field(r, "state")).collect()
}

fn state_hash(stdout: &[u8]) -> String {
    let s = String::from_utf8_lossy(stdout);
    s.split("state_hash ").nth(1).unwrap().split_whitespace().next().unwrap().to_string()
}

#[test]
fn host_run_prints_status_records_in_order_on_stderr() {
    let d = dir("run");
    let (prog, other) = (write(&d, "counter.rce", COUNTER), write(&d, "other.rce", OTHER));
    let log = d.join("d").to_string_lossy().into_owned();
    // two input lines: each is admitted, stepped and appended between a
    // `working` and an `idle`
    let o = rce(&["host", "run", &prog, "--log", &log], "Inc\nInc, Inc\n", Some("on"));
    assert_eq!(o.code, 0, "{}", String::from_utf8_lossy(&o.stderr));
    assert!(!o.stdout.contains(&0x1b), "stdout carries no record");
    let rs = records(&o.stderr);
    assert_eq!(states(&rs), ["idle", "working", "idle", "working", "idle", "done"], "{rs:?}");
    for r in &rs {
        assert_eq!(field(r, "app"), "rce");
        assert_eq!(field(r, "title"), "counter", "the title is the module name");
        assert_eq!(field(r, "kind"), "", "a report is never `blocked`");
    }
    let hash = state_hash(&o.stdout);
    let entries: Vec<&str> = rs.iter().map(|r| field(r, "msg")).collect();
    let n = String::from_utf8_lossy(&o.stdout).split("entries=").nth(1).unwrap().split_whitespace().next().unwrap().to_string();
    assert_eq!(entries[4], format!("{n} entries, state {}", &hash[..12]), "the idle msg gives the entry count and a short state hash");
    assert_eq!(entries[5], format!("stopped: {n} entries, state {}", &hash[..12]));

    // a program that requires change on this log reports `error` and stops
    let o = rce(&["host", "run", &other, "--log", &log, "Inc"], "", Some("on"));
    assert_eq!(o.code, 1);
    let rs = records(&o.stderr);
    assert_eq!(states(&rs), ["error"], "{rs:?}");
    assert!(field(&rs[0], "msg").starts_with("requires change: HC-27"), "{rs:?}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn host_serve_reports_idle_working_and_done() {
    let d = dir("serve");
    let prog = write(&d, "counter.rce", COUNTER);
    let log = d.join("d").to_string_lossy().into_owned();
    // the end of the input is a clean stop
    let o = rce(&["host", "serve", &prog, "--log", &log], "Inc\nInc\n", Some("on"));
    assert_eq!(o.code, 0, "{}", String::from_utf8_lossy(&o.stderr));
    let rs = records(&o.stderr);
    let s = states(&rs);
    assert_eq!(s.first(), Some(&"idle"), "{s:?}");
    assert_eq!(s.last(), Some(&"done"), "{s:?}");
    assert!(s.contains(&"working"), "{s:?}");
    assert!(s.windows(2).all(|w| w[0] != w[1]), "a record only on a change of state: {s:?}");
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn no_escape_byte_without_a_terminal() {
    let d = dir("pipe");
    let prog = write(&d, "counter.rce", COUNTER);
    for (i, status) in [None, Some("off"), Some("auto")].into_iter().enumerate() {
        let log = d.join(format!("d{i}")).to_string_lossy().into_owned();
        for sub in ["run", "serve"] {
            let o = rce(&["host", sub, &prog, "--log", &log], "Inc\n", status);
            assert_eq!(o.code, 0, "{}", String::from_utf8_lossy(&o.stderr));
            assert!(!o.stderr.contains(&0x1b) && !o.stdout.contains(&0x1b), "{sub} with RCE_STATUS={status:?} wrote an ESC byte");
        }
    }
    // a program that requires change, and so a stop, is not reported either
    let other = write(&d, "other.rce", OTHER);
    let o = rce(&["host", "run", &other, "--log", &d.join("d0").to_string_lossy(), "Inc"], "", None);
    assert_eq!(o.code, 1);
    assert!(!o.stderr.contains(&0x1b) && !o.stdout.contains(&0x1b));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn d_is_byte_identical_with_and_without_reports() {
    let d = dir("same");
    let prog = write(&d, "counter.rce", COUNTER);
    for sub in ["run", "serve"] {
        let (on, off) = (d.join(format!("{sub}-on")).to_string_lossy().into_owned(), d.join(format!("{sub}-off")).to_string_lossy().into_owned());
        let a = rce(&["host", sub, &prog, "--log", &on], "Inc\nInc, Inc\n", Some("on"));
        let b = rce(&["host", sub, &prog, "--log", &off], "Inc\nInc, Inc\n", Some("off"));
        assert_eq!((a.code, b.code), (0, 0));
        assert!(!records(&a.stderr).is_empty() && records(&b.stderr).is_empty());
        assert_eq!(std::fs::read(&on).unwrap(), std::fs::read(&off).unwrap(), "{sub}: D does not depend on the report");
        assert_eq!(std::fs::read(format!("{on}.program")).unwrap(), std::fs::read(format!("{off}.program")).unwrap());
        assert_eq!(a.stdout, b.stdout, "{sub}: the program's output does not depend on the report");
        // replay gives the same state from either log, with no report
        let r = rce(&["host", "replay", &prog, "--log", &on], "", Some("on"));
        assert_eq!(r.code, 0);
        assert!(!r.stderr.contains(&0x1b) && !r.stdout.contains(&0x1b), "replay is not the host loop and reports nothing");
    }
    let _ = std::fs::remove_dir_all(&d);
}
