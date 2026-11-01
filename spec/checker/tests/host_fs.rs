//! HC-30 to HC-33: the directory, stat, link, clock, zone and account-name
//! requests. Each request has one result event in D (HC-3, HC-10), replay
//! with every adapter disconnected re-derives every state (HC-8), names and
//! link texts arrive as their exact bytes (HC-10a), a request that does not
//! follow a final link is checked on the directory that holds it (HC-31),
//! and the account databases are read only under `names` (HC-33, HC-5a).
#![cfg(unix)]
use rce_spec_check::adapters::{LocalOffsetAdapter, WallClockAdapter, file_adapters, local_offset_adapter, lookup_name, name_adapters_from};
use rce_spec_check::host::{Adapter, Host, MemStore, replay};
use rce_spec_check::parse::parse;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn src() -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/fs_stat.rce")).unwrap()
}

fn log_text(st: &MemStore) -> String {
    st.lines.iter().map(|l| format!("{l}\n")).collect()
}

fn base(tag: &str) -> PathBuf {
    let b = std::env::temp_dir().join(format!("rce-fs-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&b);
    std::fs::create_dir_all(b.join("root")).unwrap();
    b
}

fn files(root: &Path) -> Vec<Box<dyn Adapter>> {
    file_adapters(root).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect()
}

/// The program's state, and the replay of its D, which must reach the
/// live state hash.
fn run(adapters: Vec<Box<dyn Adapter>>, events: &[&str]) -> (String, String) {
    let s = src();
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), adapters).unwrap();
    for e in events {
        h.admit(&s, e).unwrap();
    }
    let (live, state) = (h.state_hash().to_string(), format!("{:?}", h.state()));
    drop(h);
    let text = log_text(&st);
    let (_, replayed, _) = replay(&m, &text).unwrap();
    assert_eq!(replayed, live, "replay with adapters disconnected re-derives the state (HC-8)");
    (state, text)
}

/// The bytes of a text as the list a log holds.
fn list(b: &[u8]) -> String {
    format!("[{}]", b.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

/// HC-31: lstat answers the link's own record, readlink its target text
/// unchanged, though the target leaves the root; FileRead, which follows
/// the link, is still refused; and a dangling link has a record too.
#[test]
fn lstat_and_readlink_do_not_follow_and_replay() {
    let b = base("lstat");
    let root = b.join("root");
    std::fs::write(b.join("secret.txt"), "no").unwrap();
    std::fs::write(root.join("a.txt"), "hello").unwrap();
    std::os::unix::fs::symlink(b.join("secret.txt"), root.join("out")).unwrap();
    std::os::unix::fs::symlink("sub/missing", root.join("dangling")).unwrap();
    std::fs::create_dir(root.join("sub")).unwrap();
    let ev = [
        r#"Lstat(path: [97, 46, 116, 120, 116])"#.to_string(),
        r#"Lstat(path: [111, 117, 116])"#.to_string(),
        r#"Lstat(path: [100, 97, 110, 103, 108, 105, 110, 103])"#.to_string(),
        r#"Lstat(path: [115, 117, 98, 47, 46, 46])"#.to_string(),
        r#"Link(path: "out")"#.to_string(),
        r#"Load(path: "out")"#.to_string(),
        r#"Lstat(path: [46, 46])"#.to_string(),
        r#"Link(path: "a.txt")"#.to_string(),
    ];
    let evs: Vec<&str> = ev.iter().map(String::as_str).collect();
    let (state, text) = run(files(&root), &evs);
    use std::os::unix::fs::MetadataExt;
    let a = std::fs::symlink_metadata(root.join("a.txt")).unwrap();
    let out = std::fs::symlink_metadata(root.join("out")).unwrap();
    let r = std::fs::metadata(&root).unwrap();
    assert!(state.contains(&format!("(\"mode\", Int({}", a.mode())) && state.contains(&format!("(\"size\", Int(5")), "{state}");
    assert_eq!(out.mode() & 0o170000, 0o120000);
    assert!(state.contains(&format!("(\"mode\", Int({}", out.mode())), "the link's own mode: {state}");
    assert!(state.contains(&format!("(\"mtime\", Int({}", a.mtime())) && state.contains("mtime_ns") && state.contains("blocks"), "{state}");
    // sub/.. is the root itself
    assert!(state.contains(&format!("(\"nlink\", Int({}", r.nlink())), "{state}");
    // the target's text, byte for byte, and the secret never read
    let t = std::fs::read_link(root.join("out")).unwrap();
    assert!(text.contains(&format!("\"data\":{}", list(t.as_os_str().as_bytes()))), "{text}");
    assert!(!text.contains("\"no\""), "the secret never reaches D");
    // FileRead follows the link and is refused; `..` of the root is outside
    // it; readlink of a file that is not a link fails
    assert_eq!(text.matches("outside the granted root").count(), 2, "{text}");
    assert!(state.contains("errors\", List([Text("), "{state}");
    assert_eq!(text.matches("\"status\":\"failed\"").count(), 3, "{text}");
    let _ = std::fs::remove_dir_all(&b);
}

/// HC-30, HC-10a: two names that differ only in bytes that are not UTF-8
/// arrive as two names, each exactly, in byte order. A program that takes
/// the names as text gets `failed` with the reason, and `host.Kept` holds
/// the bytes; replay re-derives both.
#[test]
fn directory_names_arrive_as_their_exact_bytes() {
    let b = base("names");
    let root = b.join("root");
    for n in [&b"a\xffb"[..], b"a\xfeb", b"plain"] {
        std::fs::write(root.join(std::ffi::OsStr::from_bytes(n)), "").unwrap();
    }
    let (state, text) = run(files(&root), &[r#"Names(path: "")"#]);
    assert!(text.contains("\"data\":[[97,254,98],[97,255,98],[112,108,97,105,110]]"), "{text}");
    assert!(state.contains("Int(254"), "{state}");
    // as text: not UTF-8, so the result is failed and host.Kept keeps the bytes
    let s = src().replace("event FileListDone(req: u64, status: text, data: List[List[i64]]?, err: text)", "event FileListDone(req: u64, status: text, data: List[text]?, err: text)").replace("names: List[List[i64]] = []", "names: List[text] = []").replace("example Names(path: \"\"), FileListDone(req: 1, status: \"delivered\", data: some([[97, 255], [97, 254]]), err: \"\") -> names == [[97, 255], [97, 254]]", "example Names(path: \"\") -> next == 1");
    let m = parse(&s).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), files(&root)).unwrap();
    h.admit(&s, r#"Names(path: "")"#).unwrap();
    let live = h.state_hash().to_string();
    drop(h);
    let t = log_text(&st);
    assert!(t.contains("host.Kept") && t.contains("[97,254,98]") && t.contains("element 0: not UTF-8 at offset 1"), "{t}");
    assert_eq!(replay(&m, &t).unwrap().1, live);
    // all UTF-8: delivered as text, and D holds the text
    std::fs::remove_file(root.join(std::ffi::OsStr::from_bytes(b"a\xffb"))).unwrap();
    std::fs::remove_file(root.join(std::ffi::OsStr::from_bytes(b"a\xfeb"))).unwrap();
    let mut st = MemStore::default();
    let mut h = Host::start_with(&m, Box::new(&mut st), files(&root)).unwrap();
    h.admit(&s, r#"Names(path: "")"#).unwrap();
    drop(h);
    let t = log_text(&st);
    assert!(t.contains("\"data\":[\"plain\"]") && !t.contains("host.Kept"), "{t}");
    let _ = std::fs::remove_dir_all(&b);
}

/// HC-33: a uid or gid to its name, from the first line of the database
/// with that id in its third field; an id without a line fails; replay
/// reads the answers from D.
#[test]
fn account_names_come_from_the_databases_and_replay() {
    let b = base("acct");
    let (pw, gr) = (b.join("passwd"), b.join("group"));
    std::fs::write(&pw, "root:x:0:0:root:/root:/bin/sh\nalice:x:1000:1000::/home/alice:/bin/sh\nalias:x:1000:1000::/:/bin/sh\n").unwrap();
    std::fs::write(&gr, "root:x:0:\nstaff:x:50:alice\n").unwrap();
    assert_eq!(lookup_name(b"a:x:7:\nb:x:70:\n", 70), Some(b"b".to_vec()));
    assert_eq!(lookup_name(b"a:x:7:\n", 8), None);
    let ads: Vec<Box<dyn Adapter>> = name_adapters_from(&pw, &gr).into_iter().map(|a| Box::new(a) as Box<dyn Adapter>).collect();
    let (state, text) = run(ads, &["User(id: 1000)", "Group(id: 50)", "User(id: 4242)"]);
    assert!(state.contains("Text(\"alice\")") && state.contains("Text(\"staff\")"), "{state}");
    assert!(text.contains("no user with id 4242 in"), "{text}");
    let _ = std::fs::remove_dir_all(&b);
}

/// HC-2, HC-32: the clock and the zone's offset are results in D; replay
/// reads them from D and from no clock or zone file.
#[test]
fn clock_and_offset_are_logged_answers() {
    let ads: Vec<Box<dyn Adapter>> = vec![Box::new(WallClockAdapter), Box::new(local_offset_adapter())];
    let (state, text) = run(ads, &["Now", "Offset(at: 1782950400)"]);
    assert!(text.contains("WallClockDone") && text.contains("LocalOffsetDone"), "{text}");
    assert!(!state.contains("now_ns\", Int(0"), "{state}");
    let _: LocalOffsetAdapter = local_offset_adapter();
}

/// HC-32 under the CLI: the offset follows the zone's rules at each
/// instant (a POSIX TZ rule with summer time), and the run validates.
#[test]
fn offset_follows_the_zone_at_each_instant() {
    let b = base("tz");
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/fs_stat.rce");
    let o = Command::new(env!("CARGO_BIN_EXE_rce"))
        .env("TZ", "EST5EDT,M3.2.0,M11.1.0")
        .args(["host", "run"])
        .arg(&app)
        .arg(b.join("d.log"))
        .arg("Offset(at: 1578000000), Offset(at: 1782950400)")
        .output()
        .unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(o.status.success() && err.contains("validated"), "{out}\n{err}");
    assert!(out.contains("offsets = [-18000, -14400]"), "{out}");
    let _ = std::fs::remove_dir_all(&b);
}

/// HC-33, HC-5a: without `names` no request reads the account databases
/// (no adapter answers UserName) and the sandbox line names no such file;
/// with it, the request is answered inside the sandbox and the line names
/// exactly the two files. Under `fs`, lstat, readlink and listing work
/// inside the sandbox, which restricts reads to the granted roots.
#[test]
fn names_grant_reads_the_databases_and_the_sandbox_says_so() {
    let b = base("grant");
    let root = b.join("root");
    std::fs::write(root.join("f"), "x").unwrap();
    std::os::unix::fs::symlink("/etc/hostname-not-followed", root.join("l")).unwrap();
    let app = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/host/fs_stat.rce");
    let rce = |words: &[&str], log: &str| {
        let o = Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "run"]).args(words).arg(&app).arg(b.join(log)).arg(r#"User(id: 0), Names(path: ""), Lstat(path: [108]), Link(path: "l")"#).output().unwrap();
        (String::from_utf8_lossy(&o.stdout).into_owned(), String::from_utf8_lossy(&o.stderr).into_owned(), o.status.success())
    };
    let root_s = root.to_str().unwrap();
    let (out, err, ok) = rce(&["fs", root_s], "d1.log");
    assert!(ok && err.contains("validated"), "{out}\n{err}");
    assert!(!out.contains("UserNameDone"), "no adapter answers without `names`: {out}");
    assert!(!err.contains("/etc/passwd"), "{err}");
    assert!(out.contains("names = [[102], [108]]"), "{out}");
    assert!(out.contains("target = [47, 101, 116, 99"), "{out}");
    let (out, err, ok) = rce(&["fs", root_s, "names"], "d2.log");
    assert!(ok && err.contains("validated"), "{out}\n{err}");
    assert!(out.contains("UserNameDone"), "{out}");
    if cfg!(target_os = "linux") && err.contains("Landlock ABI") {
        assert!(err.contains("reads only under the log directory, /dev and the granted roots, and the files /etc/passwd and /etc/group"), "{err}");
        assert!(out.contains("user = \"root\""), "uid 0 is root here: {out}");
    }
    let _ = std::fs::remove_dir_all(&b);
}

/// HC-30 to HC-33 end to end: `programs/systems/ls.rce` on the host lists
/// a directory as GNU coreutils `ls`, `ls -a`, `ls -l` and `ls -la` do in
/// the C locale, byte for byte, names that are not UTF-8 included, and its
/// log replays. Run where GNU ls, mkfifo and python3 are found.
#[test]
fn ls_lists_through_the_host_as_coreutils_does() {
    let gnu = Command::new("ls").arg("--version").output().map(|o| String::from_utf8_lossy(&o.stdout).contains("GNU coreutils")).unwrap_or(false);
    let py = Command::new("python3").args(["-c", "pass"]).output().map(|o| o.status.success()).unwrap_or(false);
    if !gnu || !py {
        eprintln!("no GNU ls or python3 here: skipped");
        return;
    }
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../programs/systems/ls_coreutils.sh");
    let b = base("coreutils");
    let o = Command::new("bash").arg(&script).arg(&b).env("RCE", env!("CARGO_BIN_EXE_rce")).output().unwrap();
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(o.status.success(), "{out}\n{}", String::from_utf8_lossy(&o.stderr));
    for k in ["MATCH ls (", "MATCH ls -a (", "MATCH ls -l (", "MATCH ls -la ("] {
        assert!(out.contains(k), "{k}: {out}");
    }
    assert!(out.contains("replay ok"), "{out}");
    let _ = std::fs::remove_dir_all(&b);
}
