//! W1c host adapters (Appendix E.1, HC-3, HC-5): file and PTY.
//!
//! Each adapter answers one request kind (a struct the program emits with
//! `req: u64`) with exactly one result, which the admitter logs as the
//! program's `<Kind>Done(req, status, data: T?, err)` event. Replay never
//! calls an adapter (HC-8). Authority is explicit: a file adapter reaches
//! only paths under the root it was given (relative paths, no `..`, no
//! symlink leaving the root), and PTY adapters exist only when the host was
//! given PTY authority.
use crate::host::{Adapter, EffectResult};
use serde_json::{Value, json};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

/// The file request kinds and their result types:
/// `FileRead(req, path)` -> text; `FileWrite(req, path, data)` -> u64 bytes
/// (atomic: temp file, barrier, rename, directory barrier);
/// `FileAppend(req, path, data)` -> u64 bytes (barrier before the result);
/// `FileList(req, path)` -> the names, each as its exact bytes (HC-10a), in
/// byte order (HC-30);
/// `FileLstat(req, path)` -> the record {mode, nlink, uid, gid, size,
/// mtime, mtime_ns, blocks} of lstat(2): a final symbolic link is not
/// followed (HC-31);
/// `FileReadlink(req, path)` -> a symbolic link's target, as its exact
/// bytes (HC-31).
pub const FILE_KINDS: [&str; 6] = ["FileRead", "FileWrite", "FileAppend", "FileList", "FileLstat", "FileReadlink"];

pub struct FileAdapter {
    kind: &'static str,
    root: PathBuf,
}

/// One adapter per file request kind, all under `root`.
pub fn file_adapters(root: &Path) -> Vec<FileAdapter> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    FILE_KINDS.iter().map(|k| FileAdapter { kind: k, root: root.clone() }).collect()
}

/// Whether a request kind reads the entry a path names without following a
/// final symbolic link (HC-31).
fn no_follow(kind: &str) -> bool {
    matches!(kind, "FileLstat" | "FileReadlink")
}

/// A path as the OS takes it, from its exact bytes.
#[cfg(unix)]
fn path_of(b: &[u8]) -> PathBuf {
    PathBuf::from(<std::ffi::OsStr as std::os::unix::ffi::OsStrExt>::from_bytes(b))
}
#[cfg(not(unix))]
fn path_of(b: &[u8]) -> PathBuf {
    PathBuf::from(String::from_utf8_lossy(b).into_owned())
}

/// The path a request names: text, or its exact bytes as a list of
/// integers in 0..255 (a name that is not UTF-8, HC-30).
fn request_path(r: &Value) -> Result<Vec<u8>, String> {
    match &r["path"] {
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        Value::Array(_) => request_bytes(&r["path"]).map_err(|e| format!("path: {e}")),
        Value::Null => Ok(vec![]),
        _ => Err("path is neither text nor bytes".into()),
    }
}

impl FileAdapter {
    /// The path under the root, or why the access is not performed (least
    /// privilege).
    fn resolve(&self, pb: &[u8]) -> Result<PathBuf, String> {
        if pb.is_empty() && matches!(self.kind, "FileList" | "FileLstat") {
            return Ok(self.root.clone());
        }
        if no_follow(self.kind) {
            return self.resolve_entry(pb);
        }
        let p = String::from_utf8_lossy(pb);
        let rel = path_of(pb);
        if pb.is_empty() || rel.is_absolute() || rel.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
            return Err(format!("path `{p}` is outside the granted root"));
        }
        let full = self.root.join(&rel);
        let probe = if full.exists() {
            full.canonicalize().map_err(|e| e.to_string())?
        } else {
            let parent = full.parent().unwrap_or(&self.root);
            parent.canonicalize().map_err(|e| e.to_string())?.join(full.file_name().unwrap_or_default())
        };
        if !probe.starts_with(&self.root) {
            return Err(format!("path `{p}` is outside the granted root"));
        }
        Ok(full)
    }

    /// HC-31: the entry a path names, for a request that does not follow a
    /// final symbolic link. The root check applies to the directory that
    /// holds the entry, resolved with every link in it followed, and not to
    /// where the entry points: a link whose target leaves the root is
    /// still an entry under it. A path whose last part is `.` or `..` names
    /// a directory itself, which must be under the root.
    fn resolve_entry(&self, pb: &[u8]) -> Result<PathBuf, String> {
        let outside = || format!("path `{}` is outside the granted root", String::from_utf8_lossy(pb));
        let rel = path_of(pb);
        if pb.is_empty() || rel.is_absolute() || rel.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir | Component::ParentDir)) {
            return Err(outside());
        }
        let full = self.root.join(&rel);
        let trimmed = pb.iter().rposition(|c| *c != b'/').map_or(&pb[..0], |i| &pb[..=i]);
        let last = trimmed.rsplit(|c| *c == b'/').next().unwrap_or(&[]);
        let probe = match full.file_name() {
            Some(name) if last != b"." && last != b".." => {
                let parent = full.parent().unwrap_or(&self.root);
                parent.canonicalize().map_err(|e| e.to_string())?.join(name)
            }
            _ => full.canonicalize().map_err(|e| e.to_string())?,
        };
        if !probe.starts_with(&self.root) {
            return Err(outside());
        }
        Ok(probe)
    }
}

fn barrier_dir(p: &Path) -> std::io::Result<()> {
    let d = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    File::open(d)?.sync_all()
}

/// Bytes received from outside, as the host's value (HC-10a): a list of
/// integers in 0..255.
fn bytes_json(b: &[u8]) -> Value {
    Value::Array(b.iter().map(|x| Value::from(*x)).collect())
}

/// The exact bytes of a name the OS gives (HC-10a).
#[cfg(unix)]
fn os_bytes(s: &std::ffi::OsStr) -> Vec<u8> {
    std::os::unix::ffi::OsStrExt::as_bytes(s).to_vec()
}
#[cfg(not(unix))]
fn os_bytes(s: &std::ffi::OsStr) -> Vec<u8> {
    s.to_string_lossy().into_owned().into_bytes()
}

/// HC-31: the lstat(2) record of an entry.
#[cfg(unix)]
fn lstat_json(full: &Path) -> Result<Value, String> {
    use std::os::unix::fs::MetadataExt;
    let m = fs::symlink_metadata(full).map_err(|e| e.to_string())?;
    Ok(json!({
        "mode": m.mode(), "nlink": m.nlink(), "uid": m.uid(), "gid": m.gid(), "size": m.size(),
        "mtime": m.mtime(), "mtime_ns": m.mtime_nsec(), "blocks": m.blocks(),
    }))
}
#[cfg(not(unix))]
fn lstat_json(_full: &Path) -> Result<Value, String> {
    Err("FileLstat needs a Unix host".into())
}

fn perform_file(kind: &str, full: &Path, r: &Value) -> Result<Value, String> {
    let s = |e: std::io::Error| e.to_string();
    match kind {
        "FileRead" => fs::read_to_string(full).map(Value::String).map_err(s),
        "FileWrite" => {
            let data = r["data"].as_str().unwrap_or("");
            let tmp = full.with_extension(format!("rce-tmp-{}", std::process::id()));
            let mut f = File::create(&tmp).map_err(s)?;
            f.write_all(data.as_bytes()).map_err(s)?;
            f.sync_data().map_err(s)?;
            fs::rename(&tmp, full).map_err(s)?;
            barrier_dir(full).map_err(s)?;
            Ok(json!(data.len()))
        }
        "FileAppend" => {
            let data = r["data"].as_str().unwrap_or("");
            let created = !full.exists();
            let mut f = OpenOptions::new().append(true).create(true).open(full).map_err(s)?;
            f.write_all(data.as_bytes()).map_err(s)?;
            f.sync_data().map_err(s)?;
            if created {
                barrier_dir(full).map_err(s)?;
            }
            Ok(json!(data.len()))
        }
        "FileList" => {
            // HC-30: each name as its bytes, never through a lossy conversion
            let mut v: Vec<Vec<u8>> = fs::read_dir(full).map_err(s)?.flatten().map(|e| os_bytes(&e.file_name())).collect();
            v.sort();
            Ok(Value::Array(v.iter().map(|b| bytes_json(b)).collect()))
        }
        "FileLstat" => lstat_json(full),
        "FileReadlink" => fs::read_link(full).map(|t| bytes_json(&os_bytes(t.as_os_str()))).map_err(s),
        k => Err(format!("unknown file request `{k}`")),
    }
}

impl Adapter for FileAdapter {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let res = request_path(r).and_then(|p| self.resolve(&p)).and_then(|full| perform_file(self.kind, &full, r));
        match res {
            Ok(v) => EffectResult::ok(v),
            Err(e) => EffectResult::failed(e),
        }
    }
}

/// PTY request kinds and their result types:
/// `PtyOpen(req, argv: List[text], rows: u64, cols: u64)` -> u64 session id;
/// `PtyWrite(req, id: u64, data: text)` -> u64 bytes written;
/// `PtyRead(req, id: u64, max: u64)` -> text: the bytes available now
/// (waiting up to 10 ms for the first), decoded as UTF-8 with replacement;
/// `failed` with `eof` once the child closed the terminal;
/// `PtyClose(req, id: u64)` -> u64 exit status (0 when killed).
/// Timing reaches the program only through these logged results.
#[cfg(unix)]
pub mod pty {
    use super::*;
    use std::collections::BTreeMap;
    use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command, Stdio};

    pub const PTY_KINDS: [&str; 5] = ["PtyOpen", "PtyWrite", "PtyRead", "PtyResize", "PtyClose"];

    struct Session {
        master: OwnedFd,
        child: Child,
    }

    /// All PTY sessions of one host; each request kind is a view of it.
    #[derive(Default)]
    pub struct Ptys {
        next: u64,
        sessions: BTreeMap<u64, Session>,
    }

    impl Drop for Ptys {
        fn drop(&mut self) {
            for (_, mut s) in std::mem::take(&mut self.sessions) {
                let _ = s.child.kill();
                let _ = s.child.wait();
            }
        }
    }

    pub struct PtyAdapter {
        kind: &'static str,
        ptys: std::rc::Rc<std::cell::RefCell<Ptys>>,
    }

    pub fn pty_adapters() -> Vec<PtyAdapter> {
        let ptys = std::rc::Rc::new(std::cell::RefCell::new(Ptys::default()));
        PTY_KINDS.iter().map(|k| PtyAdapter { kind: k, ptys: ptys.clone() }).collect()
    }

    fn open(argv: &[String], rows: u16, cols: u16) -> Result<Session, String> {
        if argv.is_empty() {
            return Err("empty argv".into());
        }
        let (mut m, mut s) = (0, 0);
        let mut ws = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
        // SAFETY: openpty writes two descriptors through valid pointers.
        if unsafe { libc::openpty(&mut m, &mut s, std::ptr::null_mut(), std::ptr::null_mut(), &mut ws) } != 0 {
            return Err(std::io::Error::last_os_error().to_string());
        }
        // SAFETY: openpty returned two descriptors that we now own.
        let (master, slave) = unsafe { (OwnedFd::from_raw_fd(m), OwnedFd::from_raw_fd(s)) };
        // SAFETY: fcntl on a descriptor we own.
        unsafe {
            let fl = libc::fcntl(master.as_raw_fd(), libc::F_GETFL);
            libc::fcntl(master.as_raw_fd(), libc::F_SETFL, fl | libc::O_NONBLOCK);
            libc::fcntl(master.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC);
        }
        let io = |fd: &OwnedFd| fd.try_clone().map(Stdio::from).map_err(|e| e.to_string());
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]).stdin(io(&slave)?).stdout(io(&slave)?).stderr(io(&slave)?).env_clear().env("TERM", "xterm-256color");
        // SAFETY: only async-signal-safe calls run between fork and exec.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                libc::ioctl(0, libc::TIOCSCTTY as _, 0);
                // std only resets SIGPIPE. A host started from a backgrounded
                // non-interactive shell inherits SIGINT and SIGQUIT as ignored
                // (POSIX 2.11), and an ignored disposition survives exec, so
                // the terminal's Interrupt (^C) would never reach the job.
                // The pty session is a fresh login-like session: defaults.
                libc::signal(libc::SIGINT, libc::SIG_DFL);
                libc::signal(libc::SIGQUIT, libc::SIG_DFL);
                libc::signal(libc::SIGTSTP, libc::SIG_DFL);
                // the host loop blocks SIGTERM and SIGINT for its stop thread;
                // a blocked mask survives exec, and the shell must not inherit it
                let mut none: libc::sigset_t = std::mem::zeroed();
                libc::sigemptyset(&mut none);
                libc::sigprocmask(libc::SIG_SETMASK, &none, std::ptr::null_mut());
                Ok(())
            });
        }
        let child = cmd.spawn().map_err(|e| e.to_string())?;
        drop(slave);
        Ok(Session { master, child })
    }

    fn read_now(fd: &OwnedFd, max: usize) -> Result<Vec<u8>, String> {
        let mut pfd = libc::pollfd { fd: fd.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: one valid pollfd.
        unsafe { libc::poll(&mut pfd, 1, 10) };
        let mut out = vec![];
        let mut buf = [0u8; 4096];
        while out.len() < max {
            let want = buf.len().min(max - out.len());
            // SAFETY: buf has at least `want` writable bytes.
            let n = unsafe { libc::read(fd.as_raw_fd(), buf.as_mut_ptr().cast(), want) };
            if n > 0 {
                out.extend_from_slice(&buf[..n as usize]);
                continue;
            }
            if n == 0 {
                if out.is_empty() {
                    return Err("eof".into());
                }
                break;
            }
            let e = std::io::Error::last_os_error();
            match e.raw_os_error() {
                Some(libc::EAGAIN) => break,
                Some(libc::EIO) if out.is_empty() && pfd.revents & libc::POLLIN == 0 => return Err("eof".into()),
                Some(libc::EIO) if out.is_empty() => return Err("eof".into()),
                Some(libc::EIO) => break,
                _ => return Err(e.to_string()),
            }
        }
        Ok(out)
    }

    impl Adapter for PtyAdapter {
        fn kind(&self) -> &'static str {
            self.kind
        }
        /// HC-22: session ids continue from `floor`.
        fn restart(&mut self, floor: u64) {
            let mut p = self.ptys.borrow_mut();
            p.next = p.next.max(floor.saturating_sub(1));
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            let mut p = self.ptys.borrow_mut();
            let id = r["id"].as_u64().unwrap_or(0);
            let res: Result<Value, String> = match self.kind {
                "PtyOpen" => {
                    let argv: Vec<String> = r["argv"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect();
                    let rows = r["rows"].as_u64().unwrap_or(24).min(u16::MAX as u64) as u16;
                    let cols = r["cols"].as_u64().unwrap_or(80).min(u16::MAX as u64) as u16;
                    open(&argv, rows, cols).map(|s| {
                        p.next += 1;
                        let id = p.next;
                        p.sessions.insert(id, s);
                        json!(id)
                    })
                }
                "PtyWrite" => match p.sessions.get(&id) {
                    None => Err(format!("no pty {id}")),
                    Some(s) => {
                        let data = r["data"].as_str().unwrap_or("").as_bytes();
                        // SAFETY: data is a valid byte slice of that length.
                        let n = unsafe { libc::write(s.master.as_raw_fd(), data.as_ptr().cast(), data.len()) };
                        if n < 0 { Err(std::io::Error::last_os_error().to_string()) } else { Ok(json!(n)) }
                    }
                },
                "PtyRead" => match p.sessions.get(&id) {
                    None => Err(format!("no pty {id}")),
                    Some(s) => read_now(&s.master, r["max"].as_u64().unwrap_or(65536) as usize).map(|b| json!(String::from_utf8_lossy(&b))),
                },
                "PtyResize" => match p.sessions.get(&id) {
                    None => Err(format!("no pty {id}")),
                    Some(s) => {
                        let rows = r["rows"].as_u64().unwrap_or(24).min(u16::MAX as u64) as u16;
                        let cols = r["cols"].as_u64().unwrap_or(80).min(u16::MAX as u64) as u16;
                        let ws = libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 };
                        // SAFETY: TIOCSWINSZ reads one winsize from a valid pointer.
                        if unsafe { libc::ioctl(s.master.as_raw_fd(), libc::TIOCSWINSZ as _, &ws) } < 0 {
                            Err(std::io::Error::last_os_error().to_string())
                        } else {
                            Ok(json!(0))
                        }
                    }
                },
                "PtyClose" => match p.sessions.remove(&id) {
                    None => Err(format!("no pty {id}")),
                    Some(mut s) => {
                        drop(s.master);
                        let _ = s.child.kill();
                        s.child.wait().map(|st| json!(st.code().unwrap_or(0).max(0))).map_err(|e| e.to_string())
                    }
                },
                k => Err(format!("unknown pty request `{k}`")),
            };
            match res {
                Ok(v) => EffectResult::ok(v),
                Err(e) => EffectResult::failed(e),
            }
        }
    }
}

/// W1i: random bytes (HC-10): `Random(req, n)` -> text, n bytes (at most
/// 4096) from the OS CSPRNG as lowercase hex. The bytes are an input: they
/// are logged in the result event, so replay reads them from D.
pub struct RandomAdapter;

impl Adapter for RandomAdapter {
    fn kind(&self) -> &'static str {
        "Random"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let n = r["n"].as_u64().unwrap_or(32).min(4096) as usize;
        let mut b = vec![0u8; n];
        match File::open("/dev/urandom").and_then(|mut f| std::io::Read::read_exact(&mut f, &mut b)) {
            Ok(()) => EffectResult::ok(json!(b.iter().map(|x| format!("{x:02x}")).collect::<String>())),
            Err(e) => EffectResult::failed(e.to_string()),
        }
    }
}

/// HC-2, HC-32: the wall clock as a request: `WallClock(req)` -> u64,
/// nanoseconds since the Unix epoch. The admitter logs the reading in the
/// result event, so replay reads it from D and no clock.
pub struct WallClockAdapter;

impl Adapter for WallClockAdapter {
    fn kind(&self) -> &'static str {
        "WallClock"
    }
    fn perform(&mut self, _r: &Value) -> EffectResult {
        match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => EffectResult::ok(json!(d.as_nanos() as u64)),
            Err(e) => EffectResult::failed(format!("the clock is before the Unix epoch: {e}")),
        }
    }
}

/// HC-32: the UTC offset of the host's time zone at an instant:
/// `LocalOffset(req, at: i64)` -> i64 seconds east of UTC at `at` seconds
/// since the Unix epoch, from localtime_r(3) (the zone of `TZ`, else
/// `/etc/localtime`). The zone's rules are read when the adapter is made,
/// before the sandbox is entered, and each answer is logged in its result.
pub struct LocalOffsetAdapter;

/// The offset adapter, with the zone's rules read now (tzset(3)).
pub fn local_offset_adapter() -> LocalOffsetAdapter {
    #[cfg(unix)]
    {
        unsafe extern "C" {
            fn tzset();
        }
        // SAFETY: tzset reads TZ and the zone file into libc's own state.
        unsafe { tzset() };
    }
    LocalOffsetAdapter
}

/// HC-32: the payload of `host.Zone`: the value of `TZ` (`null` when it is
/// not set), the zone file the C library reads for it (`""` for a POSIX
/// rule such as `EST5EDT,M3.2.0,M11.1.0`, which names no file), and the
/// SHA-256 of that file's bytes (`""` without one). Keys are sorted.
pub fn zone_record() -> String {
    let tz = std::env::var("TZ").ok();
    let file = match tz.as_deref().map(|t| t.strip_prefix(':').unwrap_or(t)) {
        None => "/etc/localtime".to_string(),
        Some("") => String::new(),
        Some(t) if t.starts_with('/') => t.to_string(),
        Some(t) => {
            let f = format!("/usr/share/zoneinfo/{t}");
            if Path::new(&f).is_file() { f } else { String::new() }
        }
    };
    let sha = if file.is_empty() {
        String::new()
    } else {
        use sha2::{Digest, Sha256};
        fs::read(&file).map(|b| Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect()).unwrap_or_default()
    };
    json!({"file": file, "sha256": sha, "tz": tz}).to_string()
}

/// The offset east of UTC at `at`, in seconds.
#[cfg(unix)]
pub fn local_offset(at: i64) -> Result<i64, String> {
    let t: libc::time_t = at.try_into().map_err(|_| format!("{at} is outside time_t"))?;
    // SAFETY: localtime_r writes one tm through a valid pointer.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    if unsafe { libc::localtime_r(&t, &mut tm) }.is_null() {
        return Err(format!("no local time for {at}: {}", std::io::Error::last_os_error()));
    }
    Ok(tm.tm_gmtoff as i64)
}
#[cfg(not(unix))]
pub fn local_offset(_at: i64) -> Result<i64, String> {
    Err("LocalOffset needs a Unix host".into())
}

impl Adapter for LocalOffsetAdapter {
    fn kind(&self) -> &'static str {
        "LocalOffset"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let Some(at) = r["at"].as_i64() else { return EffectResult::failed("LocalOffset needs `at`, seconds since the Unix epoch") };
        match local_offset(at) {
            Ok(o) => EffectResult::ok(json!(o)),
            Err(e) => EffectResult::failed(e),
        }
    }
}

/// HC-33: under the `names` grant, a user or group id to its name:
/// `UserName(req, id)` and `GroupName(req, id)` -> the name, as its exact
/// bytes (HC-10a), from the first line of `/etc/passwd` or `/etc/group`
/// (the `files` source of the Name Service Switch) whose third field is
/// the id; `failed` when no line has it.
pub const NAME_KINDS: [&str; 2] = ["UserName", "GroupName"];

pub struct NameAdapter {
    kind: &'static str,
    db: PathBuf,
}

/// The two name adapters, reading the system's account databases.
pub fn name_adapters() -> Vec<NameAdapter> {
    name_adapters_from(Path::new(ACCOUNT_DBS[0]), Path::new(ACCOUNT_DBS[1]))
}

/// The account databases the `names` grant reads (HC-33, HC-5a).
pub const ACCOUNT_DBS: [&str; 2] = ["/etc/passwd", "/etc/group"];

/// The name adapters with other databases (tests).
pub fn name_adapters_from(passwd: &Path, group: &Path) -> Vec<NameAdapter> {
    vec![NameAdapter { kind: "UserName", db: passwd.to_path_buf() }, NameAdapter { kind: "GroupName", db: group.to_path_buf() }]
}

/// The name of `id` in a database of colon-separated lines whose first
/// field is the name and third the id.
pub fn lookup_name(db: &[u8], id: u64) -> Option<Vec<u8>> {
    let want = id.to_string();
    db.split(|b| *b == b'\n').find_map(|l| {
        let f: Vec<&[u8]> = l.split(|b| *b == b':').collect();
        (f.len() >= 3 && f[2] == want.as_bytes()).then(|| f[0].to_vec())
    })
}

impl Adapter for NameAdapter {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let what = if self.kind == "UserName" { "user" } else { "group" };
        let Some(id) = r["id"].as_u64() else { return EffectResult::failed(format!("{} needs `id`, a {what} id", self.kind)) };
        match fs::read(&self.db) {
            Ok(b) => match lookup_name(&b, id) {
                Some(n) => EffectResult::ok(bytes_json(&n)),
                None => EffectResult::failed(format!("no {what} with id {id} in {}", self.db.display())),
            },
            Err(e) => EffectResult::failed(format!("{}: {e}", self.db.display())),
        }
    }
}

/// W1i: content-addressed blobs under a granted root (files and media of
/// the chat app): `BlobPut(req, data)` -> text digest (SHA-256 of the data,
/// written atomically to `<root>/<digest>`); `BlobGet(req, digest)` -> text.
pub const BLOB_KINDS: [&str; 2] = ["BlobPut", "BlobGet"];

pub struct BlobAdapter {
    kind: &'static str,
    root: PathBuf,
}

pub fn blob_adapters(root: &Path) -> Vec<BlobAdapter> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    BLOB_KINDS.iter().map(|k| BlobAdapter { kind: k, root: root.clone() }).collect()
}

impl Adapter for BlobAdapter {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let res: Result<Value, String> = match self.kind {
            "BlobPut" => {
                let data = r["data"].as_str().unwrap_or("");
                let d = crate::interp::d_sha(data);
                let full = self.root.join(&d);
                if full.exists() {
                    Ok(json!(d))
                } else {
                    let tmp = self.root.join(format!(".{d}.tmp"));
                    (|| -> std::io::Result<()> {
                        let mut f = File::create(&tmp)?;
                        f.write_all(data.as_bytes())?;
                        f.sync_data()?;
                        fs::rename(&tmp, &full)?;
                        barrier_dir(&full)
                    })()
                    .map(|_| json!(d))
                    .map_err(|e| e.to_string())
                }
            }
            "BlobGet" => {
                let d = r["digest"].as_str().unwrap_or("");
                if d.len() != 64 || !d.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
                    Err(format!("`{d}` is not a digest"))
                } else {
                    fs::read_to_string(self.root.join(d)).map(|s| json!(s)).map_err(|e| e.to_string())
                }
            }
            k => Err(format!("unknown blob request `{k}`")),
        };
        match res {
            Ok(v) => EffectResult::ok(v),
            Err(e) => EffectResult::failed(e),
        }
    }
}

/// The bytes a request gives as `text` (its UTF-8 encoding) or as a
/// `List[i64]` of 0..255 (HC-10a); anything else fails the request.
pub fn request_bytes(v: &Value) -> Result<Vec<u8>, String> {
    match v {
        Value::Null => Ok(vec![]),
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        Value::Array(xs) => xs.iter().map(|x| x.as_u64().filter(|b| *b <= 255).map(|b| b as u8).ok_or_else(|| "byte outside 0..255".to_string())).collect(),
        _ => Err("data is neither text nor bytes".into()),
    }
}

/// W1i network adapter (TCP), the transport under P4 remote channels; its
/// contract is HC-23 of Appendix E. Authority (HC-5): only the `host:port`
/// addresses granted with `net`, by their exact text.
/// `NetConnect(req, addr)` -> u64 id (waits at most NET_CONNECT_MS);
/// `NetListen(req, addr)` -> u64 id; `NetAccept(req, id)` -> u64 id of a new
/// connection, or failed `none` when nothing is waiting; `NetSend(req, id,
/// data: text or List[i64])` -> u64 bytes handed to the connection, which
/// writes them on its own thread under NET_WRITE_MS, so a send never waits
/// for the peer; `NetRecv(req, id, max)` -> List[i64], the bytes received
/// now (possibly none), failed `eof` when the peer closed and nothing is
/// left; `NetClose(req, id)` -> u64 0, closing once the bytes handed over
/// are written. Every byte received is an input logged in D exactly as it
/// arrived (HC-10, HC-10a). Ids are never reused within D (HC-22).
pub mod net {
    use crate::host::{Adapter, EffectResult};
    use serde_json::{Value, json};
    use std::collections::{BTreeMap, VecDeque};
    use std::io::{ErrorKind, Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream, ToSocketAddrs};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    pub const NET_KINDS: [&str; 6] = ["NetConnect", "NetListen", "NetAccept", "NetSend", "NetRecv", "NetClose"];
    /// HC-23: the longest NetConnect waits for the peer (the one wait of the adapter).
    pub const NET_CONNECT_MS: u64 = 5_000;
    /// HC-23: the longest a connection waits for its peer to take bytes handed to it.
    pub const NET_WRITE_MS: u64 = 30_000;
    /// HC-23: the most bytes handed to one connection and not yet written.
    pub const NET_QUEUE_MAX: usize = 1_000_000;
    /// HC-10a: the most bytes one NetRecv delivers: N9 SIZE_LIMIT, so the
    /// delivered `List[i64]` can always be built.
    pub const NET_RECV_MAX: usize = crate::interp::SIZE_LIMIT;

    #[derive(Default)]
    struct Out {
        buf: VecDeque<u8>,
        closing: bool,
        failed: bool,
    }

    type OutQ = Arc<(Mutex<Out>, Condvar)>;

    struct Conn {
        s: TcpStream,
        out: OutQ,
        /// HD-3: the start of a UTF-8 character a read ended inside, carried
        /// to the next read when the module takes received bytes as text
        carry: Vec<u8>,
        /// a read error met after bytes were read: those bytes are
        /// delivered first, and the error is the next read's result
        error: Option<String>,
    }

    enum Sock {
        L(TcpListener),
        S(Conn),
    }

    #[derive(Default)]
    pub struct Nets {
        next: u64,
        socks: BTreeMap<u64, Sock>,
        grants: Vec<String>,
        /// HD-3: the module takes `NetRecvDone` data as text
        text: bool,
    }

    pub struct NetAdapter {
        kind: &'static str,
        nets: std::rc::Rc<std::cell::RefCell<Nets>>,
    }

    pub fn net_adapters(grants: &[String]) -> Vec<NetAdapter> {
        let nets = std::rc::Rc::new(std::cell::RefCell::new(Nets { grants: grants.to_vec(), ..Default::default() }));
        NET_KINDS.iter().map(|k| NetAdapter { kind: k, nets: nets.clone() }).collect()
    }

    /// Wait until `s` can take bytes, at most `ms` (unix: poll; elsewhere a short sleep).
    pub(crate) fn wait_writable(s: &TcpStream, ms: u64) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let mut p = libc::pollfd { fd: s.as_raw_fd(), events: libc::POLLOUT, revents: 0 };
            // SAFETY: one valid pollfd.
            unsafe { libc::poll(&mut p, 1, ms.min(1000) as i32) };
        }
        #[cfg(not(unix))]
        {
            let _ = s;
            std::thread::sleep(Duration::from_millis(ms.min(2)));
        }
    }

    /// The writer of one connection: writes the queued bytes in order. A
    /// peer that takes no byte for NET_WRITE_MS while bytes wait, or a write
    /// error, fails the connection: it is shut down and later requests on
    /// it fail with `closed`. After NetClose it closes once the queue is empty.
    fn writer(mut s: TcpStream, q: OutQ) {
        let (m, cv) = &*q;
        loop {
            let chunk: Vec<u8> = {
                let mut g = m.lock().unwrap();
                while g.buf.is_empty() && !g.closing && !g.failed {
                    g = cv.wait(g).unwrap();
                }
                if g.failed || g.buf.is_empty() {
                    break;
                }
                g.buf.iter().take(65536).copied().collect()
            };
            let mut done = 0;
            let mut last = Instant::now();
            let mut failed = false;
            while done < chunk.len() {
                match s.write(&chunk[done..]) {
                    Ok(0) => failed = true,
                    Ok(n) => {
                        done += n;
                        last = Instant::now();
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::Interrupted => {
                        let left = Duration::from_millis(NET_WRITE_MS).saturating_sub(last.elapsed());
                        if left.is_zero() {
                            failed = true;
                        } else {
                            wait_writable(&s, left.as_millis() as u64);
                        }
                    }
                    Err(_) => failed = true,
                }
                if failed {
                    break;
                }
            }
            let mut g = m.lock().unwrap();
            if failed {
                g.failed = true;
                g.buf.clear();
                break;
            }
            g.buf.drain(..done);
        }
        let _ = s.shutdown(Shutdown::Both);
    }

    fn conn(s: TcpStream) -> Result<Conn, String> {
        s.set_nodelay(true).ok();
        s.set_nonblocking(true).map_err(|e| e.to_string())?;
        let w = s.try_clone().map_err(|e| e.to_string())?;
        let out: OutQ = Default::default();
        let q = out.clone();
        std::thread::spawn(move || writer(w, q));
        Ok(Conn { s, out, carry: vec![], error: None })
    }

    impl Nets {
        fn add(&mut self, s: Sock) -> u64 {
            self.next += 1;
            self.socks.insert(self.next, s);
            self.next
        }
        fn granted(&self, addr: &str) -> Result<(), String> {
            if self.grants.iter().any(|g| g == addr) { Ok(()) } else { Err(format!("address `{addr}` is not granted")) }
        }
    }

    /// The length of the UTF-8 character whose first byte is `b`.
    fn utf8_len(b: u8) -> usize {
        match b {
            0xF0.. => 4,
            0xE0.. => 3,
            0xC0.. => 2,
            _ => 1,
        }
    }

    /// Read at most `want` bytes into `out` without waiting: Ok(true) at
    /// the end of the stream.
    fn read_some(s: &mut TcpStream, out: &mut Vec<u8>, want: usize) -> Result<bool, String> {
        let mut buf = vec![0u8; want.min(65536)];
        let mut got = 0;
        while got < want {
            let k = (want - got).min(buf.len());
            match s.read(&mut buf[..k]) {
                Ok(0) => return Ok(true),
                Ok(n) => {
                    out.extend_from_slice(&buf[..n]);
                    got += n;
                }
                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Ok(false)
    }

    fn recv(c: &mut Conn, max: usize, text: bool) -> Result<Value, String> {
        // HD-3: bytes carried to the next read are delivered before the
        // connection's failure, so nothing received is lost
        if c.out.0.lock().unwrap().failed {
            return if c.carry.is_empty() { Err("closed".into()) } else { Ok(json!(std::mem::take(&mut c.carry))) };
        }
        if let Some(e) = c.error.take() {
            return Err(e);
        }
        let mut out: Vec<u8> = std::mem::take(&mut c.carry);
        let want = max.saturating_sub(out.len());
        let mut r = read_some(&mut c.s, &mut out, want);
        // HD-3: taken as text, a read does not end inside a character: it
        // reads past max, by at most 3 bytes, to complete the last one; a
        // character whose bytes have not arrived waits for the next read,
        // and at the end of the stream it is delivered as it is
        let cut = match std::str::from_utf8(&out) {
            Err(e) if e.error_len().is_none() => Some(e.valid_up_to()),
            _ => None,
        };
        if text
            && r == Ok(false)
            && let Some(start) = cut
        {
            let need = utf8_len(out[start]).saturating_sub(out.len() - start);
            r = read_some(&mut c.s, &mut out, need);
            if r == Ok(false) && std::str::from_utf8(&out[start..]).is_err() {
                c.carry = out.split_off(start);
            }
        }
        match r {
            // HC-10a: exactly the bytes received, in order, as a list of 0..255
            Ok(true) if out.is_empty() => Err("eof".into()),
            Err(e) if out.is_empty() => Err(e),
            Err(e) => {
                c.error = Some(e);
                Ok(json!(out))
            }
            Ok(_) => Ok(json!(out)),
        }
    }

    fn send(c: &mut Conn, data: &Value) -> Result<Value, String> {
        let d = super::request_bytes(data)?;
        let (m, cv) = &*c.out;
        let mut g = m.lock().unwrap();
        if g.failed || g.closing {
            return Err("closed".into());
        }
        if g.buf.len() + d.len() > NET_QUEUE_MAX {
            return Err("send buffer full".into());
        }
        g.buf.extend(d.iter().copied());
        cv.notify_all();
        Ok(json!(d.len()))
    }

    impl Adapter for NetAdapter {
        fn kind(&self) -> &'static str {
            self.kind
        }
        /// HC-22: ids continue from `floor`, so no id issued before a restart is reused.
        fn restart(&mut self, floor: u64) {
            let mut n = self.nets.borrow_mut();
            n.next = n.next.max(floor.saturating_sub(1));
        }
        fn text(&mut self) {
            if self.kind == "NetRecv" {
                self.nets.borrow_mut().text = true;
            }
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            let mut n = self.nets.borrow_mut();
            let text = n.text;
            let id = r["id"].as_u64().unwrap_or(0);
            let addr = r["addr"].as_str().unwrap_or("").to_string();
            let res: Result<Value, String> = match self.kind {
                "NetConnect" => n.granted(&addr).and_then(|_| {
                    let sa = addr.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
                    let s = TcpStream::connect_timeout(&sa, Duration::from_millis(NET_CONNECT_MS)).map_err(|e| if e.kind() == ErrorKind::TimedOut { "timed_out".to_string() } else { e.to_string() })?;
                    let c = conn(s)?;
                    Ok(json!(n.add(Sock::S(c))))
                }),
                "NetListen" => n.granted(&addr).and_then(|_| {
                    let l = TcpListener::bind(&addr).map_err(|e| e.to_string())?;
                    l.set_nonblocking(true).map_err(|e| e.to_string())?;
                    Ok(json!(n.add(Sock::L(l))))
                }),
                "NetAccept" => match n.socks.get(&id) {
                    Some(Sock::L(l)) => match l.accept() {
                        Ok((s, _)) => conn(s).map(|c| json!(n.add(Sock::S(c)))),
                        Err(e) if e.kind() == ErrorKind::WouldBlock => Err("none".into()),
                        Err(e) => Err(e.to_string()),
                    },
                    _ => Err(format!("no listener {id}")),
                },
                "NetSend" => match n.socks.get_mut(&id) {
                    Some(Sock::S(c)) => send(c, &r["data"]),
                    _ => Err(format!("no connection {id}")),
                },
                "NetRecv" => match n.socks.get_mut(&id) {
                    Some(Sock::S(c)) => recv(c, (r["max"].as_u64().unwrap_or(65536) as usize).min(NET_RECV_MAX), text),
                    _ => Err(format!("no connection {id}")),
                },
                "NetClose" => match n.socks.remove(&id) {
                    Some(Sock::S(c)) => {
                        let (m, cv) = &*c.out;
                        m.lock().unwrap().closing = true;
                        cv.notify_all();
                        // HD-3: the bytes of a character carried for a read
                        // that never came are not delivered; err lists
                        // them, so D records them
                        if !c.carry.is_empty() {
                            return EffectResult { status: crate::host::Status::Delivered, data: Some(json!(0)), error: format!("closed with {} received bytes of an incomplete UTF-8 character not delivered: {:?}", c.carry.len(), c.carry) };
                        }
                        Ok(json!(0))
                    }
                    Some(Sock::L(_)) => Ok(json!(0)),
                    None => Err(format!("no socket {id}")),
                },
                k => Err(format!("unknown net request `{k}`")),
            };
            match res {
                Ok(v) => EffectResult::ok(v),
                Err(e) => EffectResult::failed(e),
            }
        }
    }
}

/// P3 (HC-4, E.9): foreign code as WebAssembly. `WasmCall(req, code,
/// func, args: List[i64])` -> List[i64]. `code` is the SHA-256 digest of
/// the module bytes, stored under the granted `wasm DIR` (the BlobPut
/// layout), so D names the exact code that ran. Each call gets a fresh
/// instance with no imports (nothing to reach but its own memory), runs
/// under a fixed fuel budget chosen by the engine (`FOREIGN_FUEL`), and its
/// result or trap is the logged `WasmCallDone` event: replay reads it and
/// never re-executes (T32). Parameters and results are i64 values: i32 and
/// i64 as integers, f32 and f64 as their IEEE bit patterns. Floats run only
/// when the HC-13 probe admitted them on this host, and every module is
/// first rewritten so a NaN result is the canonical NaN on every machine
/// (see `wasm_float`); a float result is never a NaN payload of the host.
/// SIMD is not compiled in (wasmi's `simd` feature is off).
pub const FOREIGN_FUEL: u64 = 100_000_000;

pub struct WasmAdapter {
    root: PathBuf,
    engine: wasmi::Engine,
    /// the `host.Exact` payload of the probe that decided `floats`
    pub exact: String,
}

fn wasm_engine(floats: bool) -> wasmi::Engine {
    let mut c = wasmi::Config::default();
    c.consume_fuel(true).floats(floats);
    wasmi::Engine::new(&c)
}

pub fn wasm_adapter(root: &Path) -> WasmAdapter {
    let (floats, exact) = crate::wasm_float::probe(&wasm_engine(true));
    WasmAdapter { root: root.canonicalize().unwrap_or_else(|_| root.to_path_buf()), engine: wasm_engine(floats), exact }
}

impl WasmAdapter {
    fn call(&self, r: &Value) -> Result<Value, String> {
        use wasmi::{Linker, Module, Store, Val, ValType};
        let d = r["code"].as_str().unwrap_or("");
        if d.len() != 64 || !d.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
            return Err(format!("`{d}` is not a module digest"));
        }
        let bytes = fs::read(self.root.join(d)).map_err(|e| e.to_string())?;
        let got = {
            use sha2::{Digest, Sha256};
            Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>()
        };
        if got != d {
            return Err(format!("module bytes hash to {got}, not {d}"));
        }
        let bytes = crate::wasm_float::canonicalize(&bytes)?;
        let module = Module::new(&self.engine, &bytes[..]).map_err(|e| format!("invalid module: {e}"))?;
        let mut store = Store::new(&self.engine, ());
        store.set_fuel(FOREIGN_FUEL).map_err(|e| e.to_string())?;
        let inst = Linker::<()>::new(&self.engine).instantiate_and_start(&mut store, &module).map_err(|e| format!("instantiate: {e}"))?;
        let fname = r["func"].as_str().unwrap_or("");
        let f = inst.get_func(&store, fname).ok_or(format!("no export `{fname}`"))?;
        let ty = f.ty(&store);
        let args: Vec<i64> = r["args"].as_array().into_iter().flatten().filter_map(|x| x.as_i64()).collect();
        if ty.params().len() != args.len() {
            return Err(format!("`{fname}` takes {} arguments, given {}", ty.params().len(), args.len()));
        }
        let mut ps = vec![];
        for (t, a) in ty.params().iter().zip(&args) {
            ps.push(match t {
                ValType::I32 => Val::I32(i32::try_from(*a).map_err(|_| format!("{a} does not fit i32"))?),
                ValType::I64 => Val::I64(*a),
                ValType::F32 => Val::F32(wasmi::F32::from_bits(u32::try_from(*a).map_err(|_| format!("{a} is not an f32 bit pattern"))?)),
                ValType::F64 => Val::F64(wasmi::F64::from_bits(*a as u64)),
                t => return Err(format!("parameter type {t:?} is not a number")),
            });
        }
        let mut rs: Vec<Val> = vec![];
        for t in ty.results() {
            rs.push(match t {
                ValType::I32 => Val::I32(0),
                ValType::I64 => Val::I64(0),
                ValType::F32 => Val::F32(wasmi::F32::from_bits(0)),
                ValType::F64 => Val::F64(wasmi::F64::from_bits(0)),
                t => return Err(format!("result type {t:?} is not a number")),
            });
        }
        f.call(&mut store, &ps, &mut rs).map_err(|e| format!("trap: {e}"))?;
        Ok(json!(rs.iter().map(|v| match v {
            Val::I32(x) => *x as i64,
            Val::I64(x) => *x,
            Val::F32(x) => crate::wasm_float::canon_bits32(x.to_bits()) as i64,
            Val::F64(x) => crate::wasm_float::canon_bits64(x.to_bits()) as i64,
            _ => 0,
        }).collect::<Vec<i64>>()))
    }
}

impl Adapter for WasmAdapter {
    fn kind(&self) -> &'static str {
        "WasmCall"
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        match self.call(r) {
            Ok(v) => EffectResult::ok(v),
            Err(e) => EffectResult::failed(e),
        }
    }
}

/// P4 remote channels: program-to-program messages between hosts over TCP.
///
/// Frames are one JSON object per line: `{"body","from","id","kind"}` with
/// kind `msg`, `lossy` or `ack`. Each host listens on its own address (the
/// `from` it sends) and reaches only granted peers (HC-5).
///
/// - `ChanSend(req, peer, id, body, reliable)` -> u64 id. Reliable: the
///   result is `delivered` only after the peer acknowledged `id`, `timed_out`
///   after the engine's fixed deadline (`CHAN_DEADLINE_MS`), `unreachable`
///   when no connection could be made. The adapter resends within the
///   deadline (at-least-once); the receiver deduplicates by (from, id), so a
///   program sees each message once. Lossy (presence, typing, media
///   metadata): written once, `delivered` means written, never resent.
/// - `ChanRecv(req, max)` -> List[text], messages received so far, each
///   `from\tid\tbody`, in arrival order. That order is an input, logged in
///   D (HC-10), so replay needs no network.
///
/// Bodies are opaque text: end-to-end ciphertext passes through unchanged.
pub mod chan {
    use crate::host::{Adapter, EffectResult, Status};
    use serde_json::{Value, json};
    use std::collections::{BTreeSet, HashMap, VecDeque};
    use std::io::{BufRead, BufReader, Write};
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    pub const CHAN_KINDS: [&str; 2] = ["ChanSend", "ChanRecv"];
    pub const CHAN_DEADLINE_MS: u64 = 3000;

    #[derive(Default)]
    struct Shared {
        inbox: VecDeque<String>,
        seen: BTreeSet<(String, u64)>,
        acked: BTreeSet<(String, u64)>,
    }

    pub struct Chans {
        me: String,
        grants: Vec<String>,
        shared: Arc<(Mutex<Shared>, Condvar)>,
        out: HashMap<String, TcpStream>,
        tls: Option<(crate::tls::Identity, Arc<HashMap<String, [u8; 32]>>)>,
        tls_out: HashMap<String, std::sync::mpsc::Sender<String>>,
    }

    pub struct ChanAdapter {
        kind: &'static str,
        c: std::rc::Rc<std::cell::RefCell<Chans>>,
    }

    /// One received frame: record it, and give the ack to send back.
    fn handle_frame(line: &str, sh: &Arc<(Mutex<Shared>, Condvar)>) -> Option<String> {
        let Ok(f) = serde_json::from_str::<Value>(line) else { return None };
        let (from, id) = (f["from"].as_str().unwrap_or("").to_string(), f["id"].as_u64().unwrap_or(0));
        let kind = f["kind"].as_str().unwrap_or("");
        let (m, cv) = &**sh;
        let mut g = m.lock().unwrap();
        match kind {
            "ack" => {
                g.acked.insert((from, id));
                cv.notify_all();
                None
            }
            "msg" | "lossy" => {
                if g.seen.insert((from.clone(), id)) {
                    g.inbox.push_back(format!("{from}\t{id}\t{}", f["body"].as_str().unwrap_or("")));
                }
                drop(g);
                // the ack goes back on the same connection
                (kind == "msg").then(|| json!({"body": "", "from": f["to"], "id": id, "kind": "ack"}).to_string())
            }
            _ => None,
        }
    }

    fn serve_conn(s: TcpStream, sh: Arc<(Mutex<Shared>, Condvar)>) {
        let mut w = match s.try_clone() {
            Ok(w) => w,
            Err(_) => return,
        };
        for line in BufReader::new(s).lines().map_while(Result::ok) {
            if let Some(a) = handle_frame(&line, &sh) {
                let _ = writeln!(w, "{a}");
            }
        }
    }

    /// A TLS stream that can name the Ed25519 key its peer presented.
    trait TlsIo: std::io::Read + Write + Send {
        fn peer_key(&self) -> Option<[u8; 32]>;
    }
    impl TlsIo for rustls::StreamOwned<rustls::ServerConnection, TcpStream> {
        fn peer_key(&self) -> Option<[u8; 32]> {
            self.conn.peer_certificates().and_then(|c| c.first()).and_then(|c| crate::tls::ed25519_key_of(c))
        }
    }
    impl TlsIo for rustls::StreamOwned<rustls::ClientConnection, TcpStream> {
        fn peer_key(&self) -> Option<[u8; 32]> {
            self.conn.peer_certificates().and_then(|c| c.first()).and_then(|c| crate::tls::ed25519_key_of(c))
        }
    }

    /// One TLS connection, owned by one thread: it writes the frames queued
    /// on `rx` and reads frames, accepting a frame only when its `from` is
    /// pinned to the key this connection presented (authenticated sender).
    fn tls_loop(mut s: Box<dyn TlsIo>, rx: Option<std::sync::mpsc::Receiver<String>>, sh: Arc<(Mutex<Shared>, Condvar)>, pins: Arc<HashMap<String, [u8; 32]>>) {
        let mut buf: Vec<u8> = vec![];
        let mut b = [0u8; 8192];
        loop {
            if let Some(rx) = &rx {
                loop {
                    match rx.try_recv() {
                        Ok(l) => {
                            if s.write_all(format!("{l}\n").as_bytes()).and_then(|_| s.flush()).is_err() {
                                return;
                            }
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => break,
                        Err(_) => return,
                    }
                }
            }
            match std::io::Read::read(&mut s, &mut b) {
                Ok(0) => return,
                Ok(n) => {
                    buf.extend_from_slice(&b[..n]);
                    while let Some(i) = buf.iter().position(|&c| c == b'\n') {
                        let raw: Vec<u8> = buf.drain(..=i).collect();
                        let line = String::from_utf8_lossy(&raw[..raw.len() - 1]).to_string();
                        let from = serde_json::from_str::<Value>(&line).ok().and_then(|f| f["from"].as_str().map(String::from));
                        let key = s.peer_key();
                        if !from.is_some_and(|fr| pins.get(&fr).is_some_and(|k| Some(*k) == key)) {
                            continue;
                        }
                        if let Some(a) = handle_frame(&line, &sh)
                            && s.write_all(format!("{a}\n").as_bytes()).and_then(|_| s.flush()).is_err()
                        {
                            return;
                        }
                    }
                }
                Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Err(_) => return,
            }
        }
    }

    const TLS_POLL: Duration = Duration::from_millis(20);

    /// As `chan_adapters`, over mutual TLS 1.3: `peers` are (address, pinned
    /// Ed25519 key); a connection whose certificate key is not pinned for its
    /// address fails the handshake, and frames claiming another sender are
    /// dropped (crate::tls).
    pub fn chan_adapters_tls(me: &str, peers: &[(String, [u8; 32])], id: &crate::tls::Identity) -> Result<Vec<ChanAdapter>, String> {
        let l = TcpListener::bind(me).map_err(|e| e.to_string())?;
        chan_adapters_tls_on(l, me, peers, id)
    }

    /// As `chan_adapters_tls`, on a listener the caller already bound (for example to
    /// port 0), so there is no window between choosing the address and listening on it.
    pub fn chan_adapters_tls_on(l: TcpListener, me: &str, peers: &[(String, [u8; 32])], id: &crate::tls::Identity) -> Result<Vec<ChanAdapter>, String> {
        let (xs, start) = chan_adapters_tls_deferred(l, me, peers, id)?;
        start();
        Ok(xs)
    }

    /// Starts the threads of a channel listener bound earlier. The host
    /// calls it after entering the sandbox (HC-5a): Landlock applies to
    /// the thread that enters it and the threads it starts afterwards, so
    /// an acceptor started before would run outside it.
    pub type Start = Box<dyn FnOnce() + Send>;

    /// `chan_adapters_tls_on` with its threads not started yet.
    pub fn chan_adapters_tls_deferred(l: TcpListener, me: &str, peers: &[(String, [u8; 32])], id: &crate::tls::Identity) -> Result<(Vec<ChanAdapter>, Start), String> {
        let shared: Arc<(Mutex<Shared>, Condvar)> = Default::default();
        let pins: Arc<HashMap<String, [u8; 32]>> = Arc::new(peers.iter().cloned().collect());
        let scfg = crate::tls::server_config(id, peers.iter().map(|p| p.1).collect())?;
        let (sh, pn) = (shared.clone(), pins.clone());
        let start: Start = Box::new(move || {
            std::thread::spawn(move || {
                for s in l.incoming().flatten() {
                    s.set_nodelay(true).ok();
                    s.set_read_timeout(Some(TLS_POLL)).ok();
                    let Ok(conn) = rustls::ServerConnection::new(scfg.clone()) else { continue };
                    let (sh, pn) = (sh.clone(), pn.clone());
                    std::thread::spawn(move || tls_loop(Box::new(rustls::StreamOwned::new(conn, s)), None, sh, pn));
                }
            });
        });
        let c = std::rc::Rc::new(std::cell::RefCell::new(Chans {
            me: me.into(),
            grants: peers.iter().map(|p| p.0.clone()).collect(),
            shared,
            out: HashMap::new(),
            tls: Some((id.clone(), pins)),
            tls_out: HashMap::new(),
        }));
        Ok((CHAN_KINDS.iter().map(|k| ChanAdapter { kind: k, c: c.clone() }).collect(), start))
    }

    /// Listen on `me` (a granted local address) and reach only `peers`.
    pub fn chan_adapters(me: &str, peers: &[String]) -> std::io::Result<Vec<ChanAdapter>> {
        let (xs, start) = chan_adapters_deferred(me, peers)?;
        start();
        Ok(xs)
    }

    /// `chan_adapters` with the listener bound and its threads not started yet.
    pub fn chan_adapters_deferred(me: &str, peers: &[String]) -> std::io::Result<(Vec<ChanAdapter>, Start)> {
        let l = TcpListener::bind(me)?;
        let shared: Arc<(Mutex<Shared>, Condvar)> = Default::default();
        let sh = shared.clone();
        let start: Start = Box::new(move || {
            std::thread::spawn(move || {
                for s in l.incoming().flatten() {
                    let sh = sh.clone();
                    std::thread::spawn(move || serve_conn(s, sh));
                }
            });
        });
        let c = std::rc::Rc::new(std::cell::RefCell::new(Chans { me: me.into(), grants: peers.to_vec(), shared, out: HashMap::new(), tls: None, tls_out: HashMap::new() }));
        Ok((CHAN_KINDS.iter().map(|k| ChanAdapter { kind: k, c: c.clone() }).collect(), start))
    }

    impl Chans {
        fn conn(&mut self, peer: &str) -> Result<&mut TcpStream, String> {
            if !self.out.contains_key(peer) {
                let sa = peer.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
                let s = TcpStream::connect_timeout(&sa, Duration::from_millis(1000)).map_err(|e| e.to_string())?;
                s.set_nodelay(true).ok();
                // acks for our messages come back on this connection
                if let Ok(r) = s.try_clone() {
                    let sh = self.shared.clone();
                    std::thread::spawn(move || serve_conn(r, sh));
                }
                self.out.insert(peer.into(), s);
            }
            Ok(self.out.get_mut(peer).unwrap())
        }

        fn write(&mut self, peer: &str, frame: &str) -> Result<(), String> {
            let Some((id, pins)) = self.tls.clone() else {
                return self.conn(peer).and_then(|s| writeln!(s, "{frame}").map_err(|e| e.to_string()));
            };
            if !self.tls_out.contains_key(peer) {
                let pin = *pins.get(peer).ok_or("peer has no pinned key")?;
                let cfg = crate::tls::client_config(&id, pin)?;
                let sa = peer.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
                let s = TcpStream::connect_timeout(&sa, Duration::from_millis(1000)).map_err(|e| e.to_string())?;
                s.set_nodelay(true).ok();
                s.set_read_timeout(Some(TLS_POLL)).ok();
                let name = rustls::pki_types::ServerName::try_from(crate::tls::SERVER_NAME).map_err(|e| e.to_string())?;
                let conn = rustls::ClientConnection::new(cfg, name).map_err(|e| e.to_string())?;
                let (tx, rx) = std::sync::mpsc::channel();
                let sh = self.shared.clone();
                std::thread::spawn(move || tls_loop(Box::new(rustls::StreamOwned::new(conn, s)), Some(rx), sh, pins));
                self.tls_out.insert(peer.into(), tx);
            }
            self.tls_out[peer].send(frame.to_string()).map_err(|e| e.to_string())
        }
    }

    impl Adapter for ChanAdapter {
        fn kind(&self) -> &'static str {
            self.kind
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            let mut c = self.c.borrow_mut();
            match self.kind {
                "ChanSend" => {
                    let peer = r["peer"].as_str().unwrap_or("").to_string();
                    if !c.grants.contains(&peer) {
                        return EffectResult::failed(format!("peer `{peer}` is not granted"));
                    }
                    let id = r["id"].as_u64().unwrap_or(0);
                    let reliable = r["reliable"].as_bool().unwrap_or(true);
                    let frame = json!({"body": r["body"].as_str().unwrap_or(""), "from": c.me, "id": id, "kind": if reliable { "msg" } else { "lossy" }, "to": peer}).to_string();
                    let start = Instant::now();
                    let deadline = Duration::from_millis(CHAN_DEADLINE_MS);
                    let mut wrote = false;
                    while start.elapsed() < deadline {
                        let sent = c.write(&peer, &frame);
                        match sent {
                            Ok(()) => wrote = true,
                            Err(_) => {
                                c.out.remove(&peer);
                                c.tls_out.remove(&peer);
                            }
                        }
                        if wrote && !reliable {
                            return EffectResult::ok(json!(id));
                        }
                        if wrote {
                            let (m, cv) = &*c.shared.clone();
                            let g = m.lock().unwrap();
                            let (g, _) = cv.wait_timeout_while(g, Duration::from_millis(500), |g| !g.acked.contains(&(peer.clone(), id))).unwrap();
                            if g.acked.contains(&(peer.clone(), id)) {
                                return EffectResult::ok(json!(id));
                            }
                        } else {
                            std::thread::sleep(Duration::from_millis(100));
                        }
                    }
                    let status = if wrote { Status::TimedOut } else { Status::Unreachable };
                    EffectResult { status, data: None, error: format!("no ack from {peer} for {id}") }
                }
                "ChanRecv" => {
                    let max = r["max"].as_u64().unwrap_or(256) as usize;
                    let (m, _) = &*c.shared;
                    let mut g = m.lock().unwrap();
                    let n = g.inbox.len().min(max);
                    let v: Vec<String> = g.inbox.drain(..n).collect();
                    EffectResult::ok(json!(v))
                }
                k => EffectResult::failed(format!("unknown channel request `{k}`")),
            }
        }
    }
}

/// W1i media: capture and playback adapters. A raw frame is never folded
/// into state and never written into D: the adapter holds it as a blob
/// addressed by its SHA-256 under the granted blob root, and D logs only
/// (track, seq, digest); the program stamps each frame with its own Tick
/// count, so replay is deterministic without re-capturing.
///
/// Capture devices are an enforcement-target detail; the contract is the
/// same on every target. The sources implemented here are portable:
/// `dir:NAME` reads the frames of `<root>/capture/NAME/` in canonical
/// (sorted) order, a recorded or device-fed stream; `pattern:N` is a
/// synthetic test source of N frames.
///
/// `MediaOpen(req, source)` -> u64 track id;
/// `MediaRead(req, id)` -> text `"<seq>:<digest>"`, failed `eof` at the end;
/// `MediaPlay(req, track, digest)` -> u64 bytes: the playback sink (λ,
/// output only) checks the blob against its digest and appends it to
/// `<root>/play/<track>`; a frame whose bytes do not match is not appended
/// (the result is failed).
pub mod media {
    use crate::host::{Adapter, EffectResult};
    use serde_json::{Value, json};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;

    pub const MEDIA_KINDS: [&str; 3] = ["MediaOpen", "MediaRead", "MediaPlay"];

    struct Track {
        frames: Vec<Vec<u8>>,
        seq: usize,
    }

    #[derive(Default)]
    pub struct Media {
        root: PathBuf,
        next: u64,
        tracks: BTreeMap<u64, Track>,
    }

    pub struct MediaAdapter {
        kind: &'static str,
        m: std::rc::Rc<std::cell::RefCell<Media>>,
    }

    pub fn media_adapters(root: &std::path::Path) -> Vec<MediaAdapter> {
        let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
        let m = std::rc::Rc::new(std::cell::RefCell::new(Media { root, ..Default::default() }));
        MEDIA_KINDS.iter().map(|k| MediaAdapter { kind: k, m: m.clone() }).collect()
    }

    fn hex(b: &[u8]) -> String {
        Sha256::digest(b).iter().map(|x| format!("{x:02x}")).collect()
    }

    fn safe_name(s: &str) -> bool {
        !s.is_empty() && s.len() <= 128 && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_' || c == b'.') && !s.starts_with('.')
    }

    impl Media {
        fn open(&mut self, source: &str) -> Result<u64, String> {
            let frames = if let Some(n) = source.strip_prefix("pattern:") {
                let n: usize = n.parse().map_err(|_| format!("`{source}`: pattern needs a frame count"))?;
                if n > 100_000 {
                    return Err("pattern: at most 100000 frames".into());
                }
                // an 8x8 grey ramp shifted by the frame number
                (0..n).map(|i| (0..64u32).map(|p| ((p * 4 + i as u32 * 7) % 256) as u8).collect()).collect()
            } else if let Some(d) = source.strip_prefix("dir:") {
                if !safe_name(d) {
                    return Err(format!("`{d}` is not a capture name"));
                }
                let dir = self.root.join("capture").join(d);
                let mut names: Vec<PathBuf> = fs::read_dir(&dir).map_err(|e| format!("{d}: {e}"))?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file()).collect();
                names.sort();
                names.iter().map(|p| fs::read(p).map_err(|e| e.to_string())).collect::<Result<_, _>>()?
            } else {
                return Err(format!("unknown media source `{source}` (dir:NAME or pattern:N)"));
            };
            self.next += 1;
            self.tracks.insert(self.next, Track { frames, seq: 0 });
            Ok(self.next)
        }
        fn store(&self, b: &[u8]) -> Result<String, String> {
            let d = hex(b);
            let full = self.root.join(&d);
            if !full.exists() {
                let tmp = self.root.join(format!(".{d}.tmp"));
                (|| -> std::io::Result<()> {
                    let mut f = File::create(&tmp)?;
                    f.write_all(b)?;
                    f.sync_data()?;
                    fs::rename(&tmp, &full)
                })()
                .map_err(|e| e.to_string())?;
            }
            Ok(d)
        }
        fn read(&mut self, id: u64) -> Result<String, String> {
            let t = self.tracks.get_mut(&id).ok_or_else(|| format!("no media track {id}"))?;
            let Some(f) = t.frames.get(t.seq).cloned() else { return Err("eof".into()) };
            t.seq += 1;
            let seq = t.seq;
            let d = self.store(&f)?;
            Ok(format!("{seq}:{d}"))
        }
        fn play(&self, track: &str, digest: &str) -> Result<u64, String> {
            if !safe_name(track) {
                return Err(format!("`{track}` is not a track name"));
            }
            if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()) {
                return Err(format!("`{digest}` is not a digest"));
            }
            let b = fs::read(self.root.join(digest)).map_err(|e| e.to_string())?;
            if hex(&b) != digest {
                return Err(format!("frame {digest} does not match its digest"));
            }
            let dir = self.root.join("play");
            fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let mut f = OpenOptions::new().create(true).append(true).open(dir.join(track)).map_err(|e| e.to_string())?;
            f.write_all(&b).map_err(|e| e.to_string())?;
            Ok(b.len() as u64)
        }
    }

    impl Adapter for MediaAdapter {
        fn kind(&self) -> &'static str {
            self.kind
        }
        /// HC-22: track ids continue from `floor`.
        fn restart(&mut self, floor: u64) {
            let mut m = self.m.borrow_mut();
            m.next = m.next.max(floor.saturating_sub(1));
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            let mut m = self.m.borrow_mut();
            let res = match self.kind {
                "MediaOpen" => m.open(r["source"].as_str().unwrap_or("")).map(|id| json!(id)),
                "MediaRead" => m.read(r["id"].as_u64().unwrap_or(0)).map(|s| json!(s)),
                "MediaPlay" => m.play(r["track"].as_str().unwrap_or(""), r["digest"].as_str().unwrap_or("")).map(|n| json!(n)),
                k => Err(format!("unknown media request `{k}`")),
            };
            match res {
                Ok(v) => EffectResult::ok(v),
                Err(e) => EffectResult::failed(e),
            }
        }
    }
}

/// The HTTP event host (Appendix E HC-24 to HC-26): `http HOST:PORT`.
///
/// The host frames requests, so it can serve connections concurrently and
/// apply limits and deadlines before any byte enters D. Each complete
/// request is one logged input `HttpRequest(id, method, target, version,
/// names, values, body)`; a request that breaks the syntax, a limit or a deadline
/// after its first byte is one `HttpMalformed(id, status, why)`. The id is
/// the seq of the input's own entry. The program answers with the effect
/// `HttpRespond(req, status, names, values, body)`. Replay reads requests
/// from D and never parses (HC-8, HC-10).
///
/// The syntax is RFC 9112 with the strict choices of `net.http`
/// (`stdlib/lib/net/http.rce`): [`parse_request`] and
/// `net.http.parse_message(b, true, false, false)` agree on every vector of
/// `stdlib/tests/net/http_checks.rce` and on mutations of them
/// (`tests/http_parse.rs`). Only the host applies the limits below, the
/// 505 for a version other than 1.x and the 501 for a transfer coding it
/// does not implement.
pub mod http {
    use crate::host::{Adapter, EffectResult};
    use serde_json::{Value, json};
    use std::collections::HashMap;
    use std::io::{ErrorKind, Read, Write};
    use std::net::{Shutdown, TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::{Duration, Instant};

    /// The longest request line, in bytes (414 past it).
    pub const LINE_MAX: usize = 8192;
    /// The longest head: request line and fields with their line ends (431).
    pub const HEAD_MAX: usize = 65536;
    /// The most field lines in a head (431).
    pub const FIELDS_MAX: usize = 100;
    /// The longest body (413): N9 SIZE_LIMIT, so `body: List[i64]` can always be built.
    pub const BODY_MAX: usize = crate::interp::SIZE_LIMIT;
    /// The most connections open at once; the acceptor waits past it.
    pub const CONNS_MAX: usize = 256;
    /// From the first byte of a request to the end of its head (408).
    pub const HEAD_MS: u64 = 10_000;
    /// From the end of the head to the end of the body (408); 1 MB at 34 KB/s.
    pub const BODY_MS: u64 = 30_000;
    /// Between requests on one connection: then it closes, and nothing is logged.
    pub const IDLE_MS: u64 = 5_000;
    /// For the program's response to an admitted request: then the connection closes.
    pub const RESPOND_MS: u64 = 60_000;
    /// For writing one response.
    pub const WRITE_MS: u64 = 30_000;

    /// The limits and deadlines as `host.HttpListen` records them (keys sorted).
    pub fn limits_json() -> String {
        format!(
            "\"body_max\":{BODY_MAX},\"body_ms\":{BODY_MS},\"conns_max\":{CONNS_MAX},\"fields_max\":{FIELDS_MAX},\"head_max\":{HEAD_MAX},\"head_ms\":{HEAD_MS},\"idle_ms\":{IDLE_MS},\"line_max\":{LINE_MAX},\"respond_ms\":{RESPOND_MS},\"write_ms\":{WRITE_MS}"
        )
    }

    /// One request as the host admits it.
    #[derive(Clone, Debug, PartialEq, Default)]
    pub struct Parsed {
        pub method: String,
        pub target: String,
        pub major: u8,
        pub minor: u8,
        /// field names as written and values without surrounding whitespace, in order
        pub names: Vec<String>,
        pub values: Vec<String>,
        pub body: Vec<u8>,
        /// trailer fields of a chunked body (parsed, not delivered: HC-24)
        pub trailer_names: Vec<String>,
        pub trailer_values: Vec<String>,
        /// the connection closes after the response
        pub close: bool,
        /// a HEAD request: the response carries no body
        pub head: bool,
        /// `Expect: 100-continue` with a body to come
        pub expect_continue: bool,
    }

    impl Parsed {
        /// The version as sent (`HTTP/1.1`): the parser admits only `HTTP/`,
        /// a digit, a dot and a digit, so this is the text of the request line.
        pub fn version(&self) -> String {
            format!("HTTP/{}.{}", self.major, self.minor)
        }
    }

    /// Whether the connection closes after the response to a request of
    /// this version with these fields (RFC 9112 9.3): from HTTP/1.1 on when
    /// Connection lists `close`, and for HTTP/1.0 unless it lists
    /// `keep-alive`. The version and the fields are in the host's value of
    /// the request, which D holds (HC-24), so this is a function of D
    /// (HC-26, `host::http_inputs`).
    pub fn closes(version: &str, names: &[String], values: &[String]) -> bool {
        let conn: Vec<String> = header(names, values, "connection").map(|c| list_items(&c.to_ascii_lowercase())).unwrap_or_default();
        let minor = version.strip_prefix("HTTP/1.").and_then(|m| m.parse::<u8>().ok()).unwrap_or(0);
        if minor >= 1 { conn.iter().any(|c| c == "close") } else { !conn.iter().any(|c| c == "keep-alive") }
    }

    /// RFC 9110 5.6.2 `tchar` characters, as `net.http`'s `tchar`.
    pub fn tchar(c: u8) -> bool {
        c.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&c)
    }

    fn trim_ows(t: &str) -> String {
        t.trim_matches([' ', '\t']).to_string()
    }

    fn ascii_text(b: &[u8]) -> Result<String, &'static str> {
        if b.iter().all(|&x| x == 9 || (32..=126).contains(&x)) { Ok(b.iter().map(|&x| x as char).collect()) } else { Err("not ASCII text") }
    }

    /// The values of a field (case-insensitive name) joined with ", ", or none.
    pub fn header(names: &[String], values: &[String], name: &str) -> Option<String> {
        let vs: Vec<&str> = names.iter().zip(values).filter(|(n, _)| n.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str()).collect();
        (!vs.is_empty()).then(|| vs.join(", "))
    }

    fn list_items(v: &str) -> Vec<String> {
        v.split(',').map(trim_ows).collect()
    }

    fn dec(t: &str) -> i64 {
        if t.is_empty() || t.len() > 15 || !t.bytes().all(|c| c.is_ascii_digit()) { -1 } else { t.parse().unwrap_or(-1) }
    }

    fn hexnum(t: &str) -> i64 {
        if t.is_empty() || t.len() > 15 || !t.bytes().all(|c| c.is_ascii_hexdigit()) { -1 } else { i64::from_str_radix(t, 16).unwrap_or(-1) }
    }

    fn field(t: &str) -> Result<(String, String), &'static str> {
        let b = t.as_bytes();
        if !b.is_empty() && (b[0] == b' ' || b[0] == b'\t') {
            return Err("obs-fold");
        }
        let mut i = 0;
        while i < b.len() && b[i] != b':' {
            if !tchar(b[i]) {
                return Err("field name");
            }
            i += 1;
        }
        if i >= b.len() || i == 0 {
            return Err("field line");
        }
        Ok((t[..i].to_string(), trim_ows(&t[i + 1..])))
    }

    fn start_line(t: &str) -> Result<(String, String, u8, u8), &'static str> {
        let ps: Vec<&str> = t.split(' ').collect();
        if ps.len() != 3 || ps[0].is_empty() || ps[1].is_empty() {
            return Err("request line");
        }
        if !ps[0].bytes().all(tchar) {
            return Err("method");
        }
        if !ps[1].bytes().all(|c| (33..=126).contains(&c)) {
            return Err("target");
        }
        let v = ps[2].as_bytes();
        if v.len() != 8 || !ps[2].starts_with("HTTP/") || v[6] != b'.' || !v[5].is_ascii_digit() || !v[7].is_ascii_digit() {
            return Err("version");
        }
        Ok((ps[0].to_string(), ps[1].to_string(), v[5] - b'0', v[7] - b'0'))
    }

    #[derive(Clone, Copy, Debug, PartialEq, Default)]
    enum Stage {
        #[default]
        Start,
        Fields,
        Body(usize),
        ChunkSize,
        ChunkData(usize),
        Trailers,
    }

    /// What one call of [`Parser::feed`] found.
    #[derive(Debug, PartialEq)]
    pub enum Progress {
        /// more bytes are needed
        Incomplete,
        /// a complete request and the number of bytes it used
        Complete(Parsed, usize),
        /// the status and the why of a request delivered as `HttpMalformed`
        Malformed(u16, String),
    }

    /// A request parser that resumes where it stopped: `feed` takes the
    /// whole buffer of the request so far, and every byte is examined a
    /// bounded number of times, so a slow sender costs no more than a fast one.
    #[derive(Default)]
    pub struct Parser {
        stage: Stage,
        /// bytes of complete units (lines, chunks) consumed
        pos: usize,
        /// where the search for the current line's end resumes
        scan: usize,
        skipped: usize,
        head_end: Option<usize>,
        p: Parsed,
    }

    fn bad(why: &str) -> Progress {
        Progress::Malformed(400, why.to_string())
    }

    impl Parser {
        /// Whether the whole head has arrived and was accepted.
        pub fn head_done(&self) -> bool {
            self.head_end.is_some()
        }
        pub fn expect_continue(&self) -> bool {
            self.p.expect_continue
        }

        /// The CR ending the line that starts at pos, `Ok(None)` when the
        /// buffer ends first; a bare CR or LF makes the request malformed
        /// (net.http `line_end`).
        fn line_end(&mut self, b: &[u8]) -> Result<Option<usize>, &'static str> {
            let mut k = self.scan.max(self.pos);
            while k < b.len() {
                match b[k] {
                    13 => {
                        if k + 1 >= b.len() {
                            self.scan = k;
                            return Ok(None);
                        }
                        if b[k + 1] != 10 {
                            self.scan = k;
                            return Err("bare CR");
                        }
                        return Ok(Some(k));
                    }
                    10 => {
                        self.scan = k;
                        return Err("bare LF");
                    }
                    _ => k += 1,
                }
            }
            self.scan = k;
            Ok(None)
        }

        fn next_line(&mut self, e: usize) {
            self.pos = e + 2;
            self.scan = self.pos;
        }

        pub fn feed(&mut self, b: &[u8]) -> Progress {
            loop {
                match self.stage {
                    Stage::Start => {
                        // leading empty lines before a request are skipped (RFC 9112 2.2), at most 16
                        while self.skipped < 16 && self.pos + 1 < b.len() && b[self.pos] == 13 && b[self.pos + 1] == 10 {
                            self.pos += 2;
                            self.scan = self.pos;
                            self.skipped += 1;
                        }
                        let e = match self.line_end(b) {
                            Err(w) => return if self.scan - self.pos > LINE_MAX { too_long_line() } else { bad(w) },
                            Ok(None) => return if self.scan - self.pos > LINE_MAX { too_long_line() } else { Progress::Incomplete },
                            Ok(Some(e)) => e,
                        };
                        if e - self.pos > LINE_MAX {
                            return too_long_line();
                        }
                        let sl = match ascii_text(&b[self.pos..e]) {
                            Ok(t) => t,
                            Err(w) => return bad(w),
                        };
                        match start_line(&sl) {
                            Ok((m, t, a, c)) => {
                                self.p.method = m;
                                self.p.target = t;
                                self.p.major = a;
                                self.p.minor = c;
                            }
                            Err(w) => return bad(w),
                        }
                        self.next_line(e);
                        self.stage = Stage::Fields;
                    }
                    Stage::Fields => {
                        let e = match self.line_end(b) {
                            Err(w) => return if self.scan > HEAD_MAX { too_long_head() } else { bad(w) },
                            Ok(None) => return if b.len() > HEAD_MAX { too_long_head() } else { Progress::Incomplete },
                            Ok(Some(e)) => e,
                        };
                        if e + 2 > HEAD_MAX {
                            return too_long_head();
                        }
                        if e == self.pos {
                            self.next_line(e);
                            if let Some(r) = self.end_of_head() {
                                return r;
                            }
                            continue;
                        }
                        if self.p.names.len() >= FIELDS_MAX {
                            return Progress::Malformed(431, format!("over {FIELDS_MAX} fields"));
                        }
                        let t = match ascii_text(&b[self.pos..e]) {
                            Ok(t) => t,
                            Err(w) => return bad(w),
                        };
                        match field(&t) {
                            Ok((n, v)) => {
                                self.p.names.push(n);
                                self.p.values.push(v);
                            }
                            Err(w) => return bad(w),
                        }
                        self.next_line(e);
                    }
                    Stage::Body(n) => {
                        let h = self.pos;
                        if b.len() < h + n {
                            return Progress::Incomplete;
                        }
                        self.p.body = b[h..h + n].to_vec();
                        return self.complete(h + n);
                    }
                    Stage::ChunkSize => {
                        let e = match self.line_end(b) {
                            Err(w) => return bad(w),
                            Ok(None) => return self.chunk_overflow(b),
                            Ok(Some(e)) => e,
                        };
                        let line = match ascii_text(&b[self.pos..e]) {
                            Ok(t) => t,
                            Err(w) => return bad(w),
                        };
                        let size: String = line.chars().take_while(|c| *c != ';' && *c != ' ' && *c != '\t').collect();
                        let n = hexnum(&size);
                        if n < 0 {
                            return bad("chunk size");
                        }
                        self.next_line(e);
                        if n == 0 {
                            self.stage = Stage::Trailers;
                        } else {
                            if self.p.body.len() + n as usize > BODY_MAX {
                                return too_big();
                            }
                            self.stage = Stage::ChunkData(n as usize);
                        }
                    }
                    Stage::ChunkData(n) => {
                        let i = self.pos;
                        if i + n + 2 > b.len() {
                            return self.chunk_overflow(b);
                        }
                        self.p.body.extend_from_slice(&b[i..i + n]);
                        if b[i + n] != 13 || b[i + n + 1] != 10 {
                            return bad("chunk end");
                        }
                        self.pos = i + n + 2;
                        self.scan = self.pos;
                        self.stage = Stage::ChunkSize;
                    }
                    Stage::Trailers => {
                        let e = match self.line_end(b) {
                            Err(w) => return bad(w),
                            Ok(None) => return self.chunk_overflow(b),
                            Ok(Some(e)) => e,
                        };
                        if e == self.pos {
                            return self.complete(e + 2);
                        }
                        let t = match ascii_text(&b[self.pos..e]) {
                            Ok(t) => t,
                            Err(w) => return bad(w),
                        };
                        match field(&t) {
                            Ok((n, v)) => {
                                self.p.trailer_names.push(n);
                                self.p.trailer_values.push(v);
                            }
                            Err(w) => return bad(w),
                        }
                        self.next_line(e);
                    }
                }
            }
        }

        /// The encoded body (sizes, data, trailers) may use at most
        /// BODY_MAX + HEAD_MAX bytes, so a chunked body is bounded too.
        fn chunk_overflow(&self, b: &[u8]) -> Progress {
            let h = self.head_end.unwrap_or(0);
            if b.len() - h > BODY_MAX + HEAD_MAX { too_big() } else { Progress::Incomplete }
        }

        /// The rules applied once the head is complete, in the order of
        /// `net.http.parse_message` (the version and Host first).
        fn end_of_head(&mut self) -> Option<Progress> {
            self.head_end = Some(self.pos);
            let p = &mut self.p;
            if p.major != 1 {
                return Some(Progress::Malformed(505, format!("HTTP/{}.{} is not supported", p.major, p.minor)));
            }
            // RFC 9112 3.2: at most one Host field, and exactly one from HTTP/1.1 on
            let hosts = p.names.iter().filter(|n| n.eq_ignore_ascii_case("host")).count();
            if hosts > 1 || (hosts == 0 && p.minor >= 1) {
                return Some(bad("Host"));
            }
            p.close = closes(&p.version(), &p.names, &p.values);
            p.head = p.method == "HEAD";
            let expect = header(&p.names, &p.values, "expect").is_some_and(|e| e.eq_ignore_ascii_case("100-continue")) && p.minor >= 1;
            let te = header(&p.names, &p.values, "transfer-encoding");
            let cl = header(&p.names, &p.values, "content-length");
            if let Some(te) = te {
                if cl.is_some() {
                    return Some(bad("both Transfer-Encoding and Content-Length"));
                }
                let items = list_items(&te);
                if !items[items.len() - 1].eq_ignore_ascii_case("chunked") {
                    return Some(bad("transfer coding"));
                }
                if items[..items.len() - 1].iter().any(|x| x.eq_ignore_ascii_case("chunked")) {
                    return Some(bad("chunked twice"));
                }
                if items.len() > 1 {
                    return Some(Progress::Malformed(501, "unsupported transfer coding".into()));
                }
                p.expect_continue = expect;
                self.stage = Stage::ChunkSize;
                return None;
            }
            if let Some(cl) = cl {
                let items = list_items(&cl);
                let n = dec(&items[0]);
                if items.iter().any(|x| dec(x) != n) || n < 0 {
                    return Some(bad("Content-Length"));
                }
                if n as usize > BODY_MAX {
                    return Some(too_big());
                }
                p.expect_continue = expect && n > 0;
                self.stage = Stage::Body(n as usize);
                return None;
            }
            let used = self.pos;
            Some(self.complete(used))
        }

        fn complete(&mut self, used: usize) -> Progress {
            Progress::Complete(std::mem::take(&mut self.p), used)
        }
    }

    fn too_long_line() -> Progress {
        Progress::Malformed(414, format!("request line over {LINE_MAX} bytes"))
    }
    fn too_long_head() -> Progress {
        Progress::Malformed(431, format!("field section over {HEAD_MAX} bytes"))
    }
    fn too_big() -> Progress {
        Progress::Malformed(413, format!("body over {BODY_MAX} bytes"))
    }

    /// Parse one request from the front of `b`: `Ok(Some((request, used)))`,
    /// `Ok(None)` when more bytes are needed, or `Err((status, why))`.
    pub fn parse_request(b: &[u8]) -> Result<Option<(Parsed, usize)>, (u16, String)> {
        match Parser::default().feed(b) {
            Progress::Incomplete => Ok(None),
            Progress::Complete(p, n) => Ok(Some((p, n))),
            Progress::Malformed(s, w) => Err((s, w)),
        }
    }

    /// The reason phrases the host writes (RFC 9110 15); others are empty.
    pub fn reason(status: u64) -> &'static str {
        match status {
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            203 => "Non-Authoritative Information",
            204 => "No Content",
            205 => "Reset Content",
            206 => "Partial Content",
            300 => "Multiple Choices",
            301 => "Moved Permanently",
            302 => "Found",
            303 => "See Other",
            304 => "Not Modified",
            307 => "Temporary Redirect",
            308 => "Permanent Redirect",
            400 => "Bad Request",
            401 => "Unauthorized",
            402 => "Payment Required",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            406 => "Not Acceptable",
            407 => "Proxy Authentication Required",
            408 => "Request Timeout",
            409 => "Conflict",
            410 => "Gone",
            411 => "Length Required",
            412 => "Precondition Failed",
            413 => "Content Too Large",
            414 => "URI Too Long",
            415 => "Unsupported Media Type",
            416 => "Range Not Satisfiable",
            417 => "Expectation Failed",
            421 => "Misdirected Request",
            422 => "Unprocessable Content",
            426 => "Upgrade Required",
            428 => "Precondition Required",
            429 => "Too Many Requests",
            431 => "Request Header Fields Too Large",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            505 => "HTTP Version Not Supported",
            507 => "Insufficient Storage",
            _ => "",
        }
    }

    /// The bytes of the response to `r` (an `HttpRespond` request as Form
    /// JSON), or why the effect fails (HC-26), and whether the connection
    /// closes after it. The host adds only the status line,
    /// `Content-Length` and `Connection: close`; with `head` and `close`
    /// from `host::http_inputs`, every byte of the response is a function
    /// of D.
    pub fn response_bytes(r: &Value, head: bool, close: bool) -> Result<(Vec<u8>, bool), String> {
        // HD-2: the request is read by field name; a missing field is empty
        let Some(status) = r["status"].as_u64() else { return Err("status is not an integer from 200 to 599".into()) };
        if !(200..=599).contains(&status) {
            return Err(format!("status {status} is outside 200 to 599"));
        }
        let strs = |k: &str| -> Vec<String> { r[k].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(String::from)).collect() };
        let (names, values) = (strs("names"), strs("values"));
        if names.len() != values.len() {
            return Err("names and values differ in length".into());
        }
        let mut program_close = false;
        for (n, v) in names.iter().zip(&values) {
            if n.is_empty() || !n.bytes().all(tchar) {
                return Err(format!("field name `{n}` has a character that is not a tchar"));
            }
            if !v.bytes().all(|c| c == 9 || (32..=126).contains(&c)) {
                return Err(format!("the value of `{n}` is not visible ASCII"));
            }
            let l = n.to_ascii_lowercase();
            if l == "content-length" || l == "transfer-encoding" {
                return Err(format!("`{n}` is written by the host"));
            }
            if l == "connection" {
                if !v.eq_ignore_ascii_case("close") {
                    return Err("Connection may only be `close`".into());
                }
                program_close = true;
            }
        }
        let body = super::request_bytes(&r["body"])?;
        let bodyless = status == 204 || status == 304;
        if bodyless && !body.is_empty() {
            return Err(format!("a {status} response has no body"));
        }
        let close = close || program_close;
        let mut out = format!("HTTP/1.1 {status} {}\r\n", reason(status));
        for (n, v) in names.iter().zip(&values) {
            out.push_str(&format!("{n}: {v}\r\n"));
        }
        if !bodyless {
            out.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }
        if close && !program_close {
            out.push_str("Connection: close\r\n");
        }
        out.push_str("\r\n");
        let mut bytes = out.into_bytes();
        if !head {
            bytes.extend_from_slice(&body);
        }
        Ok((bytes, close))
    }

    /// The bytes the host writes for one response.
    pub struct Reply {
        pub bytes: Vec<u8>,
        pub close: bool,
    }

    /// What the HTTP adapter hands the admitter.
    pub enum HttpIn {
        Request { req: Parsed, reply: mpsc::Sender<Reply> },
        Malformed { status: u16, why: String, reply: mpsc::Sender<Reply> },
        /// a bind that had failed succeeded on a retry
        Listening,
    }

    /// How the adapter hands an input to the admitter; false when the admitter is gone.
    pub type Deliver = Arc<dyn Fn(HttpIn) -> bool + Send + Sync>;

    /// A request admitted and not yet answered.
    pub struct Pending {
        pub reply: mpsc::Sender<Reply>,
        pub close: bool,
        pub head: bool,
    }

    /// The open requests, by id (the seq of their entry).
    pub type Open = std::rc::Rc<std::cell::RefCell<HashMap<u64, Pending>>>;

    /// The adapter of the `HttpRespond` effect (HC-26). Its result is
    /// `delivered` with the length of the response handed to the request's
    /// connection thread, which writes it later; a write that fails then
    /// is not reported to the program.
    pub struct RespondAdapter {
        open: Open,
    }

    impl Adapter for RespondAdapter {
        fn kind(&self) -> &'static str {
            "HttpRespond"
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            let id = r["req"].as_u64().unwrap_or(0);
            let mut open = self.open.borrow_mut();
            let Some(p) = open.get(&id) else { return EffectResult::failed(format!("no request {id}")) };
            // a response that fails the checks leaves the request open
            let (bytes, close) = match response_bytes(r, p.head, p.close) {
                Ok(x) => x,
                Err(e) => return EffectResult::failed(e),
            };
            let p = open.remove(&id).unwrap();
            let n = bytes.len();
            match p.reply.send(Reply { bytes, close }) {
                Ok(()) => EffectResult::ok(json!(n)),
                Err(_) => EffectResult::failed(format!("no request {id}")),
            }
        }
    }

    struct Ctx {
        stopping: AtomicBool,
        active: AtomicUsize,
    }

    /// The HTTP listener of one host process. It binds before the sandbox
    /// is entered ([`Server::bind`]) and starts its threads after
    /// ([`Server::start`]), so they run under the sandbox.
    pub struct Server {
        pub addr: String,
        l: Option<TcpListener>,
        error: Option<String>,
        ctx: Arc<Ctx>,
        open: Open,
    }

    impl Server {
        /// Bind the granted address; a failure is recorded and retried by `start`.
        pub fn bind(addr: &str) -> Server {
            let (l, error) = match TcpListener::bind(addr) {
                Ok(l) => (Some(l), None),
                Err(e) => (None, Some(e.to_string())),
            };
            Server { addr: addr.to_string(), l, error, ctx: Arc::new(Ctx { stopping: AtomicBool::new(false), active: AtomicUsize::new(0) }), open: Default::default() }
        }
        /// Whether the address is bound now.
        pub fn bound(&self) -> bool {
            self.error.is_none()
        }
        /// The local address actually bound (for a grant of port 0).
        pub fn local_addr(&self) -> Option<std::net::SocketAddr> {
            self.l.as_ref().and_then(|l| l.local_addr().ok())
        }
        /// The payload of `host.HttpListen`: the address and the limits, or
        /// the address and the bind error.
        pub fn listen_payload(&self) -> String {
            let a = serde_json::to_string(&self.addr).unwrap();
            match &self.error {
                None => format!("{{\"addr\":{a},{}}}", limits_json()),
                Some(e) => format!("{{\"addr\":{a},\"error\":{}}}", serde_json::to_string(e).unwrap()),
            }
        }
        /// After a retry bound the address.
        pub fn listening(&mut self) {
            self.error = None;
        }
        pub fn responder(&self) -> RespondAdapter {
            RespondAdapter { open: self.open.clone() }
        }
        pub fn open(&self) -> Open {
            self.open.clone()
        }
        /// Connections open now.
        pub fn active(&self) -> usize {
            self.ctx.active.load(Ordering::SeqCst)
        }
        /// Stop accepting: new connections are closed at once, idle ones
        /// close, and a request already being read is finished.
        pub fn stop(&self) {
            self.ctx.stopping.store(true, Ordering::SeqCst);
        }
        /// Start the acceptor (or the retry of a failed bind, every second).
        pub fn start(&mut self, deliver: Deliver) {
            let ctx = self.ctx.clone();
            match self.l.take() {
                Some(l) => {
                    std::thread::spawn(move || accept_loop(l, ctx, deliver));
                }
                None => {
                    let addr = self.addr.clone();
                    std::thread::spawn(move || {
                        loop {
                            std::thread::sleep(Duration::from_secs(1));
                            if let Ok(l) = TcpListener::bind(&addr) {
                                if deliver(HttpIn::Listening) {
                                    accept_loop(l, ctx, deliver);
                                }
                                return;
                            }
                        }
                    });
                }
            }
        }
    }

    struct Active(Arc<Ctx>);
    impl Drop for Active {
        fn drop(&mut self) {
            self.0.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    fn accept_loop(l: TcpListener, ctx: Arc<Ctx>, deliver: Deliver) {
        loop {
            while ctx.active.load(Ordering::SeqCst) >= CONNS_MAX {
                std::thread::sleep(Duration::from_millis(5));
            }
            match l.accept() {
                Ok((s, _)) => {
                    if ctx.stopping.load(Ordering::SeqCst) {
                        continue;
                    }
                    ctx.active.fetch_add(1, Ordering::SeqCst);
                    let (c, d) = (ctx.clone(), deliver.clone());
                    std::thread::spawn(move || {
                        let _a = Active(c.clone());
                        connection(s, &c, &d);
                    });
                }
                Err(_) => std::thread::sleep(Duration::from_millis(10)),
            }
        }
    }

    /// Close after a response: stop writing, read what the client still
    /// sends for a moment so the response is not lost to a reset, then close.
    fn finish(mut s: TcpStream) {
        let _ = s.shutdown(Shutdown::Write);
        let _ = s.set_read_timeout(Some(Duration::from_millis(100)));
        let start = Instant::now();
        let mut b = [0u8; 65536];
        let mut left: usize = 4 << 20;
        while start.elapsed() < Duration::from_secs(1) && left > 0 {
            match s.read(&mut b) {
                Ok(0) => break,
                Ok(n) => left = left.saturating_sub(n),
                Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut || e.kind() == ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
    }

    /// One connection: requests are read under the deadlines, parsed, and
    /// admitted one at a time, the next only after the previous response,
    /// so responses stay in order (RFC 9112 9.3.2).
    fn connection(mut s: TcpStream, ctx: &Ctx, deliver: &Deliver) {
        s.set_nodelay(true).ok();
        let mut buf: Vec<u8> = vec![];
        let mut tmp = vec![0u8; 65536];
        loop {
            let mut p = Parser::default();
            let idle_since = Instant::now();
            let mut first: Option<Instant> = (!buf.is_empty()).then(Instant::now);
            let mut head_at: Option<Instant> = None;
            let mut continued = false;
            let outcome: Result<(Parsed, usize), (u16, String)> = loop {
                if !buf.is_empty() {
                    match p.feed(&buf) {
                        Progress::Complete(r, n) => break Ok((r, n)),
                        Progress::Malformed(st, w) => break Err((st, w)),
                        Progress::Incomplete => {}
                    }
                }
                if p.head_done() && head_at.is_none() {
                    head_at = Some(Instant::now());
                    if p.expect_continue() && !continued {
                        continued = true;
                        let _ = s.set_write_timeout(Some(Duration::from_millis(WRITE_MS)));
                        if s.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").is_err() {
                            return;
                        }
                    }
                }
                let (deadline, expired) = match (first, head_at) {
                    (None, _) => (idle_since + Duration::from_millis(IDLE_MS), None),
                    (Some(f), None) => (f + Duration::from_millis(HEAD_MS), Some("head timed out")),
                    (Some(_), Some(h)) => (h + Duration::from_millis(BODY_MS), Some("body timed out")),
                };
                if first.is_none() && ctx.stopping.load(Ordering::SeqCst) {
                    return;
                }
                let now = Instant::now();
                if now >= deadline {
                    match expired {
                        None => return,
                        Some(w) => break Err((408, w.to_string())),
                    }
                }
                let _ = s.set_read_timeout(Some((deadline - now).clamp(Duration::from_millis(1), Duration::from_millis(200))));
                match s.read(&mut tmp) {
                    // closed between requests or inside one: nothing is logged
                    Ok(0) => return,
                    Ok(n) => {
                        buf.extend_from_slice(&tmp[..n]);
                        first.get_or_insert_with(Instant::now);
                    }
                    Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut || e.kind() == ErrorKind::Interrupted => {}
                    Err(_) => return,
                }
            };
            let (tx, rx) = mpsc::channel();
            let close_after = match outcome {
                Ok((r, used)) => {
                    buf.drain(..used);
                    let c = r.close;
                    if !deliver(HttpIn::Request { req: r, reply: tx }) {
                        return;
                    }
                    c
                }
                Err((status, why)) => {
                    if !deliver(HttpIn::Malformed { status, why, reply: tx }) {
                        return;
                    }
                    true
                }
            };
            let reply = match rx.recv_timeout(Duration::from_millis(RESPOND_MS)) {
                Ok(r) => r,
                // no response in time, or the admitter is gone: close; a
                // later HttpRespond for this request fails `no request N`
                Err(_) => return,
            };
            let _ = s.set_write_timeout(Some(Duration::from_millis(WRITE_MS)));
            if s.write_all(&reply.bytes).and_then(|_| s.flush()).is_err() {
                return;
            }
            if reply.close || close_after {
                finish(s);
                return;
            }
        }
    }
}
