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
/// `FileList(req, path)` -> List[text] in canonical (sorted) order.
pub const FILE_KINDS: [&str; 4] = ["FileRead", "FileWrite", "FileAppend", "FileList"];

pub struct FileAdapter {
    kind: &'static str,
    root: PathBuf,
}

/// One adapter per file request kind, all under `root`.
pub fn file_adapters(root: &Path) -> Vec<FileAdapter> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    FILE_KINDS.iter().map(|k| FileAdapter { kind: k, root: root.clone() }).collect()
}

impl FileAdapter {
    /// The path under the root, or why it is refused (least privilege).
    fn resolve(&self, p: &str) -> Result<PathBuf, String> {
        if p.is_empty() && self.kind == "FileList" {
            return Ok(self.root.clone());
        }
        let rel = Path::new(p);
        if p.is_empty() || rel.is_absolute() || rel.components().any(|c| !matches!(c, Component::Normal(_) | Component::CurDir)) {
            return Err(format!("path `{p}` is outside the granted root"));
        }
        let full = self.root.join(rel);
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
}

fn barrier_dir(p: &Path) -> std::io::Result<()> {
    let d = p.parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    File::open(d)?.sync_all()
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
            let mut v: Vec<String> = fs::read_dir(full).map_err(s)?.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
            v.sort();
            Ok(json!(v))
        }
        k => Err(format!("unknown file request `{k}`")),
    }
}

impl Adapter for FileAdapter {
    fn kind(&self) -> &'static str {
        self.kind
    }
    fn perform(&mut self, r: &Value) -> EffectResult {
        let res = self.resolve(r["path"].as_str().unwrap_or("")).and_then(|full| perform_file(self.kind, &full, r));
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

/// W1i network adapter (TCP), the transport under P4 remote channels.
/// Authority (HC-5): only the `host:port` addresses granted with `--net`.
/// `NetConnect(req, addr)` -> u64 id; `NetListen(req, addr)` -> u64 id
/// (the address must be granted); `NetAccept(req, id)` -> u64 id of a new
/// connection, or failed `none` when nothing is waiting; `NetSend(req, id,
/// data)` -> u64 bytes; `NetRecv(req, id, max)` -> text received now (may be
/// empty), failed `eof` when closed; `NetClose(req, id)` -> u64 0. Every
/// byte received is an input logged in D (HC-10).
pub mod net {
    use crate::host::{Adapter, EffectResult};
    use serde_json::{Value, json};
    use std::collections::BTreeMap;
    use std::io::{ErrorKind, Read, Write};
    use std::net::{TcpListener, TcpStream, ToSocketAddrs};
    use std::time::Duration;

    pub const NET_KINDS: [&str; 6] = ["NetConnect", "NetListen", "NetAccept", "NetSend", "NetRecv", "NetClose"];

    enum Sock {
        L(TcpListener),
        S(TcpStream),
    }

    #[derive(Default)]
    pub struct Nets {
        next: u64,
        socks: BTreeMap<u64, Sock>,
        grants: Vec<String>,
    }

    pub struct NetAdapter {
        kind: &'static str,
        nets: std::rc::Rc<std::cell::RefCell<Nets>>,
    }

    pub fn net_adapters(grants: &[String]) -> Vec<NetAdapter> {
        let nets = std::rc::Rc::new(std::cell::RefCell::new(Nets { grants: grants.to_vec(), ..Default::default() }));
        NET_KINDS.iter().map(|k| NetAdapter { kind: k, nets: nets.clone() }).collect()
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

    impl Adapter for NetAdapter {
        fn kind(&self) -> &'static str {
            self.kind
        }
        fn perform(&mut self, r: &Value) -> EffectResult {
            let mut n = self.nets.borrow_mut();
            let id = r["id"].as_u64().unwrap_or(0);
            let addr = r["addr"].as_str().unwrap_or("").to_string();
            let res: Result<Value, String> = match self.kind {
                "NetConnect" => n.granted(&addr).and_then(|_| {
                    let sa = addr.to_socket_addrs().map_err(|e| e.to_string())?.next().ok_or("no address")?;
                    let s = TcpStream::connect_timeout(&sa, Duration::from_secs(5)).map_err(|e| e.to_string())?;
                    s.set_nodelay(true).ok();
                    s.set_nonblocking(true).map_err(|e| e.to_string())?;
                    Ok(json!(n.add(Sock::S(s))))
                }),
                "NetListen" => n.granted(&addr).and_then(|_| {
                    let l = TcpListener::bind(&addr).map_err(|e| e.to_string())?;
                    l.set_nonblocking(true).map_err(|e| e.to_string())?;
                    Ok(json!(n.add(Sock::L(l))))
                }),
                "NetAccept" => match n.socks.get(&id) {
                    Some(Sock::L(l)) => match l.accept() {
                        Ok((s, _)) => {
                            s.set_nodelay(true).ok();
                            s.set_nonblocking(true).map_err(|e| e.to_string()).map(|_| s).map(|s| json!(n.add(Sock::S(s))))
                        }
                        Err(e) if e.kind() == ErrorKind::WouldBlock => Err("none".into()),
                        Err(e) => Err(e.to_string()),
                    },
                    _ => Err(format!("no listener {id}")),
                },
                "NetSend" => match n.socks.get_mut(&id) {
                    Some(Sock::S(s)) => {
                        let d = r["data"].as_str().unwrap_or("").as_bytes();
                        s.set_nonblocking(false).ok();
                        let w = s.write_all(d).map(|_| json!(d.len())).map_err(|e| e.to_string());
                        s.set_nonblocking(true).ok();
                        w
                    }
                    _ => Err(format!("no connection {id}")),
                },
                "NetRecv" => match n.socks.get_mut(&id) {
                    Some(Sock::S(s)) => {
                        let max = r["max"].as_u64().unwrap_or(65536).min(1 << 20) as usize;
                        let mut out = vec![];
                        let mut buf = [0u8; 4096];
                        let mut eof = false;
                        while out.len() < max {
                            match s.read(&mut buf[..(max - out.len()).min(4096)]) {
                                Ok(0) => {
                                    eof = true;
                                    break;
                                }
                                Ok(k) => out.extend_from_slice(&buf[..k]),
                                Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                                Err(e) => return EffectResult::failed(e.to_string()),
                            }
                        }
                        if eof && out.is_empty() { Err("eof".into()) } else { Ok(json!(String::from_utf8_lossy(&out))) }
                    }
                    _ => Err(format!("no connection {id}")),
                },
                "NetClose" => n.socks.remove(&id).map(|_| json!(0)).ok_or(format!("no socket {id}")),
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
/// the module bytes, stored under the granted `--wasm DIR` (the BlobPut
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
        let shared: Arc<(Mutex<Shared>, Condvar)> = Default::default();
        let pins: Arc<HashMap<String, [u8; 32]>> = Arc::new(peers.iter().cloned().collect());
        let scfg = crate::tls::server_config(id, peers.iter().map(|p| p.1).collect())?;
        let (sh, pn) = (shared.clone(), pins.clone());
        std::thread::spawn(move || {
            for s in l.incoming().flatten() {
                s.set_nodelay(true).ok();
                s.set_read_timeout(Some(TLS_POLL)).ok();
                let Ok(conn) = rustls::ServerConnection::new(scfg.clone()) else { continue };
                let (sh, pn) = (sh.clone(), pn.clone());
                std::thread::spawn(move || tls_loop(Box::new(rustls::StreamOwned::new(conn, s)), None, sh, pn));
            }
        });
        let c = std::rc::Rc::new(std::cell::RefCell::new(Chans {
            me: me.into(),
            grants: peers.iter().map(|p| p.0.clone()).collect(),
            shared,
            out: HashMap::new(),
            tls: Some((id.clone(), pins)),
            tls_out: HashMap::new(),
        }));
        Ok(CHAN_KINDS.iter().map(|k| ChanAdapter { kind: k, c: c.clone() }).collect())
    }

    /// Listen on `me` (a granted local address) and reach only `peers`.
    pub fn chan_adapters(me: &str, peers: &[String]) -> std::io::Result<Vec<ChanAdapter>> {
        let l = TcpListener::bind(me)?;
        let shared: Arc<(Mutex<Shared>, Condvar)> = Default::default();
        let sh = shared.clone();
        std::thread::spawn(move || {
            for s in l.incoming().flatten() {
                let sh = sh.clone();
                std::thread::spawn(move || serve_conn(s, sh));
            }
        });
        let c = std::rc::Rc::new(std::cell::RefCell::new(Chans { me: me.into(), grants: peers.to_vec(), shared, out: HashMap::new(), tls: None, tls_out: HashMap::new() }));
        Ok(CHAN_KINDS.iter().map(|k| ChanAdapter { kind: k, c: c.clone() }).collect())
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
/// `<root>/play/<track>`; a frame whose bytes do not match is refused.
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
