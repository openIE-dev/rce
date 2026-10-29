//! HC-5 (W1h): after the log and the granted adapters are open, the host
//! process also enters the sandbox's OS enforcement target. There is one RCE
//! sandbox with one contract (HC-5 capability checking), independent of the
//! OS: the checker and the deterministic host enforce it before any OS layer.
//! Underneath, an enforcement target makes the host process itself hold no
//! more than the grants: network only as the target can name the grants,
//! writes only under the log directory and the granted roots, no exec unless
//! a PTY is granted. Entering it changes no meaning: the program has no
//! ambient authority anyway (5.15); the sandbox removes the host's own.
//!
//! HC-5a: the line the host prints when it enters a target is generated
//! from the rules it installed ([`Rule`], [`seatbelt_rules`],
//! `landlock::Installed`), so it names each restriction with its
//! granularity and never one the target did not install.
//!
//! Only the `rce` binary enables this ([`enable`]); library callers and
//! in-process tests are never sandboxed, since a sandbox is per process and
//! cannot be left.
//!
//! OS enforcement targets in this file:
//! - macOS target (Seatbelt): `sandbox_init` with a profile that denies all
//!   network use but the granted ports (Seatbelt names a port at
//!   `localhost` or at any address, nothing finer; the net adapter checks
//!   the exact address), denies writes outside the log's directory, the
//!   granted roots and `/dev`, and denies process exec unless `--pty` was
//!   granted.
//! - Linux target: Landlock (see `landlock`): TCP bind and connect by port,
//!   then a seccomp-BPF filter on every thread (see [`seccomp`]), which
//!   refuses new IP sockets when the only network grant is a listener
//!   opened before entering.
//! - To do: Windows (AppContainer), FreeBSD (Capsicum), OpenBSD
//!   (pledge/unveil), iOS, Android; on these no OS target is entered and
//!   [`enter`] says so. The WASM and bare-metal targets are not OS layers
//!   and live outside this file (REMAINING-PLAN W1h-wasm, W1h-metal).
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

static ENABLED: AtomicBool = AtomicBool::new(false);

/// Called once by the `rce` binary's `main`.
pub fn enable() {
    ENABLED.store(true, Ordering::SeqCst);
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

/// How a granted address is used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GrantKind {
    /// the host connects to it (`--net`, `--peer`)
    Connect,
    /// the host binds it after entering the target (`--net`, a listener to retry)
    Listen,
    /// a listener bound before entering the target (`--http`, `--chan`):
    /// only its accepted connections are used afterwards
    Prebound,
}

/// One granted network address (HC-5, HC-5a).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct NetGrant {
    pub host: String,
    pub port: u16,
    pub loopback: bool,
    /// the host is an IP address, so no name is resolved
    pub numeric: bool,
    pub kind: GrantKind,
}

impl NetGrant {
    /// `HOST:PORT` (an IPv6 host in brackets); `None` when it has no port.
    pub fn parse(addr: &str, kind: GrantKind) -> Option<NetGrant> {
        let (h, p) = addr.rsplit_once(':')?;
        let port: u16 = p.parse().ok()?;
        let host = h.trim_start_matches('[').trim_end_matches(']').to_string();
        let ip: Option<std::net::IpAddr> = host.parse().ok();
        let loopback = ip.map_or(host == "localhost", |i| i.is_loopback());
        Some(NetGrant { host, port, loopback, numeric: ip.is_some(), kind })
    }
    pub fn addr(&self) -> String {
        if self.host.contains(':') { format!("[{}]:{}", self.host, self.port) } else { format!("{}:{}", self.host, self.port) }
    }
}

fn quote(p: &Path) -> String {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    format!("\"{}\"", c.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\""))
}

/// One network rule of the macOS target, from which both the SBPL text and
/// the printed description are made.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Rule {
    /// bind and accept on a port, at localhost or (`any`) at any address
    Inbound { port: u16, any: bool },
    /// connect to a port, at localhost or (`any`) at any address
    Outbound { port: u16, any: bool },
    /// the system resolver, for a grant that names a host by name
    Resolver,
}

impl Rule {
    fn sbpl(&self) -> String {
        let at = |port: &u16, any: &bool| format!("{}:{port}", if *any { "*" } else { "localhost" });
        match self {
            Rule::Inbound { port, any } => format!("(allow network-bind network-inbound (local ip \"{}\"))", at(port, any)),
            Rule::Outbound { port, any } => format!("(allow network-outbound (remote ip \"{}\"))", at(port, any)),
            Rule::Resolver => "(allow network-outbound (remote unix-socket (path-literal \"/private/var/run/mDNSResponder\")))".into(),
        }
    }
}

/// The network rules Seatbelt gets for the grants. Seatbelt names a host
/// only as `localhost` or `*`, so a grant that is not a loopback address
/// is allowed on its port at any address.
pub fn seatbelt_rules(grants: &[NetGrant]) -> Vec<Rule> {
    let mut rs: Vec<Rule> = vec![];
    for g in grants {
        let any = !g.loopback;
        rs.push(match g.kind {
            GrantKind::Connect => Rule::Outbound { port: g.port, any },
            GrantKind::Listen | GrantKind::Prebound => Rule::Inbound { port: g.port, any },
        });
        if !g.numeric {
            rs.push(Rule::Resolver);
        }
    }
    rs.sort();
    rs.dedup();
    rs
}

fn port_list(ps: &[u16]) -> String {
    format!("{ps:?}")
}

/// The description of the network rules actually installed (HC-5a).
pub fn seatbelt_description(rules: &[Rule]) -> String {
    let mut parts: Vec<String> = vec![];
    for (inbound, any) in [(true, false), (true, true), (false, false), (false, true)] {
        let ps: Vec<u16> = rules
            .iter()
            .filter_map(|r| match r {
                Rule::Inbound { port, any: a } if inbound && *a == any => Some(*port),
                Rule::Outbound { port, any: a } if !inbound && *a == any => Some(*port),
                _ => None,
            })
            .collect();
        if !ps.is_empty() {
            parts.push(format!("{} ports {} at {}", if inbound { "listening on" } else { "connecting to" }, port_list(&ps), if any { "any address" } else { "localhost" }));
        }
    }
    if parts.is_empty() {
        return "no network".into();
    }
    let resolver = if rules.contains(&Rule::Resolver) { ", and the name resolver" } else { "" };
    format!("network only {}{resolver} (TCP and UDP; Seatbelt cannot name other addresses; the net adapter checks the exact address)", parts.join(" and "))
}

/// The macOS target's Seatbelt profile (SBPL) for the given grants.
pub fn profile(log_dir: &Path, writable: &[&Path], pty: bool, grants: &[NetGrant]) -> String {
    let mut w = format!("(require-not (subpath {}))", quote(log_dir));
    for r in writable {
        w.push_str(&format!(" (require-not (subpath {}))", quote(r)));
    }
    let mut p = "(version 1)\n(allow default)\n(deny network*)\n".to_string();
    for r in seatbelt_rules(grants) {
        p.push_str(&r.sbpl());
        p.push('\n');
    }
    p.push_str(&format!("(deny file-write* (require-all (require-not (subpath \"/dev\")) {w}))\n"));
    if !pty {
        p.push_str("(deny process-exec*)\n");
    }
    p
}

/// Whether the host may still create IP sockets after entering: only when
/// some grant is used by connecting or binding after the target is entered.
pub fn needs_ip_sockets(grants: &[NetGrant]) -> bool {
    grants.iter().any(|g| g.kind != GrantKind::Prebound)
}

/// Enter this OS's enforcement target of the sandbox. `Ok(description)`
/// when entered, `Err(reason)` when there is none here yet.
pub fn enter(log_dir: &Path, writable: &[&Path], pty: bool, grants: &[NetGrant]) -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        use std::ffi::{CStr, CString, c_char, c_int};
        unsafe extern "C" {
            fn sandbox_init(profile: *const c_char, flags: u64, errorbuf: *mut *mut c_char) -> c_int;
            fn sandbox_free_error(errorbuf: *mut c_char);
        }
        let prof = CString::new(profile(log_dir, writable, pty, grants)).map_err(|e| e.to_string())?;
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: prof is a valid NUL-terminated string; err is freed below.
        let rc = unsafe { sandbox_init(prof.as_ptr(), 0, &mut err) };
        if rc != 0 {
            let msg = if err.is_null() { "sandbox_init failed".to_string() } else { unsafe { CStr::from_ptr(err) }.to_string_lossy().into_owned() };
            if !err.is_null() {
                unsafe { sandbox_free_error(err) };
            }
            return Err(msg);
        }
        Ok(format!("sandbox, macOS target (Seatbelt): {}, writes only under the log directory{}{}", seatbelt_description(&seatbelt_rules(grants)), if writable.is_empty() { "" } else { " and the granted roots" }, if pty { "" } else { ", no exec" }))
    }
    #[cfg(target_os = "linux")]
    {
        let net = needs_ip_sockets(grants);
        let ll = landlock::enter(log_dir, writable, pty, grants);
        // no_new_privs is set by landlock::enter; set it again in case
        // Landlock was unavailable, so the filter can still be installed.
        // SAFETY: plain prctl on our own process.
        unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        let sc = seccomp::install(pty, net);
        match (ll, sc) {
            (Ok(l), Ok(s)) => Ok(format!("sandbox, Linux target: {}; {s}", l.describe(grants, net))),
            (Err(l), Ok(s)) => Ok(format!("sandbox, Linux target: {s} (Landlock not entered: {l})")),
            (Ok(l), Err(s)) => Ok(format!("sandbox, Linux target: {} (seccomp not entered: {s})", l.describe(grants, true))),
            (Err(l), Err(s)) => Err(format!("Linux target: {l}; {s}")),
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (log_dir, writable, pty, grants);
        Err("no OS target for this platform yet (to do: Windows AppContainer, FreeBSD, OpenBSD, iOS, Android); the capability contract is still enforced by the checker and host".into())
    }
}

/// The Linux description of what Landlock installed (HC-5a). `abi` is the
/// Landlock ABI; `rules` the (port, bind, connect) rules added; `ip` says
/// whether seccomp leaves new IP sockets allowed (then UDP is not
/// restricted by any layer).
pub fn landlock_description(abi: i64, rules: &[(u16, bool, bool)], grants: &[NetGrant], ip: bool, writable: bool, pty: bool) -> String {
    let udp = if ip { ", UDP not restricted" } else { "" };
    let net = if abi < 4 {
        format!("network not restricted by Landlock (ABI {abi}){udp}")
    } else {
        let bind: Vec<u16> = rules.iter().filter(|r| r.1).map(|r| r.0).collect();
        let conn: Vec<u16> = rules.iter().filter(|r| r.2).map(|r| r.0).collect();
        let what = if bind.is_empty() && conn.is_empty() {
            "no TCP bind or connect".to_string()
        } else if bind == conn {
            format!("TCP bind and connect only on ports {}", port_list(&bind))
        } else {
            let b = if bind.is_empty() { "no TCP bind".to_string() } else { format!("TCP bind only on ports {}", port_list(&bind)) };
            let c = if conn.is_empty() { "no connect".to_string() } else { format!("connect only on ports {}", port_list(&conn)) };
            format!("{b} and {c}")
        };
        let adapter: Vec<String> = grants.iter().filter(|g| g.kind != GrantKind::Prebound).map(NetGrant::addr).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        let by_port = if rules.is_empty() { String::new() } else { format!(" (by port: any address on those ports; the net adapter allows only {})", adapter.join(", ")) };
        let pre: Vec<String> = grants.iter().filter(|g| g.kind == GrantKind::Prebound).map(NetGrant::addr).collect();
        let pre = if pre.is_empty() { String::new() } else { format!(" (the listener {} was opened before entering)", pre.join(", ")) };
        format!("{what}{by_port}{pre}{udp}")
    };
    format!("Landlock ABI {abi}: {net}, writes only under the log directory{}{}", if writable { " and the granted roots" } else { "" }, if pty { "" } else { ", no exec" })
}

/// Linux target, part 1: Landlock (kernel 5.13+), entered with raw syscalls. Writes are
/// handled (denied) everywhere but the log directory, the granted roots and
/// `/dev`; exec is handled (denied) unless `--pty`; on ABI 4+ TCP bind and
/// connect are handled: bind is allowed only on the ports of Listen grants
/// and connect only on the ports of Connect grants (a Prebound listener was
/// bound before, and accepting is not restricted). Reads are not
/// restricted. `no_new_privs` is set first.
#[cfg(target_os = "linux")]
mod landlock {
    use super::{GrantKind, NetGrant};
    use std::os::fd::AsRawFd;
    use std::path::Path;

    // Landlock's numbers are 444..446 in the unified syscall table; MIPS
    // offsets its tables (o32 +4000, n64 +5000).
    #[cfg(target_arch = "mips")]
    const BASE: libc::c_long = 4000;
    #[cfg(target_arch = "mips64")]
    const BASE: libc::c_long = 5000;
    #[cfg(not(any(target_arch = "mips", target_arch = "mips64")))]
    const BASE: libc::c_long = 0;
    const CREATE_RULESET: libc::c_long = BASE + 444;
    const ADD_RULE: libc::c_long = BASE + 445;
    const RESTRICT_SELF: libc::c_long = BASE + 446;
    const FLAG_VERSION: u32 = 1;
    const RULE_PATH_BENEATH: u32 = 1;
    const RULE_NET_PORT: u32 = 2;
    const EXECUTE: u64 = 1 << 0;
    // WRITE_FILE, REMOVE_DIR, REMOVE_FILE, MAKE_CHAR..MAKE_SYM
    const WRITES_V1: u64 = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7) | (1 << 8) | (1 << 9) | (1 << 10) | (1 << 11) | (1 << 12);
    const REFER: u64 = 1 << 13;
    const TRUNCATE: u64 = 1 << 14;
    const BIND_TCP: u64 = 1;
    const CONNECT_TCP: u64 = 2;

    #[repr(C)]
    struct RulesetAttr {
        fs: u64,
        net: u64,
    }
    #[repr(C, packed)]
    struct PathBeneath {
        allowed: u64,
        parent_fd: i32,
    }
    #[repr(C)]
    struct NetPort {
        allowed: u64,
        port: u64,
    }

    fn err(what: &str) -> String {
        format!("{what}: {}", std::io::Error::last_os_error())
    }

    /// What Landlock installed: its ABI and the port rules (port, bind, connect).
    pub struct Installed {
        abi: i64,
        rules: Vec<(u16, bool, bool)>,
        writable: bool,
        pty: bool,
    }

    impl Installed {
        pub fn describe(&self, grants: &[NetGrant], ip: bool) -> String {
            super::landlock_description(self.abi, &self.rules, grants, ip, self.writable, self.pty)
        }
    }

    pub fn enter(log_dir: &Path, writable: &[&Path], pty: bool, grants: &[NetGrant]) -> Result<Installed, String> {
        // SAFETY: the version query takes no attribute.
        let abi = unsafe { libc::syscall(CREATE_RULESET, std::ptr::null::<RulesetAttr>(), 0usize, FLAG_VERSION) };
        if abi < 1 {
            return Err(format!("Landlock unavailable ({})", std::io::Error::last_os_error()));
        }
        let mut writes = WRITES_V1;
        if abi >= 2 {
            writes |= REFER;
        }
        if abi >= 3 {
            writes |= TRUNCATE;
        }
        let fs = writes | if pty { 0 } else { EXECUTE };
        let net_handled = if abi >= 4 { BIND_TCP | CONNECT_TCP } else { 0 };
        let attr = RulesetAttr { fs, net: net_handled };
        let size = if abi >= 4 { std::mem::size_of::<RulesetAttr>() } else { 8 };
        // SAFETY: attr is valid for `size` bytes.
        let rs = unsafe { libc::syscall(CREATE_RULESET, &attr as *const RulesetAttr, size, 0u32) };
        if rs < 0 {
            return Err(err("landlock_create_ruleset"));
        }
        let rs = rs as i32;
        let mut dirs: Vec<&Path> = vec![log_dir, Path::new("/dev")];
        dirs.extend_from_slice(writable);
        for d in dirs {
            let f = std::fs::File::open(d).map_err(|e| format!("{}: {e}", d.display()))?;
            let rule = PathBeneath { allowed: writes, parent_fd: f.as_raw_fd() };
            // SAFETY: rule is a valid path-beneath attribute.
            if unsafe { libc::syscall(ADD_RULE, rs, RULE_PATH_BENEATH, &rule as *const PathBeneath, 0u32) } < 0 {
                return Err(err("landlock_add_rule"));
            }
        }
        // one rule per port: bind for Listen grants, connect for Connect grants
        let mut ports: std::collections::BTreeMap<u16, (bool, bool)> = Default::default();
        for g in grants {
            let e = ports.entry(g.port).or_default();
            match g.kind {
                GrantKind::Listen => e.0 = true,
                GrantKind::Connect => e.1 = true,
                GrantKind::Prebound => {}
            }
        }
        let mut rules = vec![];
        if abi >= 4 {
            for (p, (bind, connect)) in ports {
                if !bind && !connect {
                    continue;
                }
                let allowed = if bind { BIND_TCP } else { 0 } | if connect { CONNECT_TCP } else { 0 };
                let rule = NetPort { allowed, port: p as u64 };
                // SAFETY: rule is a valid net-port attribute.
                if unsafe { libc::syscall(ADD_RULE, rs, RULE_NET_PORT, &rule as *const NetPort, 0u32) } < 0 {
                    return Err(err("landlock_add_rule net"));
                }
                rules.push((p, bind, connect));
            }
        }
        // SAFETY: plain prctl and syscall on our own process.
        unsafe {
            if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                return Err(err("no_new_privs"));
            }
            if libc::syscall(RESTRICT_SELF, rs, 0u32) < 0 {
                return Err(err("landlock_restrict_self"));
            }
            libc::close(rs);
        }
        Ok(Installed { abi, rules, writable: !writable.is_empty(), pty })
    }
}
/// Linux: a seccomp-BPF filter, installed on every thread of the process
/// (`SECCOMP_FILTER_FLAG_TSYNC`, so a thread started before the sandbox is
/// covered too; Landlock is not, which is why the channel and HTTP
/// threads start after [`enter`]). It is a deny list on top of Landlock: syscalls an
/// RCE host never needs fail with EPERM: debugging other processes,
/// mounts and namespaces, kernel modules, kexec and reboot, BPF and perf,
/// keyrings, clock setting, io_uring (which would bypass the filter), and
/// on x86_64 the x32 ABI and port I/O. `clone` with any CLONE_NEW* flag is
/// refused and `clone3` answers ENOSYS (its flags cannot be inspected; libc
/// falls back to `clone`). Without `--pty`, execve and execveat fail;
/// unless a grant is connected to or bound after entering (a listener bound
/// before, such as `--http`, needs no new socket), `socket` for AF_INET and
/// AF_INET6 fails, which covers UDP and raw sockets too. A foreign
/// architecture kills the process.
///
/// The filter needs the audit architecture of the build target ([`arch`]).
/// The table covers every Linux architecture Rust targets that Landlock
/// runs on: x86_64, x86, aarch64, arm, riscv64, loongarch64, powerpc64
/// (both byte orders), s390x and MIPS (32 and 64, both byte orders). On any
/// other architecture the module still builds, [`install`] returns `Err`
/// naming the architecture, and the host reports "seccomp not entered", so
/// enforcement is reduced visibly, never silently.
#[cfg(target_os = "linux")]
pub mod seccomp {
    /// The `AUDIT_ARCH_*` value of the build target, its name, and whether a
    /// filter can be built for it (`None` outside the table).
    pub const fn arch() -> (Option<u32>, &'static str) {
        const LE: u32 = 0x4000_0000; // __AUDIT_ARCH_LE
        const B64: u32 = 0x8000_0000; // __AUDIT_ARCH_64BIT
        let le = if cfg!(target_endian = "little") { LE } else { 0 };
        let a = if cfg!(target_arch = "x86_64") {
            Some(B64 | LE | 62) // EM_X86_64
        } else if cfg!(target_arch = "x86") {
            Some(LE | 3) // EM_386
        } else if cfg!(target_arch = "aarch64") {
            Some(B64 | LE | 183) // EM_AARCH64
        } else if cfg!(target_arch = "arm") {
            Some(le | 40) // EM_ARM
        } else if cfg!(target_arch = "riscv64") {
            Some(B64 | LE | 243) // EM_RISCV
        } else if cfg!(target_arch = "loongarch64") {
            Some(B64 | LE | 258) // EM_LOONGARCH
        } else if cfg!(target_arch = "powerpc64") {
            Some(B64 | le | 21) // EM_PPC64
        } else if cfg!(target_arch = "s390x") {
            Some(B64 | 22) // EM_S390, big-endian only
        } else if cfg!(target_arch = "mips") {
            Some(le | 8) // EM_MIPS, o32
        } else if cfg!(target_arch = "mips64") {
            Some(B64 | le | 8) // EM_MIPS, n64
        } else {
            None
        };
        (a, std::env::consts::ARCH)
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64", target_arch = "arm", target_arch = "riscv64", target_arch = "loongarch64", target_arch = "powerpc64", target_arch = "s390x", target_arch = "mips", target_arch = "mips64"))]
    pub use filter::*;

    /// Outside the table: no filter, and saying so.
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64", target_arch = "arm", target_arch = "riscv64", target_arch = "loongarch64", target_arch = "powerpc64", target_arch = "s390x", target_arch = "mips", target_arch = "mips64")))]
    pub fn install(_pty: bool, _net: bool) -> Result<String, String> {
        Err(format!("seccomp: no audit-architecture table for {}, filter not installed", arch().1))
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64", target_arch = "arm", target_arch = "riscv64", target_arch = "loongarch64", target_arch = "powerpc64", target_arch = "s390x", target_arch = "mips", target_arch = "mips64"))]
    mod filter {
    use libc::sock_filter;

    const LD_W_ABS: u16 = 0x20;
    const JEQ_K: u16 = 0x15;
    const JSET_K: u16 = 0x45;
    const RET_K: u16 = 0x06;
    const RET_ALLOW: u32 = 0x7fff_0000;
    const RET_ERRNO: u32 = 0x0005_0000;
    const RET_KILL_PROCESS: u32 = 0x8000_0000;
    const OFF_NR: u32 = 0;
    const OFF_ARCH: u32 = 4;
    /// `seccomp_data.args[i]` is a u64 at offset 16 + 8i; BPF loads 32
    /// bits, and the low half sits at +0 on little-endian, +4 on big-endian.
    pub const fn off_arg_lo(i: u32) -> u32 {
        16 + 8 * i + if cfg!(target_endian = "big") { 4 } else { 0 }
    }
    const OFF_ARG0: u32 = off_arg_lo(0);
    /// The argument of `clone` that carries the flags: s390 swaps the first
    /// two (CLONE_BACKWARDS2), every other table architecture passes them first.
    pub const OFF_CLONE_FLAGS: u32 = off_arg_lo(if cfg!(target_arch = "s390x") { 1 } else { 0 });
    const AUDIT_ARCH: u32 = match super::arch().0 { Some(a) => a, None => 0 };
    const CLONE_NEW: u32 = 0x0002_0000 | 0x0200_0000 | 0x0400_0000 | 0x0800_0000 | 0x1000_0000 | 0x2000_0000 | 0x4000_0000 | 0x0000_0080;

    fn st(code: u16, k: u32) -> sock_filter {
        sock_filter { code, jt: 0, jf: 0, k }
    }
    fn jmp(code: u16, k: u32, jt: u8, jf: u8) -> sock_filter {
        sock_filter { code, jt, jf, k }
    }

    /// Syscalls refused whatever was granted.
    pub fn denied() -> Vec<libc::c_long> {
        use libc::*;
        #[allow(unused_mut)]
        let mut v = vec![
            SYS_ptrace, SYS_process_vm_readv, SYS_process_vm_writev, SYS_mount, SYS_umount2, SYS_pivot_root, SYS_chroot,
            SYS_kexec_load, SYS_init_module, SYS_finit_module, SYS_delete_module, SYS_reboot,
            SYS_swapon, SYS_swapoff, SYS_bpf, SYS_perf_event_open, SYS_userfaultfd, SYS_keyctl, SYS_add_key,
            SYS_request_key, SYS_unshare, SYS_setns, SYS_personality, SYS_acct, SYS_quotactl, SYS_open_by_handle_at,
            SYS_name_to_handle_at, SYS_settimeofday, SYS_clock_settime, SYS_clock_adjtime, SYS_adjtimex, SYS_syslog,
            SYS_fsopen, SYS_fsmount, SYS_fsconfig, SYS_fspick, SYS_move_mount, SYS_open_tree, SYS_mount_setattr,
            SYS_io_uring_setup, SYS_io_uring_enter, SYS_io_uring_register, SYS_pidfd_getfd,
        ];
        // kexec_file_load: libc names it where it has the number; riscv64
        // (294) and s390x (381) have the syscall but not the libc name; x86
        // (32-bit) has no such syscall.
        #[cfg(not(any(target_arch = "riscv64", target_arch = "s390x", target_arch = "x86")))]
        v.push(SYS_kexec_file_load);
        #[cfg(target_arch = "riscv64")]
        v.push(294);
        #[cfg(target_arch = "s390x")]
        v.push(381);
        #[cfg(target_arch = "x86_64")]
        v.extend([SYS_iopl, SYS_ioperm, SYS_uselib, SYS_ustat, SYS_sysfs, SYS_vhangup, SYS_modify_ldt]);
        v
    }

    /// The filter program for the given grants.
    pub fn program(pty: bool, net: bool) -> Vec<sock_filter> {
        let eperm = RET_ERRNO | libc::EPERM as u32;
        let mut p = vec![
            st(LD_W_ABS, OFF_ARCH),
            jmp(JEQ_K, AUDIT_ARCH, 1, 0),
            st(RET_K, RET_KILL_PROCESS),
            st(LD_W_ABS, OFF_NR),
        ];
        #[cfg(target_arch = "x86_64")]
        {
            // the x32 ABI: nr | 0x40000000
            p.push(jmp(0x35 /* JGE_K */, 0x4000_0000, 0, 1));
            p.push(st(RET_K, eperm));
        }
        let mut deny = denied();
        if !pty {
            deny.extend([libc::SYS_execve, libc::SYS_execveat]);
        }
        for nr in deny {
            p.push(jmp(JEQ_K, nr as u32, 0, 1));
            p.push(st(RET_K, eperm));
        }
        p.push(jmp(JEQ_K, libc::SYS_clone3 as u32, 0, 1));
        p.push(st(RET_K, RET_ERRNO | libc::ENOSYS as u32));
        p.push(jmp(JEQ_K, libc::SYS_clone as u32, 0, 4));
        p.push(st(LD_W_ABS, OFF_CLONE_FLAGS));
        p.push(jmp(JSET_K, CLONE_NEW, 0, 1));
        p.push(st(RET_K, eperm));
        p.push(st(RET_K, RET_ALLOW));
        #[cfg(any(target_arch = "x86", target_arch = "s390x", target_arch = "powerpc64", target_arch = "mips"))]
        if !net {
            // socketcall multiplexes socket(2); its domain sits behind a
            // pointer the filter cannot read, so without a network grant it
            // is refused whole (libc uses the direct calls on these kernels).
            p.push(jmp(JEQ_K, libc::SYS_socketcall as u32, 0, 1));
            p.push(st(RET_K, eperm));
        }
        if !net {
            p.push(jmp(JEQ_K, libc::SYS_socket as u32, 0, 5));
            p.push(st(LD_W_ABS, OFF_ARG0));
            p.push(jmp(JEQ_K, libc::AF_INET as u32, 1, 0));
            p.push(jmp(JEQ_K, libc::AF_INET6 as u32, 0, 1));
            p.push(st(RET_K, eperm));
            p.push(st(RET_K, RET_ALLOW));
        }
        p.push(st(RET_K, RET_ALLOW));
        p
    }

    /// Install the filter on every thread. Needs no_new_privs (set by the
    /// caller) or CAP_SYS_ADMIN.
    pub fn install(pty: bool, net: bool) -> Result<String, String> {
        let prog = program(pty, net);
        let fprog = libc::sock_fprog { len: prog.len() as u16, filter: prog.as_ptr() as *mut sock_filter };
        // SAFETY: fprog points at `prog`, which outlives the call; the
        // kernel copies the program.
        let rc = unsafe {
            libc::syscall(libc::SYS_seccomp, libc::SECCOMP_SET_MODE_FILTER, libc::SECCOMP_FILTER_FLAG_TSYNC, &fprog as *const libc::sock_fprog)
        };
        if rc != 0 {
            return Err(format!("seccomp: {}", if rc > 0 { format!("thread {rc} could not be synchronised") } else { std::io::Error::last_os_error().to_string() }));
        }
        Ok(format!("seccomp filter ({} instructions) on all threads{}{}", prog.len(), if pty { "" } else { ", no exec" }, if net { "" } else { ", no new IP sockets" }))
    }
}
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::seccomp;

    /// The table against the kernel's `AUDIT_ARCH_*` values (include/uapi/linux/audit.h).
    #[test]
    fn audit_arch_matches_the_kernel_header() {
        let want: Option<u32> = if cfg!(target_arch = "x86_64") {
            Some(0xC000_003E)
        } else if cfg!(target_arch = "x86") {
            Some(0x4000_0003)
        } else if cfg!(target_arch = "aarch64") {
            Some(0xC000_00B7)
        } else if cfg!(target_arch = "arm") {
            Some(0x4000_0028)
        } else if cfg!(target_arch = "riscv64") {
            Some(0xC000_00F3)
        } else if cfg!(target_arch = "loongarch64") {
            Some(0xC000_0102)
        } else if cfg!(all(target_arch = "powerpc64", target_endian = "little")) {
            Some(0xC000_0015)
        } else if cfg!(target_arch = "powerpc64") {
            Some(0x8000_0015)
        } else if cfg!(target_arch = "s390x") {
            Some(0x8000_0016)
        } else {
            seccomp::arch().0
        };
        assert_eq!(seccomp::arch().0, want);
    }

    /// The low 32 bits of each argument, by byte order (s390x: +4).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "s390x", target_arch = "powerpc64", target_arch = "riscv64"))]
    #[test]
    fn argument_offsets_follow_byte_order() {
        let be = cfg!(target_endian = "big");
        assert_eq!(seccomp::off_arg_lo(0), if be { 20 } else { 16 });
        assert_eq!(seccomp::off_arg_lo(1), if be { 28 } else { 24 });
        assert_eq!(seccomp::OFF_CLONE_FLAGS, if cfg!(target_arch = "s390x") { 28 } else { 16 });
    }
}
