//! HC-5 (W1h): after the log and the granted adapters are open, the host
//! process also enters the sandbox's OS enforcement target. There is one RCE
//! sandbox with one contract (HC-5 capability checking), independent of the
//! OS: the checker and the deterministic host enforce it before any OS layer.
//! Underneath, an enforcement target makes the host process itself hold no
//! more than the grants: no network but the granted addresses, writes only
//! under the log directory and the granted roots, no exec unless a PTY is
//! granted. Entering it
//! changes no meaning: the program has no ambient authority anyway (5.15);
//! the sandbox removes the host's own.
//!
//! Only the `rce` binary enables this ([`enable`]); library callers and
//! in-process tests are never sandboxed, since a sandbox is per process and
//! cannot be left.
//!
//! OS enforcement targets in this file:
//! - macOS target (Seatbelt): `sandbox_init` with a profile that denies all network access,
//!   (unless `--net` grants addresses; the net adapter then checks each
//!   address), denies writes outside the log's directory, the granted roots and
//!   `/dev`, and denies process exec unless `--pty` was granted.
//! - Linux target: Landlock (see `landlock`), then a seccomp-BPF filter on every
//!   thread (see [`seccomp`]).
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

fn quote(p: &Path) -> String {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    format!("\"{}\"", c.to_string_lossy().replace('\\', "\\\\").replace('"', "\\\""))
}

/// The macOS target's Seatbelt profile (SBPL) for the given grants.
pub fn profile(log_dir: &Path, writable: &[&Path], pty: bool, net: bool) -> String {
    let mut w = format!("(require-not (subpath {}))", quote(log_dir));
    for r in writable {
        w.push_str(&format!(" (require-not (subpath {}))", quote(r)));
    }
    let mut p = "(version 1)\n(allow default)\n".to_string();
    if !net {
        p.push_str("(deny network*)\n");
    }
    p.push_str(&format!("(deny file-write* (require-all (require-not (subpath \"/dev\")) {w}))\n"));
    if !pty {
        p.push_str("(deny process-exec*)\n");
    }
    p
}

/// Enter this OS's enforcement target of the sandbox. `Ok(description)`
/// when entered, `Err(reason)` when there is none here yet.
pub fn enter(log_dir: &Path, writable: &[&Path], pty: bool, net: bool, ports: &[u16]) -> Result<String, String> {
    #[cfg(target_os = "macos")]
    {
        let _ = ports;
        use std::ffi::{CStr, CString, c_char, c_int};
        unsafe extern "C" {
            fn sandbox_init(profile: *const c_char, flags: u64, errorbuf: *mut *mut c_char) -> c_int;
            fn sandbox_free_error(errorbuf: *mut c_char);
        }
        let prof = CString::new(profile(log_dir, writable, pty, net)).map_err(|e| e.to_string())?;
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
        Ok(format!("sandbox, macOS target (Seatbelt): {}, writes only under the log directory{}{}", if net { "network only to granted addresses (--net, --chan, --peer)" } else { "no network" }, if writable.is_empty() { "" } else { " and the granted roots" }, if pty { "" } else { ", no exec" }))
    }
    #[cfg(target_os = "linux")]
    {
        let ll = landlock::enter(log_dir, writable, pty, net, ports);
        // no_new_privs is set by landlock::enter; set it again in case
        // Landlock was unavailable, so the filter can still be installed.
        // SAFETY: plain prctl on our own process.
        unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) };
        let sc = seccomp::install(pty, net);
        match (ll, sc) {
            (Ok(l), Ok(s)) => Ok(format!("sandbox, Linux target: {l}; {s}")),
            (Err(l), Ok(s)) => Ok(format!("sandbox, Linux target: {s} (Landlock not entered: {l})")),
            (Ok(l), Err(s)) => Ok(format!("sandbox, Linux target: {l} (seccomp not entered: {s})")),
            (Err(l), Err(s)) => Err(format!("Linux target: {l}; {s}")),
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        let _ = (log_dir, writable, pty, net, ports);
        Err("no OS target for this platform yet (to do: Windows AppContainer, FreeBSD, OpenBSD, iOS, Android); the capability contract is still enforced by the checker and host".into())
    }
}

/// Linux target, part 1: Landlock (kernel 5.13+), entered with raw syscalls. Writes are
/// handled (denied) everywhere but the log directory, the granted roots and
/// `/dev`; exec is handled (denied) unless `--pty`; on ABI 4+ TCP bind and
/// connect are handled and allowed only on the granted ports (none without
/// a network grant). Reads are not restricted. `no_new_privs` is set first.
#[cfg(target_os = "linux")]
mod landlock {
    use std::os::fd::AsRawFd;
    use std::path::Path;

    const CREATE_RULESET: libc::c_long = 444;
    const ADD_RULE: libc::c_long = 445;
    const RESTRICT_SELF: libc::c_long = 446;
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

    pub fn enter(log_dir: &Path, writable: &[&Path], pty: bool, net: bool, ports: &[u16]) -> Result<String, String> {
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
        if abi >= 4 && net {
            for p in ports {
                let rule = NetPort { allowed: BIND_TCP | CONNECT_TCP, port: *p as u64 };
                // SAFETY: rule is a valid net-port attribute.
                if unsafe { libc::syscall(ADD_RULE, rs, RULE_NET_PORT, &rule as *const NetPort, 0u32) } < 0 {
                    return Err(err("landlock_add_rule net"));
                }
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
        let netd = if abi < 4 {
            "network not restricted (Landlock ABI < 4)".to_string()
        } else if net {
            format!("TCP only on granted ports {ports:?}")
        } else {
            "no TCP".to_string()
        };
        Ok(format!("Landlock ABI {abi}: {netd}, writes only under the log directory{}{}", if writable.is_empty() { "" } else { " and the granted roots" }, if pty { "" } else { ", no exec" }))
    }
}

/// Linux: a seccomp-BPF filter, installed on every thread of the process
/// (`SECCOMP_FILTER_FLAG_TSYNC`, so adapter threads started before the
/// sandbox are covered). It is a deny list on top of Landlock: syscalls an
/// RCE host never needs fail with EPERM: debugging other processes,
/// mounts and namespaces, kernel modules, kexec and reboot, BPF and perf,
/// keyrings, clock setting, io_uring (which would bypass the filter), and
/// on x86_64 the x32 ABI and port I/O. `clone` with any CLONE_NEW* flag is
/// refused and `clone3` answers ENOSYS (its flags cannot be inspected; libc
/// falls back to `clone`). Without `--pty`, execve and execveat fail;
/// without a network grant, `socket` for AF_INET and AF_INET6 fails. A
/// foreign architecture kills the process.
#[cfg(target_os = "linux")]
pub mod seccomp {
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
    const OFF_ARG0: u32 = 16; // low 32 bits on little-endian targets
    #[cfg(target_arch = "x86_64")]
    const AUDIT_ARCH: u32 = 0xC000_003E;
    #[cfg(target_arch = "aarch64")]
    const AUDIT_ARCH: u32 = 0xC000_00B7;
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
            SYS_kexec_load, SYS_kexec_file_load, SYS_init_module, SYS_finit_module, SYS_delete_module, SYS_reboot,
            SYS_swapon, SYS_swapoff, SYS_bpf, SYS_perf_event_open, SYS_userfaultfd, SYS_keyctl, SYS_add_key,
            SYS_request_key, SYS_unshare, SYS_setns, SYS_personality, SYS_acct, SYS_quotactl, SYS_open_by_handle_at,
            SYS_name_to_handle_at, SYS_settimeofday, SYS_clock_settime, SYS_clock_adjtime, SYS_adjtimex, SYS_syslog,
            SYS_fsopen, SYS_fsmount, SYS_fsconfig, SYS_fspick, SYS_move_mount, SYS_open_tree, SYS_mount_setattr,
            SYS_io_uring_setup, SYS_io_uring_enter, SYS_io_uring_register, SYS_pidfd_getfd,
        ];
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
        p.push(st(LD_W_ABS, OFF_ARG0));
        p.push(jmp(JSET_K, CLONE_NEW, 0, 1));
        p.push(st(RET_K, eperm));
        p.push(st(RET_K, RET_ALLOW));
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
        Ok(format!("seccomp filter ({} instructions) on all threads{}{}", prog.len(), if pty { "" } else { ", no exec" }, if net { "" } else { ", no IP sockets" }))
    }
}
