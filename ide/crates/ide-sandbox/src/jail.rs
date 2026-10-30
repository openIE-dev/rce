//! Native process jail.
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, Command, Output};

#[derive(Debug, thiserror::Error)]
pub enum JailError {
    /// The platform cannot enforce the policy. Nothing was run.
    #[error("jail unavailable, not run: {0}")]
    Unavailable(String),
    #[error("spawn: {0}")]
    Spawn(std::io::Error),
}

/// What a jailed process may touch.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JailPolicy {
    /// Read and write.
    pub rw: Vec<PathBuf>,
    /// Read and execute only.
    pub ro: Vec<PathBuf>,
    /// Unrestricted network. `false` means only `tcp_connect` below.
    pub allow_net: bool,
    /// With `allow_net` false: TCP ports the process may connect to
    /// (Landlock `ConnectTcp` rules; no bind anywhere). Seccomp then allows
    /// only AF_INET/AF_INET6 stream sockets, so UDP, raw and Unix sockets
    /// stay denied. Landlock filters by port, not host: host restriction is
    /// the proxy's job (point the declared port at the proxy).
    pub tcp_connect: Vec<u16>,
}

impl JailPolicy {
    /// A native program confined to `workspace` (read/write) plus what it
    /// needs to start: its own binaries (`exes`, read/execute) and the
    /// dynamic loader and shared libraries (`/lib*`, `/usr/lib*`,
    /// `/etc/ld.so.cache`). Nothing else is readable: not $HOME, not /etc,
    /// not other projects. `/dev/null` is writable. No network.
    pub fn for_program(workspace: &Path, exes: &[PathBuf]) -> JailPolicy {
        let mut ro: Vec<PathBuf> = exes.to_vec();
        for d in [
            "/lib",
            "/lib64",
            "/usr/lib",
            "/usr/lib64",
            "/etc/ld.so.cache",
        ] {
            let d = PathBuf::from(d);
            if d.exists() && !ro.contains(&d) {
                ro.push(d);
            }
        }
        JailPolicy {
            rw: vec![workspace.to_path_buf(), "/dev/null".into()],
            ro,
            allow_net: false,
            tcp_connect: vec![],
        }
    }
}

pub trait Jail {
    fn name(&self) -> &'static str;
    /// Check that the policy can be fully enforced here, without running anything.
    fn probe(&self, policy: &JailPolicy) -> Result<(), JailError>;
    /// Run `cmd` under the policy. Fails closed: if any rule cannot be
    /// enforced, returns `Unavailable` and the command never starts.
    fn run(&self, cmd: Command, policy: &JailPolicy) -> Result<Output, JailError>;
    /// Start `cmd` under the policy and return the running child (stdio as
    /// configured on `cmd`). Same fail-closed rule as [`Jail::run`].
    fn spawn(&self, cmd: Command, policy: &JailPolicy) -> Result<Child, JailError>;
}

/// A platform with no jail backend: never runs anything.
pub struct NoJail;
impl Jail for NoJail {
    fn name(&self) -> &'static str {
        "none"
    }
    fn probe(&self, _: &JailPolicy) -> Result<(), JailError> {
        Err(JailError::Unavailable(
            "no jail backend for this platform".into(),
        ))
    }
    fn run(&self, _: Command, p: &JailPolicy) -> Result<Output, JailError> {
        self.probe(p).map(|_| unreachable!())
    }
    fn spawn(&self, _: Command, p: &JailPolicy) -> Result<Child, JailError> {
        self.probe(p).map(|_| unreachable!())
    }
}

/// Confine the current process (not a child) under `policy`, for a
/// long-running server that has already bound its listening socket.
/// Fails closed: an error means nothing should be served.
pub fn restrict_self(policy: &JailPolicy) -> Result<&'static str, JailError> {
    #[cfg(target_os = "linux")]
    {
        linux::restrict_self(policy).map(|_| "landlock+seccomp")
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = policy;
        Err(JailError::Unavailable("no self-jail backend for this platform".into()))
    }
}

/// The best jail for this platform.
pub fn platform_jail() -> Box<dyn Jail> {
    #[cfg(target_os = "linux")]
    {
        Box::new(linux::LandlockJail)
    }
    #[cfg(not(target_os = "linux"))]
    {
        Box::new(NoJail)
    }
}

#[cfg(target_os = "linux")]
pub mod linux {
    //! Landlock ABI v4 (filesystem plus TCP bind/connect) as a hard
    //! requirement, then a seccomp filter that denies socket creation,
    //! io_uring, ptrace, bpf, mounts, namespaces, keyrings and module loading.
    use super::*;
    use landlock::{
        ABI, Access, AccessFs, AccessNet, CompatLevel, Compatible, NetPort, Ruleset, RulesetAttr,
        RulesetCreated, RulesetCreatedAttr, RulesetStatus, path_beneath_rules,
    };
    use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, TargetArch};
    use seccompiler::{SeccompCmpArgLen, SeccompCmpOp, SeccompCondition, SeccompRule};
    use std::collections::BTreeMap;
    use std::os::unix::process::CommandExt;
    use std::sync::Mutex;

    pub const ABI_REQUIRED: ABI = ABI::V4;

    /// Syscalls denied with EPERM. Socket creation is always denied when
    /// `allow_net` is false; the rest always.
    pub fn denied_syscalls(allow_net: bool) -> Vec<i64> {
        denied(allow_net, false)
    }

    fn denied(allow_net: bool, tcp_only: bool) -> Vec<i64> {
        let mut v = vec![
            libc::SYS_io_uring_setup,
            libc::SYS_io_uring_enter,
            libc::SYS_io_uring_register,
            libc::SYS_ptrace,
            libc::SYS_process_vm_writev,
            libc::SYS_process_vm_readv,
            libc::SYS_bpf,
            libc::SYS_mount,
            libc::SYS_umount2,
            libc::SYS_pivot_root,
            libc::SYS_unshare,
            libc::SYS_setns,
            libc::SYS_keyctl,
            libc::SYS_add_key,
            libc::SYS_request_key,
            libc::SYS_init_module,
            libc::SYS_finit_module,
            libc::SYS_kexec_load,
        ];
        if !allow_net {
            v.push(libc::SYS_socketpair);
            if !tcp_only {
                v.push(libc::SYS_socket);
            }
        }
        v
    }

    /// `socket()` denied unless the domain is AF_INET or AF_INET6 and
    /// `type & 0xf` is SOCK_STREAM (flags like SOCK_CLOEXEC are allowed).
    fn tcp_only_socket_rules() -> Result<Vec<SeccompRule>, JailError> {
        let e = |e: seccompiler::BackendError| JailError::Unavailable(format!("seccomp rule: {e}"));
        let c = |arg, op, v| SeccompCondition::new(arg, SeccompCmpArgLen::Dword, op, v).map_err(e);
        let mut v = vec![
            SeccompRule::new(vec![
                c(0, SeccompCmpOp::Ne, libc::AF_INET as u64)?,
                c(0, SeccompCmpOp::Ne, libc::AF_INET6 as u64)?,
            ])
            .map_err(e)?,
        ];
        for t in (0u64..16).filter(|t| *t != libc::SOCK_STREAM as u64) {
            v.push(SeccompRule::new(vec![c(1, SeccompCmpOp::MaskedEq(0xf), t)?]).map_err(e)?);
        }
        Ok(v)
    }

    pub fn seccomp_program(allow_net: bool) -> Result<BpfProgram, JailError> {
        seccomp_for(allow_net, &[])
    }

    pub fn seccomp_for(allow_net: bool, tcp_connect: &[u16]) -> Result<BpfProgram, JailError> {
        let tcp_only = !allow_net && !tcp_connect.is_empty();
        let arch: TargetArch = std::env::consts::ARCH
            .try_into()
            .map_err(|e| JailError::Unavailable(format!("seccomp arch: {e:?}")))?;
        let mut rules: BTreeMap<i64, Vec<seccompiler::SeccompRule>> = denied(allow_net, tcp_only)
            .into_iter()
            .map(|s| (s, vec![]))
            .collect();
        if tcp_only {
            rules.insert(libc::SYS_socket, tcp_only_socket_rules()?);
        }
        let f = SeccompFilter::new(
            rules,
            SeccompAction::Allow,
            SeccompAction::Errno(libc::EPERM as u32),
            arch,
        )
        .map_err(|e| JailError::Unavailable(format!("seccomp filter: {e}")))?;
        f.try_into().map_err(|e: seccompiler::BackendError| {
            JailError::Unavailable(format!("seccomp compile: {e}"))
        })
    }

    pub fn ruleset(p: &JailPolicy) -> Result<RulesetCreated, JailError> {
        let u = |e: landlock::RulesetError| JailError::Unavailable(format!("landlock: {e}"));
        let mut r = Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessFs::from_all(ABI_REQUIRED))
            .map_err(u)?;
        if !p.allow_net {
            // Handled with no allow rules: every TCP bind and connect is denied.
            r = r
                .handle_access(AccessNet::from_all(ABI_REQUIRED))
                .map_err(u)?;
        }
        let mut c = r.create().map_err(u)?;
        if !p.allow_net {
            for port in &p.tcp_connect {
                c = c
                    .add_rule(NetPort::new(*port, AccessNet::ConnectTcp))
                    .map_err(u)?;
            }
        }
        c = c
            .add_rules(path_beneath_rules(&p.ro, AccessFs::from_read(ABI_REQUIRED)))
            .map_err(u)?;
        c = c
            .add_rules(path_beneath_rules(&p.rw, AccessFs::from_all(ABI_REQUIRED)))
            .map_err(u)?;
        Ok(c)
    }

    pub struct LandlockJail;

    impl Jail for LandlockJail {
        fn name(&self) -> &'static str {
            "landlock+seccomp"
        }
        fn probe(&self, p: &JailPolicy) -> Result<(), JailError> {
            for d in p.ro.iter().chain(&p.rw) {
                if !d.exists() {
                    return Err(JailError::Unavailable(format!(
                        "{}: does not exist",
                        d.display()
                    )));
                }
            }
            ruleset(p)?;
            seccomp_for(p.allow_net, &p.tcp_connect)?;
            Ok(())
        }
        fn run(&self, mut cmd: Command, p: &JailPolicy) -> Result<Output, JailError> {
            confine(&mut cmd, p)?;
            cmd.output().map_err(spawn_err)
        }
        fn spawn(&self, mut cmd: Command, p: &JailPolicy) -> Result<Child, JailError> {
            confine(&mut cmd, p)?;
            cmd.spawn().map_err(spawn_err)
        }
    }

    /// Restrict this process: Landlock fully enforced (else error), then
    /// the seccomp filter. Irreversible for the life of the process.
    pub fn restrict_self(p: &JailPolicy) -> Result<(), JailError> {
        LandlockJail.probe(p)?;
        let prog = seccomp_for(p.allow_net, &p.tcp_connect)?;
        let st = ruleset(p)?
            .restrict_self()
            .map_err(|e| JailError::Unavailable(format!("landlock restrict_self: {e}")))?;
        if st.ruleset != RulesetStatus::FullyEnforced || !st.no_new_privs {
            return Err(JailError::Unavailable("landlock not fully enforced".into()));
        }
        seccompiler::apply_filter_all_threads(&prog)
            .map_err(|e| JailError::Unavailable(format!("seccomp: {e}")))?;
        Ok(())
    }

    fn spawn_err(e: std::io::Error) -> JailError {
        if e.raw_os_error() == Some(libc::EPERM) {
            JailError::Unavailable(format!("restriction failed in child: {e}"))
        } else {
            JailError::Spawn(e)
        }
    }

    /// Arrange for the child to restrict itself before exec.
    fn confine(cmd: &mut Command, p: &JailPolicy) -> Result<(), JailError> {
        LandlockJail.probe(p)?;
        let rs = Mutex::new(Some(ruleset(p)?));
        let prog = seccomp_for(p.allow_net, &p.tcp_connect)?;
        // SAFETY: runs in the forked child before exec. It takes the
        // prepared ruleset, makes the landlock_restrict_self and
        // prctl/seccomp calls, and returns an error (which aborts the
        // spawn) if Landlock is not fully enforced.
        unsafe {
            cmd.pre_exec(move || {
                let r = rs
                    .lock()
                    .ok()
                    .and_then(|mut g| g.take())
                    .ok_or_else(|| std::io::Error::other("ruleset taken"))?;
                let st = r
                    .restrict_self()
                    .map_err(|_| std::io::Error::from_raw_os_error(libc::EPERM))?;
                if st.ruleset != RulesetStatus::FullyEnforced || !st.no_new_privs {
                    return Err(std::io::Error::from_raw_os_error(libc::EPERM));
                }
                seccompiler::apply_filter(&prog)
                    .map_err(|_| std::io::Error::from_raw_os_error(libc::EPERM))?;
                Ok(())
            });
        }
        Ok(())
    }
}
