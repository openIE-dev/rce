//! The Linux seccomp filter (HC-5): each child is forked, installs the
//! filter and probes syscalls; the parent reads the verdicts from the exit
//! status. In-process tests are otherwise never sandboxed.
#![cfg(target_os = "linux")]
use rce_spec_check::sandbox::seccomp;

/// Fork; the child sets no_new_privs, installs the filter with the grants,
/// runs `probe` and exits with its result. Returns the child's exit code.
fn in_child(pty: bool, net: bool, probe: fn() -> i32) -> i32 {
    // SAFETY: the child only makes raw syscalls and calls _exit.
    unsafe {
        let pid = libc::fork();
        assert!(pid >= 0);
        if pid == 0 {
            libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            let code = if seccomp::install(pty, net).is_ok() { probe() } else { 99 };
            libc::_exit(code);
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        assert!(libc::WIFEXITED(st), "child died: status {st}");
        libc::WEXITSTATUS(st)
    }
}

fn eperm(rc: libc::c_long) -> bool {
    rc == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[test]
fn dangerous_syscalls_fail_and_ordinary_ones_work() {
    let code = in_child(false, false, || unsafe {
        let mut bad = 0;
        if !eperm(libc::syscall(libc::SYS_ptrace, libc::PTRACE_TRACEME, 0, 0, 0)) { bad |= 1; }
        if !eperm(libc::syscall(libc::SYS_unshare, libc::CLONE_NEWUSER)) { bad |= 2; }
        if !eperm(libc::syscall(libc::SYS_socket, libc::AF_INET, libc::SOCK_STREAM, 0)) { bad |= 4; }
        if !eperm(libc::syscall(libc::SYS_io_uring_setup, 1, std::ptr::null_mut::<u8>())) { bad |= 8; }
        let sh = c"/bin/true";
        let argv = [sh.as_ptr(), std::ptr::null()];
        if !eperm(libc::syscall(libc::SYS_execve, sh.as_ptr(), argv.as_ptr(), std::ptr::null::<*const i8>())) { bad |= 16; }
        // ordinary work still runs: a unix socket, a file, a thread
        let u = libc::socket(libc::AF_UNIX, libc::SOCK_STREAM, 0);
        if u < 0 { bad |= 32; } else { libc::close(u); }
        if libc::getpid() <= 0 { bad |= 64; }
        if std::thread::spawn(|| 7).join().ok() != Some(7) { bad |= 128; }
        bad
    });
    assert_eq!(code, 0, "bitmask of probes that did not behave");
}

#[test]
fn grants_open_exec_and_ip_sockets() {
    let code = in_child(true, true, || unsafe {
        let mut bad = 0;
        let s = libc::socket(libc::AF_INET, libc::SOCK_STREAM, 0);
        if s < 0 { bad |= 1; } else { libc::close(s); }
        if !eperm(libc::syscall(libc::SYS_ptrace, libc::PTRACE_TRACEME, 0, 0, 0)) { bad |= 2; }
        let pid = libc::fork();
        if pid == 0 {
            let t = c"/bin/true";
            let argv = [t.as_ptr(), std::ptr::null()];
            libc::execv(t.as_ptr(), argv.as_ptr());
            libc::_exit(42);
        }
        let mut st = 0;
        libc::waitpid(pid, &mut st, 0);
        if !(libc::WIFEXITED(st) && libc::WEXITSTATUS(st) == 0) { bad |= 4; }
        bad
    });
    assert_eq!(code, 0, "bitmask of probes that did not behave");
}
