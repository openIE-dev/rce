//! Jail for `rce-mcp --http`: after the listener is bound, the server
//! confines itself (Landlock + seccomp on Linux, fail closed).
//!
//! Hosts: the server only listens; it never dials out. The bind address
//! must be a declared host (loopback always; anything else only with
//! `--allow-host`). Outbound TCP is limited to the ports of endpoints the
//! workspace declares in `.rce-ide/net.json`; Landlock filters by port, so
//! host filtering for those endpoints stays with the ide-sandbox proxy.
//! Files: the workspace and the ledger/key directories read-write; shared
//! libraries, `git` and the read-only energy counters read-only.
use ide_sandbox::jail::JailPolicy;
use std::net::{IpAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};

/// Refuse a bind address that is not a declared host.
pub fn check_bind(addr: &str, allowed_hosts: &[String]) -> Result<(), String> {
    let host = addr.rsplit_once(':').map(|(h, _)| h.trim_matches(['[', ']'])).unwrap_or(addr);
    if allowed_hosts.iter().any(|h| h == host) {
        return Ok(());
    }
    let ips: Vec<IpAddr> = addr.to_socket_addrs().map_err(|e| format!("{addr}: {e}"))?.map(|s| s.ip()).collect();
    if !ips.is_empty() && ips.iter().all(IpAddr::is_loopback) {
        Ok(())
    } else {
        Err(format!("{addr}: not a declared host (loopback, or pass --allow-host {host})"))
    }
}

pub fn http_policy(root: &Path, state_files: &[PathBuf], declared_ports: &[u16]) -> JailPolicy {
    let mut p = JailPolicy::for_program(root, &[]);
    for f in state_files {
        if let Some(d) = f.parent().filter(|d| !d.as_os_str().is_empty())
            && !p.rw.iter().any(|x| d.starts_with(x))
        {
            p.rw.push(d.to_path_buf());
        }
    }
    for d in ["/usr/bin", "/bin", "/etc/gitconfig", "/proc", "/sys/class/powercap", "/sys/devices", "/dev/urandom"] {
        let d = PathBuf::from(d);
        if d.exists() && !p.ro.contains(&d) {
            p.ro.push(d);
        }
    }
    let mut ports = declared_ports.to_vec();
    ports.sort_unstable();
    ports.dedup();
    p.tcp_connect = ports;
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bind_hosts() {
        assert!(check_bind("127.0.0.1:0", &[]).is_ok());
        assert!(check_bind("[::1]:8080", &[]).is_ok());
        assert!(check_bind("0.0.0.0:8080", &[]).is_err());
        assert!(check_bind("0.0.0.0:8080", &["0.0.0.0".into()]).is_ok());
    }
    #[test]
    fn policy_shape() {
        let d = tempfile::tempdir().unwrap();
        let other = tempfile::tempdir().unwrap();
        let p = http_policy(d.path(), &[d.path().join("x.jsonl"), other.path().join("k")], &[443, 443, 80]);
        assert!(!p.allow_net);
        assert_eq!(p.tcp_connect, vec![80, 443]);
        assert!(p.rw.contains(&other.path().to_path_buf()));
        assert_eq!(p.rw.iter().filter(|x| *x == d.path()).count(), 1);
    }
}
