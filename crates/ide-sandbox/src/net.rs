//! Declared network capabilities, granted through a host-side proxy.
//!
//! A program or project manifest declares every endpoint it needs
//! ([`NetManifest`]: host, port, protocol). The guest never gets a socket:
//! it calls the `rce_net.request` host import (see [`crate::wasi`]), and the
//! [`NetProxy`] decides. A request goes out only if its endpoint is
//! declared *and* approved; the first use of a declared endpoint asks the
//! [`Approver`] (the IDE shows that as an approval card) and the answer is
//! remembered in [`Approvals`]. Anything undeclared is denied without asking.
//!
//! Credentials live with the supervisor ([`Credentials`]) and are added by
//! the proxy on the way out (an HTTP header); the guest cannot read them,
//! and any credential header the guest wrote itself is stripped.
//! Bytes in and out are counted per endpoint and become the simulator's
//! `net_bytes` feature and the receipt's [`NetEvent`] list.
//!
//! Encrypted traffic ([`crate::tls`], rustls on the pure-Rust graviola
//! provider):
//! - `https` endpoints: the guest hands the proxy a plain HTTP request; the
//!   proxy injects the credential and originates TLS itself, verifying the
//!   server certificate for the declared host. The guest never holds the
//!   secret and never sees another host.
//! - [`listen`]: an HTTP proxy for native processes (jailed agents whose
//!   Landlock rules allow only the proxy port). `CONNECT host:port` opens
//!   an opaque tunnel only to a declared, approved `https`/`tcp` endpoint,
//!   and the tunnel's TLS ClientHello must name that same host (SNI), so
//!   a tunnel cannot be fronted to another site. A tunnel cannot carry an
//!   injected credential, so endpoints with a credential refuse CONNECT;
//!   clients use absolute-form `https://` requests to the proxy instead,
//!   which take the injecting path above. The proxy itself requires a
//!   token (`Proxy-Authorization`).
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

pub use crate::proxy::{ProxyListener, listen};

/// Largest response the proxy returns to a guest, bytes.
pub const RESPONSE_CAP: usize = 4 << 20;
/// Connect and I/O timeout.
pub const TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Proto {
    /// Plain HTTP/1.1 through the proxy (credential injection supported).
    Http,
    /// Raw TCP: request bytes written, response read to EOF.
    Tcp,
    /// HTTP/1.1 over TLS originated by the proxy (credential injection
    /// supported; certificate verified for the declared host). Also the
    /// endpoint kind a `CONNECT` tunnel is checked against.
    Https,
}

impl Proto {
    pub fn code(self) -> i32 {
        match self {
            Proto::Http => 0,
            Proto::Tcp => 1,
            Proto::Https => 2,
        }
    }
    pub fn from_code(c: i32) -> Option<Self> {
        match c {
            0 => Some(Proto::Http),
            1 => Some(Proto::Tcp),
            2 => Some(Proto::Https),
            _ => None,
        }
    }
}

/// One declared endpoint.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NetCap {
    pub host: String,
    pub port: u16,
    pub proto: Proto,
    /// Why the program needs it (shown on the approval card).
    #[serde(default)]
    pub reason: String,
}

impl NetCap {
    /// Stable key for approvals and credentials: `http://host:port`.
    pub fn key(&self) -> String {
        endpoint_key(self.proto, &self.host, self.port)
    }
}

pub fn endpoint_key(p: Proto, host: &str, port: u16) -> String {
    let s = match p {
        Proto::Http => "http",
        Proto::Tcp => "tcp",
        Proto::Https => "https",
    };
    format!("{s}://{}:{port}", host.to_ascii_lowercase())
}

/// The `net` section of a project manifest (`rce.json` → `{"net": [...]}`)
/// or a program's own declaration.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NetManifest {
    #[serde(default)]
    pub net: Vec<NetCap>,
}

impl NetManifest {
    pub fn parse(json: &str) -> Result<Self, NetError> {
        let m: NetManifest =
            serde_json::from_str(json).map_err(|e| NetError::Manifest(e.to_string()))?;
        for c in &m.net {
            if c.host.is_empty() || c.port == 0 || c.host.contains(['/', ' ', '@']) {
                return Err(NetError::Manifest(format!("bad endpoint {:?}", c.key())));
            }
        }
        Ok(m)
    }
    /// Read `<workspace>/rce.json`; a missing file declares nothing.
    pub fn load(ws: &Path) -> Result<Self, NetError> {
        match std::fs::read_to_string(ws.join("rce.json")) {
            Ok(s) => Self::parse(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(NetError::Manifest(e.to_string())),
        }
    }
    pub fn find(&self, p: Proto, host: &str, port: u16) -> Option<&NetCap> {
        let k = endpoint_key(p, host, port);
        self.net.iter().find(|c| c.key() == k)
    }
    /// The TCP ports the native jail must allow connecting to.
    pub fn ports(&self) -> Vec<u16> {
        let s: BTreeSet<u16> = self.net.iter().map(|c| c.port).collect();
        s.into_iter().collect()
    }
}

/// Asked the first time a declared endpoint is used.
pub trait Approver: Send {
    fn approve(&mut self, cap: &NetCap) -> bool;
}

/// Approves nothing (headless default).
pub struct DenyAll;
impl Approver for DenyAll {
    fn approve(&mut self, _: &NetCap) -> bool {
        false
    }
}

/// A fixed answer per key (tests, or a pre-approved policy).
pub struct Fixed(pub BTreeMap<String, bool>);
impl Approver for Fixed {
    fn approve(&mut self, cap: &NetCap) -> bool {
        self.0.get(&cap.key()).copied().unwrap_or(false)
    }
}

/// Remembered decisions, persisted as JSON (`.rce-ide/net-approvals.json`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Approvals {
    pub decided: BTreeMap<String, bool>,
}

impl Approvals {
    pub fn load(path: &Path) -> Self {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(d) = path.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(
            path,
            serde_json::to_vec_pretty(self).map_err(std::io::Error::other)?,
        )
    }
}

/// Supervisor-held secrets: endpoint key → (header name, value).
#[derive(Default)]
pub struct Credentials(BTreeMap<String, (String, String)>);

impl Credentials {
    pub fn set(&mut self, key: &str, header: &str, value: &str) {
        self.0.insert(key.into(), (header.into(), value.into()));
    }
    fn get(&self, key: &str) -> Option<&(String, String)> {
        self.0.get(key)
    }
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.0.keys()).finish() // never the values
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Allowed,
    /// Declared, but the user said no.
    Denied,
    /// Not in the manifest: never asked, never allowed.
    Undeclared,
}

/// One proxied (or refused) request, for the receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NetEvent {
    pub endpoint: String,
    pub decision: Decision,
    pub bytes_out: u64,
    pub bytes_in: u64,
    pub credential_injected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// A CONNECT tunnel (opaque bytes) rather than a proxied request.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub tunnel: bool,
    /// The server name the tunnel's TLS ClientHello asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sni: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum NetError {
    #[error("manifest: {0}")]
    Manifest(String),
    #[error("undeclared endpoint {0}")]
    Undeclared(String),
    #[error("endpoint {0} not approved")]
    Denied(String),
    #[error("io: {0}")]
    Io(String),
    #[error("bad request: {0}")]
    BadRequest(String),
}

impl NetError {
    /// Negative status returned to the guest.
    pub fn code(&self) -> i64 {
        match self {
            NetError::Undeclared(_) => -1,
            NetError::Denied(_) => -2,
            NetError::Io(_) => -3,
            NetError::BadRequest(_) => -4,
            NetError::Manifest(_) => -5,
        }
    }
}

/// The host-side proxy. Pure Rust (std sockets), synchronous.
pub struct NetProxy {
    pub manifest: NetManifest,
    pub approvals: Approvals,
    approver: Box<dyn Approver>,
    creds: Credentials,
    pub events: Vec<NetEvent>,
    tls: Option<Arc<rustls::ClientConfig>>,
    extra_roots: Vec<rustls::pki_types::CertificateDer<'static>>,
}

impl NetProxy {
    pub fn new(
        manifest: NetManifest,
        approvals: Approvals,
        approver: Box<dyn Approver>,
        creds: Credentials,
    ) -> Self {
        NetProxy {
            manifest,
            approvals,
            approver,
            creds,
            events: vec![],
            tls: None,
            extra_roots: vec![],
        }
    }

    /// Also trust these roots for `https` endpoints (a company CA, a test CA).
    pub fn with_roots(mut self, roots: Vec<rustls::pki_types::CertificateDer<'static>>) -> Self {
        self.extra_roots = roots;
        self.tls = None;
        self
    }

    fn tls(&mut self) -> Result<Arc<rustls::ClientConfig>, NetError> {
        if self.tls.is_none() {
            self.tls = Some(crate::tls::client_config(&self.extra_roots).map_err(NetError::Io)?);
        }
        Ok(self.tls.clone().expect("set"))
    }

    pub(crate) fn event(key: &str) -> NetEvent {
        NetEvent {
            endpoint: key.into(),
            decision: Decision::Allowed,
            bytes_out: 0,
            bytes_in: 0,
            credential_injected: false,
            error: None,
            tunnel: false,
            sni: None,
        }
    }

    pub(crate) fn has_credential(&self, key: &str) -> bool {
        self.creds.get(key).is_some()
    }

    /// No declared endpoints, nothing allowed.
    pub fn closed() -> Self {
        Self::new(
            NetManifest::default(),
            Approvals::default(),
            Box::new(DenyAll),
            Credentials::default(),
        )
    }

    pub fn bytes(&self) -> u64 {
        self.events.iter().map(|e| e.bytes_in + e.bytes_out).sum()
    }

    /// Endpoints declared but not yet decided: what the IDE should ask about.
    pub fn pending(&self) -> Vec<&NetCap> {
        self.manifest
            .net
            .iter()
            .filter(|c| !self.approvals.decided.contains_key(&c.key()))
            .collect()
    }

    pub(crate) fn authorize(
        &mut self,
        p: Proto,
        host: &str,
        port: u16,
    ) -> Result<NetCap, NetError> {
        let key = endpoint_key(p, host, port);
        let cap = self
            .manifest
            .find(p, host, port)
            .cloned()
            .ok_or(NetError::Undeclared(key.clone()))?;
        let ok = match self.approvals.decided.get(&key) {
            Some(d) => *d,
            None => {
                let d = self.approver.approve(&cap);
                self.approvals.decided.insert(key.clone(), d);
                d
            }
        };
        if ok {
            Ok(cap)
        } else {
            Err(NetError::Denied(key))
        }
    }

    /// Send `req` to the endpoint and return the raw response bytes.
    pub fn request(
        &mut self,
        p: Proto,
        host: &str,
        port: u16,
        req: &[u8],
    ) -> Result<Vec<u8>, NetError> {
        let (ev, r) = match self.prepare(p, host, port, req) {
            Ok(prep) => prep.send(),
            Err(x) => x,
        };
        self.events.push(ev);
        r
    }

    /// The decision half of [`NetProxy::request`]: authorize, inject the
    /// credential, rewrite. Cheap, so callers sharing the proxy behind a
    /// lock hold it only for this, then [`Prepared::send`] without it and
    /// [`NetProxy::record`] the event.
    #[allow(clippy::type_complexity, clippy::result_large_err)]
    pub fn prepare(
        &mut self,
        p: Proto,
        host: &str,
        port: u16,
        req: &[u8],
    ) -> Result<Prepared, (NetEvent, Result<Vec<u8>, NetError>)> {
        let key = endpoint_key(p, host, port);
        let mut ev = Self::event(&key);
        let r = self.authorize(p, host, port).and_then(|_| {
            let out = match p {
                Proto::Tcp => req.to_vec(),
                Proto::Http | Proto::Https => {
                    let cred = self.creds.get(&key);
                    ev.credential_injected = cred.is_some();
                    rewrite_http(req, host, port, cred)?
                }
            };
            let tls = if p == Proto::Https {
                Some(self.tls()?)
            } else {
                None
            };
            Ok((out, tls))
        });
        match r {
            Ok((out, tls)) => {
                ev.bytes_out = out.len() as u64;
                Ok(Prepared {
                    proto: p,
                    host: host.into(),
                    port,
                    out,
                    tls,
                    ev,
                })
            }
            Err(e) => Err((failed(ev, &e), Err(e))),
        }
    }

    /// Log an event from a [`Prepared::send`].
    pub fn record(&mut self, ev: NetEvent) {
        self.events.push(ev);
    }
}

fn failed(mut ev: NetEvent, e: &NetError) -> NetEvent {
    ev.decision = match e {
        NetError::Undeclared(_) => Decision::Undeclared,
        NetError::Denied(_) => Decision::Denied,
        _ => Decision::Allowed,
    };
    ev.error = Some(e.to_string());
    ev
}

/// An authorized, rewritten request ready to go out (no proxy state needed).
pub struct Prepared {
    proto: Proto,
    host: String,
    port: u16,
    out: Vec<u8>,
    tls: Option<Arc<rustls::ClientConfig>>,
    ev: NetEvent,
}

impl Prepared {
    /// Do the network I/O. Returns the event to record and the response.
    pub fn send(self) -> (NetEvent, Result<Vec<u8>, NetError>) {
        let Prepared {
            proto,
            host,
            port,
            out,
            tls,
            mut ev,
        } = self;
        let r = match (proto, tls) {
            (Proto::Https, Some(cfg)) => connect(&host, port).and_then(|sock| {
                crate::tls::exchange(&cfg, sock, &host, &out, RESPONSE_CAP).map_err(NetError::Io)
            }),
            _ => exchange(&host, port, &out),
        };
        match r {
            Ok(resp) => {
                ev.bytes_in = resp.len() as u64;
                (ev, Ok(resp))
            }
            Err(e) => (failed(ev, &e), Err(e)),
        }
    }
}

pub(crate) fn connect(host: &str, port: u16) -> Result<TcpStream, NetError> {
    let io = |e: std::io::Error| NetError::Io(e.to_string());
    // Every resolved address in order (`localhost` is often ::1 first).
    let mut last = NetError::Io(format!("{host}: no address"));
    let mut sock = None;
    for addr in (host, port).to_socket_addrs().map_err(io)? {
        match TcpStream::connect_timeout(&addr, TIMEOUT) {
            Ok(s) => {
                sock = Some(s);
                break;
            }
            Err(e) => last = io(e),
        }
    }
    let s = sock.ok_or(last)?;
    s.set_read_timeout(Some(TIMEOUT)).map_err(io)?;
    s.set_write_timeout(Some(TIMEOUT)).map_err(io)?;
    Ok(s)
}

fn exchange(host: &str, port: u16, out: &[u8]) -> Result<Vec<u8>, NetError> {
    let io = |e: std::io::Error| NetError::Io(e.to_string());
    let mut s = connect(host, port)?;
    s.write_all(out).map_err(io)?;
    s.shutdown(std::net::Shutdown::Write).map_err(io)?;
    let mut buf = Vec::new();
    s.take(RESPONSE_CAP as u64)
        .read_to_end(&mut buf)
        .map_err(io)?;
    Ok(buf)
}

/// Rebuild an HTTP/1.x request: origin-form target only, `Host` forced to
/// the declared endpoint, `Connection: close`, guest-written credential
/// headers removed, and the supervisor's credential added.
pub fn rewrite_http(
    req: &[u8],
    host: &str,
    port: u16,
    cred: Option<&(String, String)>,
) -> Result<Vec<u8>, NetError> {
    let bad = |m: &str| NetError::BadRequest(m.into());
    let split = req
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| bad("no header end"))?;
    let head = std::str::from_utf8(&req[..split]).map_err(|_| bad("header not utf-8"))?;
    let body = &req[split + 4..];
    let mut lines = head.split("\r\n");
    let start = lines.next().ok_or_else(|| bad("empty"))?;
    let mut parts = start.split(' ');
    let (m, t, v) = (parts.next(), parts.next(), parts.next());
    let (Some(m), Some(t), Some(v), None) = (m, t, v, parts.next()) else {
        return Err(bad("request line"));
    };
    if !t.starts_with('/') || !v.starts_with("HTTP/1.") || m.is_empty() {
        return Err(bad("only origin-form HTTP/1.x requests"));
    }
    let drop = [
        "host",
        "connection",
        "authorization",
        "proxy-authorization",
        "cookie",
    ];
    let cred_name = cred.map(|c| c.0.to_ascii_lowercase());
    let mut out = format!("{m} {t} {v}\r\nHost: {host}:{port}\r\nConnection: close\r\n");
    for l in lines {
        let name = l
            .split(':')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if drop.contains(&name.as_str()) || Some(&name) == cred_name.as_ref() {
            continue;
        }
        out.push_str(l);
        out.push_str("\r\n");
    }
    if let Some((h, val)) = cred {
        out.push_str(&format!("{h}: {val}\r\n"));
    }
    out.push_str("\r\n");
    let mut v = out.into_bytes();
    v.extend_from_slice(body);
    Ok(v)
}
