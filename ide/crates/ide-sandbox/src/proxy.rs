//! The network proxy as a local HTTP proxy server, for native processes
//! (jailed agents): `CONNECT` tunnels with host + SNI checks, and
//! absolute-form `http://` / `https://` requests that take the
//! credential-injecting path. See [`crate::net`].
use crate::net::{Decision, NetError, NetEvent, NetProxy, Proto, endpoint_key};
use crate::tls::client_hello_sni;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

const MAX_HEAD: usize = 64 * 1024;
const IDLE: Duration = Duration::from_secs(60);

pub struct ProxyListener {
    pub addr: SocketAddr,
    /// The value clients send as `Proxy-Authorization: Bearer <token>` (or
    /// Basic with any user and the token as password).
    pub token: String,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl ProxyListener {
    /// `http://rce:TOKEN@127.0.0.1:PORT`, for `HTTPS_PROXY`/`HTTP_PROXY`.
    pub fn proxy_url(&self) -> String {
        format!("http://rce:{}@{}", self.token, self.addr)
    }
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

impl Drop for ProxyListener {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `Proxy-Authorization` value for clients that only speak Basic.
pub fn basic_auth(token: &str) -> String {
    format!("Basic {}", b64(format!("rce:{token}").as_bytes()))
}

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(T[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        } as u32)
    };
    let s = s.trim_end_matches('=').as_bytes();
    let mut out = vec![];
    for c in s.chunks(4) {
        let mut n = 0u32;
        for (i, b) in c.iter().enumerate() {
            n |= val(*b)? << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if c.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if c.len() > 3 {
            out.push(n as u8);
        }
    }
    Some(out)
}

fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

fn authorized(head: &str, token: &str) -> bool {
    head.lines()
        .filter_map(|l| l.split_once(':'))
        .filter(|(k, _)| k.trim().eq_ignore_ascii_case("proxy-authorization"))
        .any(|(_, v)| {
            let v = v.trim();
            if let Some(t) = v.strip_prefix("Bearer ") {
                return same(t.as_bytes(), token.as_bytes());
            }
            if let Some(b) = v.strip_prefix("Basic ")
                && let Some(d) = b64_decode(b)
                && let Some(p) = d.iter().position(|c| *c == b':')
            {
                return same(&d[p + 1..], token.as_bytes());
            }
            false
        })
}

/// Start the proxy on a loopback address (`127.0.0.1:0` picks a port).
pub fn listen(proxy: Arc<Mutex<NetProxy>>, addr: &str) -> std::io::Result<ProxyListener> {
    let l = TcpListener::bind(addr)?;
    let addr = l.local_addr()?;
    if !addr.ip().is_loopback() {
        return Err(std::io::Error::other("the proxy binds to loopback only"));
    }
    let mut t = [0u8; 24];
    getrandom::fill(&mut t).map_err(std::io::Error::other)?;
    let token = hex::encode(t);
    l.set_nonblocking(true)?;
    let stop = Arc::new(AtomicBool::new(false));
    let (st, tk) = (stop.clone(), token.clone());
    let handle = std::thread::spawn(move || {
        while !st.load(Ordering::SeqCst) {
            match l.accept() {
                Ok((s, _)) => {
                    let (p, tk) = (proxy.clone(), tk.clone());
                    std::thread::spawn(move || {
                        let _ = s.set_nonblocking(false);
                        let _ = connection(&p, s, &tk);
                    });
                }
                Err(_) => std::thread::sleep(Duration::from_millis(5)),
            }
        }
    });
    Ok(ProxyListener {
        addr,
        token,
        stop,
        handle: Some(handle),
    })
}

fn reply(s: &mut TcpStream, code: u16, why: &str) -> std::io::Result<()> {
    let reason = match code {
        403 => "Forbidden",
        407 => "Proxy Authentication Required",
        400 => "Bad Request",
        _ => "Bad Gateway",
    };
    let extra = if code == 407 {
        "Proxy-Authenticate: Basic realm=\"rce\"\r\n"
    } else {
        ""
    };
    write!(
        s,
        "HTTP/1.1 {code} {reason}\r\n{extra}Content-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{why}",
        why.len()
    )
}

fn split_host_port(hp: &str, default: u16) -> Option<(String, u16)> {
    let (h, p) = match hp.rsplit_once(':') {
        Some((h, p)) if !h.contains(':') || h.starts_with('[') => (h, p.parse().ok()?),
        _ => (hp, default),
    };
    let h = h.trim_start_matches('[').trim_end_matches(']');
    (!h.is_empty()).then(|| (h.to_ascii_lowercase(), p))
}

fn connection(p: &Arc<Mutex<NetProxy>>, s: TcpStream, token: &str) -> std::io::Result<()> {
    s.set_read_timeout(Some(IDLE))?;
    let mut r = BufReader::new(s.try_clone()?);
    let mut head = vec![];
    while !head.ends_with(b"\r\n\r\n") {
        if r.read_until(b'\n', &mut head)? == 0 || head.len() > MAX_HEAD {
            return Ok(());
        }
    }
    let head = String::from_utf8_lossy(&head).to_string();
    let mut w = s;
    if !authorized(&head, token) {
        return reply(&mut w, 407, "proxy token required");
    }
    let first = head.lines().next().unwrap_or("");
    let mut parts = first.split(' ');
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    if method.eq_ignore_ascii_case("CONNECT") {
        let Some((host, port)) = split_host_port(target, 443) else {
            return reply(&mut w, 400, "CONNECT host:port");
        };
        return tunnel(p, r, w, &host, port);
    }
    // Absolute-form request: `GET https://host[:port]/path HTTP/1.1`.
    let (proto, rest, default) = if let Some(r) = target.strip_prefix("https://") {
        (Proto::Https, r, 443)
    } else if let Some(r) = target.strip_prefix("http://") {
        (Proto::Http, r, 80)
    } else {
        return reply(
            &mut w,
            400,
            "absolute-form http:// or https:// target, or CONNECT",
        );
    };
    let (hp, path) = rest
        .split_once('/')
        .map_or((rest, "/".to_string()), |(a, b)| (a, format!("/{b}")));
    let Some((host, port)) = split_host_port(hp, default) else {
        return reply(&mut w, 400, "bad host");
    };
    let len: usize = head
        .lines()
        .filter_map(|l| l.split_once(':'))
        .find(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.trim().parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len.min(crate::net::RESPONSE_CAP)];
    r.read_exact(&mut body)?;
    let origin = head.replacen(target, &path, 1);
    let mut req = origin.into_bytes();
    req.extend_from_slice(&body);
    // Decide under the lock; fetch without it (other guests' requests and
    // tunnels proceed meanwhile); record under it again.
    let prep = p
        .lock()
        .map_err(|_| std::io::Error::other("proxy poisoned"))?
        .prepare(proto, &host, port, &req);
    let (ev, res) = match prep {
        Ok(prep) => prep.send(),
        Err(x) => x,
    };
    record(p, ev);
    match res {
        Ok(resp) => w.write_all(&resp),
        Err(e @ (NetError::Undeclared(_) | NetError::Denied(_))) => {
            reply(&mut w, 403, &e.to_string())
        }
        Err(e) => reply(&mut w, 502, &e.to_string()),
    }
}

fn record(p: &Arc<Mutex<NetProxy>>, ev: NetEvent) {
    if let Ok(mut g) = p.lock() {
        g.events.push(ev);
    }
}

fn tunnel(
    p: &Arc<Mutex<NetProxy>>,
    mut r: BufReader<TcpStream>,
    mut w: TcpStream,
    host: &str,
    port: u16,
) -> std::io::Result<()> {
    // Declared as https (TLS, SNI checked) or as raw tcp.
    let decided = {
        let mut g = p
            .lock()
            .map_err(|_| std::io::Error::other("proxy poisoned"))?;
        let https = endpoint_key(Proto::Https, host, port);
        if g.manifest.find(Proto::Https, host, port).is_some() {
            if g.has_credential(&https) {
                Err((
                    NetEvent {
                        decision: Decision::Allowed,
                        error: Some("endpoint has a supervisor credential: send absolute-form https:// requests to the proxy so it can add it".into()),
                        tunnel: true,
                        ..NetProxy::event(&https)
                    },
                    403,
                ))
            } else {
                g.authorize(Proto::Https, host, port)
                    .map(|_| Proto::Https)
                    .map_err(|e| (NetProxy::event(&https), e))
                    .map_err(|(ev, e)| (fail(ev, &e), 403))
            }
        } else {
            g.authorize(Proto::Tcp, host, port)
                .map(|_| Proto::Tcp)
                .map_err(|e| {
                    (
                        fail(NetProxy::event(&endpoint_key(Proto::Https, host, port)), &e),
                        403,
                    )
                })
        }
    };
    let proto = match decided {
        Ok(p) => p,
        Err((ev, code)) => {
            let why = ev.error.clone().unwrap_or_default();
            record(p, NetEvent { tunnel: true, ..ev });
            return reply(&mut w, code, &why);
        }
    };
    let key = endpoint_key(proto, host, port);
    let mut ev = NetEvent {
        tunnel: true,
        ..NetProxy::event(&key)
    };
    let up = match crate::net::connect(host, port) {
        Ok(u) => u,
        Err(e) => {
            ev.error = Some(e.to_string());
            record(p, ev);
            return reply(&mut w, 502, &e.to_string());
        }
    };
    w.write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")?;
    let mut up_w = up.try_clone()?;
    if proto == Proto::Https {
        // The first TLS record must be a ClientHello naming this host.
        let mut hdr = [0u8; 5];
        r.read_exact(&mut hdr)?;
        let n = u16::from_be_bytes([hdr[3], hdr[4]]) as usize;
        let mut rec = hdr.to_vec();
        rec.resize(5 + n, 0);
        r.read_exact(&mut rec[5..])?;
        let sni = client_hello_sni(&rec);
        let is_ip = host.parse::<IpAddr>().is_ok();
        let ok = match &sni {
            Ok(Some(n)) => n == host,
            Ok(None) => is_ip,
            Err(_) => false,
        };
        ev.sni = sni.clone().ok().flatten();
        if !ok {
            ev.error = Some(match sni {
                Ok(Some(n)) => format!("TLS server name {n} is not the declared host {host}"),
                Ok(None) => "TLS ClientHello without a server name".into(),
                Err(e) => e,
            });
            record(p, ev);
            return Ok(()); // close both sides
        }
        up_w.write_all(&rec)?;
        ev.bytes_out += rec.len() as u64;
    }
    // Leftover bytes the reader already buffered go first.
    let buffered = r.buffer().to_vec();
    r.consume(buffered.len());
    up_w.write_all(&buffered)?;
    ev.bytes_out += buffered.len() as u64;
    let mut from_client = r.into_inner();
    let mut up_r = up;
    up_r.set_read_timeout(Some(IDLE))?;
    let t = std::thread::spawn(move || {
        let n = std::io::copy(&mut from_client, &mut up_w).unwrap_or(0);
        let _ = up_w.shutdown(std::net::Shutdown::Write);
        n
    });
    let back = std::io::copy(&mut up_r, &mut w).unwrap_or(0);
    let _ = w.shutdown(std::net::Shutdown::Both);
    let out = t.join().unwrap_or(0);
    ev.bytes_out += out;
    ev.bytes_in += back;
    record(p, ev);
    Ok(())
}

fn fail(mut ev: NetEvent, e: &NetError) -> NetEvent {
    ev.decision = match e {
        NetError::Undeclared(_) => Decision::Undeclared,
        NetError::Denied(_) => Decision::Denied,
        _ => Decision::Allowed,
    };
    ev.error = Some(e.to_string());
    ev
}

#[cfg(test)]
mod t {
    #[test]
    fn base64_round_trip() {
        for s in ["", "a", "ab", "abc", "rce:0123456789abcdef"] {
            assert_eq!(
                super::b64_decode(&super::b64(s.as_bytes())).unwrap(),
                s.as_bytes()
            );
        }
        assert_eq!(super::b64(b"rce:x"), "cmNlOng=");
    }
}
