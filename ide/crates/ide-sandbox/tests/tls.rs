//! Encrypted traffic through the network proxy, against a local TLS mock
//! server (rustls + graviola, test CA in tests/fixtures; the CA's private
//! key was discarded after signing the leaf).
use ide_sandbox::net::{
    Approvals, Credentials, Decision, Fixed, NetError, NetManifest, NetProxy, Proto, listen,
};
use ide_sandbox::tls::client_hello_sni;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

fn ca() -> CertificateDer<'static> {
    CertificateDer::from_pem_file(format!("{FIX}/ca.pem")).unwrap()
}

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls_graviola::default_provider())
}

/// A TLS server for `localhost` that answers every request with the
/// request head it received (so tests can see injected headers) and the
/// SNI it was asked for.
fn mock_tls() -> u16 {
    let cert = CertificateDer::from_pem_file(format!("{FIX}/leaf.pem")).unwrap();
    let key = PrivateKeyDer::from_pem_file(format!("{FIX}/leaf.key")).unwrap();
    let cfg = Arc::new(
        rustls::ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .unwrap(),
    );
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            let cfg = cfg.clone();
            std::thread::spawn(move || {
                let conn = rustls::ServerConnection::new(cfg).unwrap();
                let mut t = rustls::StreamOwned::new(conn, s);
                let mut head = vec![];
                let mut r = BufReader::new(&mut t);
                while !head.ends_with(b"\r\n\r\n") {
                    if r.read_until(b'\n', &mut head).unwrap_or(0) == 0 {
                        return;
                    }
                }
                let sni = t.conn.server_name().unwrap_or("").to_string();
                let body = format!("sni={sni}\n{}", String::from_utf8_lossy(&head));
                let _ = write!(
                    t,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                t.conn.send_close_notify();
                let _ = t.flush();
            });
        }
    });
    port
}

fn proxy(port: u16, allow: bool, cred: bool) -> NetProxy {
    let m = NetManifest::parse(&format!(
        r#"{{"net":[{{"host":"localhost","port":{port},"proto":"https","reason":"test api"}}]}}"#
    ))
    .unwrap();
    let key = format!("https://localhost:{port}");
    let mut c = Credentials::default();
    if cred {
        c.set(&key, "Authorization", "Bearer supervisor-secret");
    }
    NetProxy::new(
        m,
        Approvals::default(),
        Box::new(Fixed(BTreeMap::from([(key, allow)]))),
        c,
    )
    .with_roots(vec![ca()])
}

const REQ: &[u8] = b"GET /v1/models HTTP/1.1\r\nHost: wrong\r\nAuthorization: Bearer guest-stolen\r\nX-Test: 1\r\n\r\n";

#[test]
fn https_endpoint_injects_the_credential_over_verified_tls() {
    let port = mock_tls();
    let mut p = proxy(port, true, true);
    let r = p.request(Proto::Https, "localhost", port, REQ).unwrap();
    let t = String::from_utf8_lossy(&r);
    assert!(t.starts_with("HTTP/1.1 200 OK"), "{t}");
    assert!(t.contains("sni=localhost"), "{t}");
    assert!(t.contains("Authorization: Bearer supervisor-secret"), "{t}");
    assert!(!t.contains("guest-stolen"), "{t}");
    assert!(t.contains(&format!("Host: localhost:{port}")), "{t}");
    let ev = &p.events[0];
    assert_eq!(ev.decision, Decision::Allowed);
    assert!(ev.credential_injected);
    assert!(ev.bytes_out > 0 && ev.bytes_in as usize == r.len());
}

#[test]
fn https_deny_undeclared_and_unverifiable() {
    let port = mock_tls();
    // Declared but denied.
    let mut p = proxy(port, false, true);
    assert!(matches!(
        p.request(Proto::Https, "localhost", port, REQ),
        Err(NetError::Denied(_))
    ));
    // Same port, a host that is not declared (127.0.0.1).
    let mut p = proxy(port, true, true);
    assert!(matches!(
        p.request(Proto::Https, "127.0.0.1", port, REQ),
        Err(NetError::Undeclared(_))
    ));
    // Declared as https, asked as plain http: undeclared.
    assert!(matches!(
        p.request(Proto::Http, "localhost", port, REQ),
        Err(NetError::Undeclared(_))
    ));
    // Without the test CA the certificate does not verify: no bytes leak.
    let mut p = proxy(port, true, true).with_roots(vec![]);
    match p.request(Proto::Https, "localhost", port, REQ) {
        Err(NetError::Io(e)) => assert!(e.contains("certificate") || e.contains("Unknown"), "{e}"),
        r => panic!("{r:?}"),
    }
}

/// A client doing its own TLS through `CONNECT`, naming `sni`.
fn connect_tls(proxy: &str, token: &str, target: &str, sni: &str) -> Result<String, String> {
    let mut s = TcpStream::connect(proxy).unwrap();
    write!(
        s,
        "CONNECT {target} HTTP/1.1\r\nHost: {target}\r\nProxy-Authorization: Bearer {token}\r\n\r\n"
    )
    .unwrap();
    let mut r = BufReader::new(s.try_clone().unwrap());
    let mut status = String::new();
    r.read_line(&mut status).unwrap();
    if !status.contains(" 200 ") {
        let mut rest = String::new();
        let _ = r.read_to_string(&mut rest);
        return Err(format!("{status}{rest}"));
    }
    let mut l = String::new();
    while l != "\r\n" {
        l.clear();
        r.read_line(&mut l).unwrap();
    }
    let mut roots = rustls::RootCertStore::empty();
    roots.add(ca()).unwrap();
    let cfg = rustls::ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let conn = rustls::ClientConnection::new(
        Arc::new(cfg),
        ServerName::try_from(sni.to_string()).unwrap(),
    )
    .unwrap();
    let mut t = rustls::StreamOwned::new(conn, s);
    t.write_all(b"GET /x HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .map_err(|e| e.to_string())?;
    let mut out = vec![];
    match t.read_to_end(&mut out) {
        Ok(_) => {}
        Err(e) if !out.is_empty() && e.kind() == std::io::ErrorKind::UnexpectedEof => {}
        Err(e) => return Err(e.to_string()),
    }
    Ok(String::from_utf8_lossy(&out).to_string())
}

#[test]
fn connect_tunnels_only_to_declared_hosts_with_matching_sni() {
    let port = mock_tls();
    let p = Arc::new(Mutex::new(proxy(port, true, false)));
    let l = listen(p.clone(), "127.0.0.1:0").unwrap();
    let addr = l.addr.to_string();
    // Allowed: declared, approved, SNI = host. The client's own TLS works
    // end to end (the proxy cannot read it).
    let t = connect_tls(&addr, &l.token, &format!("localhost:{port}"), "localhost").unwrap();
    assert!(t.contains("sni=localhost"), "{t}");
    // Fronting: CONNECT to the declared host but TLS names another one.
    let e = connect_tls(
        &addr,
        &l.token,
        &format!("localhost:{port}"),
        "evil.example",
    )
    .unwrap_err();
    assert!(!e.contains("200"), "{e}");
    // Undeclared target.
    let e = connect_tls(&addr, &l.token, "example.org:443", "example.org").unwrap_err();
    assert!(e.starts_with("HTTP/1.1 403"), "{e}");
    // No proxy token.
    let e = connect_tls(&addr, "nope", &format!("localhost:{port}"), "localhost").unwrap_err();
    assert!(e.starts_with("HTTP/1.1 407"), "{e}");
    std::thread::sleep(std::time::Duration::from_millis(100));
    let g = p.lock().unwrap();
    let evs: Vec<_> = g.events.iter().filter(|e| e.tunnel).collect();
    assert_eq!(evs.len(), 3, "{evs:?}");
    assert!(evs[0].error.is_none() && evs[0].sni.as_deref() == Some("localhost"));
    assert!(evs[0].bytes_in > 0 && evs[0].bytes_out > 0);
    assert!(evs[1].error.as_deref().unwrap().contains("evil.example"));
    assert_eq!(evs[2].decision, Decision::Undeclared);
}

#[test]
fn credentialed_endpoints_refuse_tunnels_and_inject_on_absolute_https() {
    let port = mock_tls();
    let p = Arc::new(Mutex::new(proxy(port, true, true)));
    let l = listen(p.clone(), "127.0.0.1:0").unwrap();
    let addr = l.addr.to_string();
    let e = connect_tls(&addr, &l.token, &format!("localhost:{port}"), "localhost").unwrap_err();
    assert!(
        e.starts_with("HTTP/1.1 403") && e.contains("credential"),
        "{e}"
    );
    // Absolute-form https:// request in clear to the loopback proxy, Basic
    // proxy auth (what HTTPS_PROXY=http://rce:TOKEN@... clients send).
    let mut s = TcpStream::connect(&addr).unwrap();
    write!(
        s,
        "GET https://localhost:{port}/v1/models HTTP/1.1\r\nHost: localhost\r\nProxy-Authorization: {}\r\nAuthorization: Bearer guest-stolen\r\n\r\n",
        ide_sandbox::proxy::basic_auth(&l.token)
    )
    .unwrap();
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    assert!(out.starts_with("HTTP/1.1 200"), "{out}");
    assert!(
        out.contains("Authorization: Bearer supervisor-secret"),
        "{out}"
    );
    assert!(
        !out.contains("guest-stolen") && !out.contains(&l.token),
        "{out}"
    );
    assert!(l.proxy_url().starts_with("http://rce:"));
}

#[test]
fn sni_parser_reads_real_client_hellos() {
    let cfg = rustls::ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(rustls::RootCertStore::empty())
        .with_no_client_auth();
    let mut c = rustls::ClientConnection::new(
        Arc::new(cfg),
        ServerName::try_from("api.example.com").unwrap(),
    )
    .unwrap();
    let mut rec = vec![];
    c.write_tls(&mut rec).unwrap();
    assert_eq!(
        client_hello_sni(&rec).unwrap().as_deref(),
        Some("api.example.com")
    );
    assert!(client_hello_sni(b"GET / HTTP/1.1\r\n\r\n").is_err());
}

/// End to end with a real client: curl in the Landlock jail, allowed to
/// connect only to the proxy port, reaches the declared TLS host through
/// CONNECT and cannot reach it directly. Skipped (and says so) without curl.
#[cfg(target_os = "linux")]
#[test]
fn jailed_curl_reaches_declared_https_only_through_the_proxy() {
    use ide_sandbox::jail::{JailPolicy, platform_jail};
    let Some(curl) = ["/usr/bin/curl", "/bin/curl"]
        .into_iter()
        .map(std::path::PathBuf::from)
        .find(|p| p.exists())
    else {
        eprintln!("SKIP: no curl on this host");
        return;
    };
    let port = mock_tls();
    let p = Arc::new(Mutex::new(proxy(port, true, false)));
    let l = listen(p.clone(), "127.0.0.1:0").unwrap();
    let ws = tempfile::tempdir().unwrap();
    let mut pol = JailPolicy::for_program(ws.path(), &[curl.clone(), FIX.into()]);
    pol.tcp_connect = vec![l.addr.port()];
    let run = |args: Vec<String>| {
        let mut c = std::process::Command::new(&curl);
        c.args([
            "-sS",
            "--max-time",
            "10",
            "--cacert",
            &format!("{FIX}/ca.pem"),
        ])
        .args(args)
        .env_clear();
        platform_jail().run(c, &pol).unwrap()
    };
    let url = format!("https://localhost:{port}/via-proxy");
    let o = run(vec!["--proxy".into(), l.proxy_url(), url.clone()]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(
        o.status.success(),
        "{} {}",
        out,
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        out.contains("sni=localhost") && out.contains("GET /via-proxy"),
        "{out}"
    );
    // Direct: the jail refuses the connect (port not granted).
    let o = run(vec![
        "--resolve".into(),
        format!("localhost:{port}:127.0.0.1"),
        url,
    ]);
    assert!(!o.status.success());
    let ev = p.lock().unwrap();
    assert!(
        ev.events
            .iter()
            .any(|e| e.tunnel && e.error.is_none() && e.bytes_in > 0)
    );
}

/// A plain HTTP server that waits `delay` before answering.
fn slow_http(delay: std::time::Duration) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for s in l.incoming().flatten() {
            std::thread::spawn(move || {
                let mut s = s;
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut line = String::new();
                while r.read_line(&mut line).unwrap_or(0) > 2 {
                    line.clear();
                }
                std::thread::sleep(delay);
                let _ = s.write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
                );
            });
        }
    });
    port
}

#[test]
fn a_slow_fetch_holds_no_lock() {
    let slow = slow_http(std::time::Duration::from_millis(1500));
    let fast = slow_http(std::time::Duration::ZERO);
    let m = NetManifest::parse(&format!(
        r#"{{"net":[{{"host":"127.0.0.1","port":{slow},"proto":"http","reason":"s"}},{{"host":"127.0.0.1","port":{fast},"proto":"http","reason":"f"}}]}}"#
    ))
    .unwrap();
    let p = Arc::new(Mutex::new(NetProxy::new(
        m,
        Approvals::default(),
        Box::new(Fixed(BTreeMap::from([
            (format!("http://127.0.0.1:{slow}"), true),
            (format!("http://127.0.0.1:{fast}"), true),
        ]))),
        Credentials::default(),
    )));
    let l = listen(p.clone(), "127.0.0.1:0").unwrap();
    let get = |port: u16, addr: String, tok: String| {
        let mut s = TcpStream::connect(addr).unwrap();
        write!(
            s,
            "GET http://127.0.0.1:{port}/ HTTP/1.1\r\nProxy-Authorization: {}\r\n\r\n",
            ide_sandbox::proxy::basic_auth(&tok)
        )
        .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        out
    };
    let (a, t) = (l.addr.to_string(), l.token.clone());
    let slow_t = std::thread::spawn(move || get(slow, a, t));
    std::thread::sleep(std::time::Duration::from_millis(200));
    let t0 = std::time::Instant::now();
    // The supervisor can read proxy state, and another guest request
    // completes, while the slow one is in flight.
    drop(p.lock().unwrap());
    let r = get(fast, l.addr.to_string(), l.token.clone());
    assert!(r.ends_with("ok"), "{r}");
    assert!(
        t0.elapsed() < std::time::Duration::from_millis(800),
        "{:?}",
        t0.elapsed()
    );
    assert!(!slow_t.is_finished());
    assert!(slow_t.join().unwrap().ends_with("ok"));
    assert_eq!(p.lock().unwrap().events.len(), 2);
}
