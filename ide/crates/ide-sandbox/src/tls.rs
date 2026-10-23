//! TLS for the network proxy: rustls with the graviola crypto provider
//! (pure Rust, no C), Mozilla roots from webpki-roots, plus any extra roots
//! the supervisor trusts. Also a ClientHello SNI reader for CONNECT tunnels.
use rustls::pki_types::{CertificateDer, ServerName};
use rustls::{ClientConfig, RootCertStore};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Arc;

/// Client config trusting the Mozilla roots and `extra`.
pub fn client_config(extra: &[CertificateDer<'static>]) -> Result<Arc<ClientConfig>, String> {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    for c in extra {
        roots.add(c.clone()).map_err(|e| format!("root: {e}"))?;
    }
    let cfg = ClientConfig::builder_with_provider(Arc::new(rustls_graviola::default_provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(cfg))
}

/// Send `out` over TLS to `host:port` (certificate verified for `host`)
/// and read the reply until the server closes, at most `cap` bytes.
pub fn exchange(
    cfg: &Arc<ClientConfig>,
    sock: TcpStream,
    host: &str,
    out: &[u8],
    cap: usize,
) -> Result<Vec<u8>, String> {
    let name = ServerName::try_from(host.to_string()).map_err(|e| format!("server name: {e}"))?;
    let conn = rustls::ClientConnection::new(cfg.clone(), name).map_err(|e| e.to_string())?;
    let mut tls = rustls::StreamOwned::new(conn, sock);
    tls.write_all(out).map_err(|e| format!("tls: {e}"))?;
    tls.flush().map_err(|e| format!("tls: {e}"))?;
    let mut buf = vec![];
    let mut chunk = [0u8; 16384];
    loop {
        match tls.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.len() >= cap {
                    buf.truncate(cap);
                    break;
                }
            }
            // A server that closes without close_notify after a complete
            // `Connection: close` response.
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && !buf.is_empty() => break,
            Err(e) => return Err(format!("tls: {e}")),
        }
    }
    Ok(buf)
}

/// The server name in a TLS ClientHello record (`None` if absent).
/// `Err` when the bytes are not a ClientHello.
pub fn client_hello_sni(rec: &[u8]) -> Result<Option<String>, String> {
    let e = |m: &str| Err(format!("not a TLS ClientHello: {m}"));
    if rec.len() < 5 || rec[0] != 0x16 {
        return e("record type");
    }
    let len = u16::from_be_bytes([rec[3], rec[4]]) as usize;
    let Some(hs) = rec.get(5..5 + len) else {
        return e("short record");
    };
    if hs.first() != Some(&0x01) || hs.len() < 4 {
        return e("handshake type");
    }
    let mut p = 4 + 2 + 32; // header, version, random
    let get = |p: usize, n: usize| hs.get(p..p + n);
    let Some(&[sid]) = get(p, 1) else {
        return e("session id");
    };
    p += 1 + sid as usize;
    let Some(cs) = get(p, 2) else {
        return e("cipher suites");
    };
    p += 2 + u16::from_be_bytes([cs[0], cs[1]]) as usize;
    let Some(&[cm]) = get(p, 1) else {
        return e("compression");
    };
    p += 1 + cm as usize;
    let Some(el) = get(p, 2) else { return Ok(None) };
    let end = p + 2 + u16::from_be_bytes([el[0], el[1]]) as usize;
    p += 2;
    while p + 4 <= end.min(hs.len()) {
        let ty = u16::from_be_bytes([hs[p], hs[p + 1]]);
        let n = u16::from_be_bytes([hs[p + 2], hs[p + 3]]) as usize;
        p += 4;
        let Some(ext) = get(p, n) else {
            return e("extension");
        };
        if ty == 0 {
            // server_name: list length, then (type 0 = host_name, len, name).
            if ext.len() >= 5 && ext[2] == 0 {
                let l = u16::from_be_bytes([ext[3], ext[4]]) as usize;
                if let Some(name) = ext.get(5..5 + l) {
                    return Ok(Some(String::from_utf8_lossy(name).to_ascii_lowercase()));
                }
            }
            return e("server_name");
        }
        p += n;
    }
    Ok(None)
}
