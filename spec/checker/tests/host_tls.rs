//! P4 channels over mutual TLS 1.3 with pinned Ed25519 keys (src/tls.rs):
//! pinned peers exchange frames; a peer pinned to a different key gets no
//! delivery; a plain-TCP sender is never heard; `rce host tls-id` is stable,
//! the seed file is private and D logs only the public key.
use rce_spec_check::adapters::chan::{chan_adapters_tls_on, ChanAdapter};
use std::net::TcpListener;
use rce_spec_check::host::{Adapter, Status};
use rce_spec_check::tls::{Identity, hex};
use serde_json::json;
use std::io::Write;

/// A listener on a kernel-chosen port, kept open so no other process can take the
/// port before the adapter listens on it, and its address.
fn listen() -> (TcpListener, String) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let a = l.local_addr().unwrap().to_string();
    (l, a)
}

/// An address for a peer that is never listened on in-process (it is handed to a
/// child process or only dialled).
fn port() -> String {
    format!("127.0.0.1:{}", std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port())
}

fn send(a: &mut [ChanAdapter], to: &str, id: u64, body: &str) -> Status {
    a[0].perform(&json!({"req": id, "peer": to, "id": id, "body": body, "reliable": true})).status
}

fn recv(a: &mut [ChanAdapter]) -> Vec<String> {
    serde_json::from_value(a[1].perform(&json!({"req": 0, "max": 64})).data.unwrap()).unwrap()
}

#[test]
fn pinned_peers_talk_and_others_do_not() {
    let (ia, ib, ic) = (Identity::from_seed([1; 32]).unwrap(), Identity::from_seed([2; 32]).unwrap(), Identity::from_seed([3; 32]).unwrap());
    let ((la, pa), (lb, pb), (lc, pc)) = (listen(), listen(), listen());
    let mut a = chan_adapters_tls_on(la, &pa, &[(pb.clone(), ib.public)], &ia).unwrap();
    let mut b = chan_adapters_tls_on(lb, &pb, &[(pa.clone(), ia.public), (pc.clone(), ic.public)], &ib).unwrap();
    assert_eq!(send(&mut a, &pb, 1, "over tls"), Status::Delivered);
    assert_eq!(send(&mut b, &pa, 7, "and back"), Status::Delivered);
    assert_eq!(recv(&mut b), vec![format!("{pa}\t1\tover tls")]);
    assert_eq!(recv(&mut a), vec![format!("{pb}\t7\tand back")]);

    // c pins the wrong key for b (a's): the handshake fails, nothing arrives
    let mut c = chan_adapters_tls_on(lc, &pc, &[(pb.clone(), ia.public)], &ic).unwrap();
    assert_ne!(send(&mut c, &pb, 2, "wrong pin"), Status::Delivered);
    // a plain TCP sender claiming to be a is not heard
    let mut s = std::net::TcpStream::connect(&pb).unwrap();
    let _ = writeln!(s, "{}", json!({"body": "forged", "from": pa, "id": 99, "kind": "msg", "to": pb}));
    std::thread::sleep(std::time::Duration::from_millis(300));
    let got = recv(&mut b);
    assert!(got.is_empty(), "{got:?}");
}

#[test]
fn tls_id_is_stable_private_and_only_public_in_d() {
    let d = std::env::temp_dir().join(format!("rce-tls-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let log = d.join("a.log");
    let app = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../apps/chat.rce");
    let id = |_: ()| {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).args(["host", "tls-id"]).arg(&app).arg(&log).output().unwrap();
        String::from_utf8_lossy(&o.stdout).trim().to_string()
    };
    let (k1, k2) = (id(()), id(()));
    assert_eq!(k1.len(), 64);
    assert_eq!(k1, k2);
    let key = d.join("a.log.tlskey");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&key).unwrap().permissions().mode() & 0o777, 0o600);
    }
    let seed: [u8; 32] = std::fs::read(&key).unwrap().try_into().unwrap();
    assert_eq!(hex(&Identity::from_seed(seed).unwrap().public), k1);
    // a run under tls logs host.TlsIdentity with the public key, never the seed
    let (me, peer) = (port(), port());
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce"))
        .args(["host", "run", "chan", &me, "tls", "peer", &peer, &hex(&[9; 32])])
        .arg(&app).arg(&log)
        .arg(r#"Setup(me: "ann", peers: [])"#)
        .output().unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(text.contains("host.TlsIdentity") && text.contains(&k1), "{text}");
    assert!(!text.contains(&hex(&seed)), "seed leaked into D");
    let _ = std::fs::remove_dir_all(&d);
}
