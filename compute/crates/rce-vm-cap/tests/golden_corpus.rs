//! Golden corpus integration test for the capability wire format.
//!
//! Two purposes:
//! 1. Lock the byte sequence emitted by `WireRequest::encode` /
//!    `WireResponse::encode` for canonical sample inputs. Any change
//!    that perturbs the byte sequence will fail the test loudly so we
//!    notice schema drift before downstream bindings break.
//! 2. Generate the corpus files that Go / JVM / Swift bindings
//!    consume in their wire-compat tests. Set `RCE_REGEN_GOLDEN=1`
//!    in the env to regenerate; default behaviour is byte-exact
//!    comparison.
//!
//! The corpus lives at `tests/golden/cap-*.cbor`. Each file is the
//! exact bytes a passing rce-vm-cap encoder produces for a named
//! sample. Cross-language bindings load the file, round-trip it
//! through their CBOR codec, and assert byte-equality with the
//! original.

use rce_vm_cap::{WireRequest, WireResponse};
use std::fs;
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("golden");
    p
}

fn assert_or_regen(path: &PathBuf, bytes: &[u8]) {
    if std::env::var_os("RCE_REGEN_GOLDEN").is_some() {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
        eprintln!("[regen] wrote {} ({} bytes)", path.display(), bytes.len());
        return;
    }
    let want = fs::read(path).unwrap_or_else(|e| {
        panic!(
            "golden file missing: {} ({e}). Run `RCE_REGEN_GOLDEN=1 cargo test -p rce-vm-cap` to regenerate.",
            path.display()
        )
    });
    assert_eq!(
        bytes,
        want.as_slice(),
        "golden mismatch: {} ({} bytes encoded, {} bytes on disk). Schema drift?",
        path.display(),
        bytes.len(),
        want.len()
    );
}

#[test]
fn golden_request_full() {
    let req = WireRequest {
        capability_id: "joulesperbit.pay.ledger".into(),
        version: "1.0.0".into(),
        method: "transfer".into(),
        // CBOR [42, 100, 1000]
        args: vec![0x83, 0x18, 0x2a, 0x18, 0x64, 0x19, 0x03, 0xe8],
        effects: vec!["regulatory:financial".into(), "pii:read".into()],
        delegation_chain: Some(vec![0xde, 0xad, 0xbe, 0xef]),
        nonce: Some([0x11; 16]),
        trace_id: Some([0x22; 16]),
    };
    let bytes = req.encode().expect("encode");
    let mut path = golden_dir();
    path.push("cap-req-full.cbor");
    assert_or_regen(&path, &bytes);

    // Sanity round-trip on the bytes we just compared.
    let back = WireRequest::decode(&bytes).expect("decode golden");
    assert_eq!(req, back);
}

#[test]
fn golden_request_minimal() {
    let req = WireRequest {
        capability_id: "test.cap".into(),
        version: "0.1".into(),
        method: "noop".into(),
        args: vec![],
        effects: vec![],
        delegation_chain: None,
        nonce: None,
        trace_id: None,
    };
    let bytes = req.encode().expect("encode");
    let mut path = golden_dir();
    path.push("cap-req-minimal.cbor");
    assert_or_regen(&path, &bytes);
    let back = WireRequest::decode(&bytes).expect("decode golden");
    assert_eq!(req, back);
}

#[test]
fn golden_response_ok() {
    let resp = WireResponse::Ok {
        result: vec![0x18, 0x2a], // CBOR uint 42
    };
    let bytes = resp.encode().expect("encode");
    let mut path = golden_dir();
    path.push("cap-resp-ok.cbor");
    assert_or_regen(&path, &bytes);
    let back = WireResponse::decode(&bytes).expect("decode golden");
    assert_eq!(resp, back);
}

#[test]
fn golden_response_err() {
    let resp = WireResponse::Err {
        code: "insufficient_funds".into(),
        message: "balance < amount".into(),
        details: Some(vec![0xa1, 0x01, 0x18, 0x2a]), // CBOR {1: 42}
    };
    let bytes = resp.encode().expect("encode");
    let mut path = golden_dir();
    path.push("cap-resp-err.cbor");
    assert_or_regen(&path, &bytes);
    let back = WireResponse::decode(&bytes).expect("decode golden");
    assert_eq!(resp, back);
}
