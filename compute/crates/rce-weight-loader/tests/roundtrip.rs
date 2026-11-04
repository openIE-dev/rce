//! End-to-end safetensors round-trip tests.
//!
//! These tests build real safetensors byte buffers, write them to disk,
//! load them via mmap, and assert the full binary round-trip — no stubs,
//! no mocks.

use std::collections::BTreeMap;

use rce_weight_loader::{write_safetensors, LoadError, SafetensorsFile};
use tempfile::NamedTempFile;

fn f32_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 4);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn i64_bytes(xs: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(xs.len() * 8);
    for &x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

#[test]
fn roundtrip_single_f32_tensor_in_memory() {
    let weight: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
    let bytes = f32_bytes(&weight);

    let st = write_safetensors(
        &[("linear.weight", "f32", &[3, 4], &bytes)],
        None,
    );
    let file = SafetensorsFile::from_bytes(&st).unwrap();
    assert_eq!(file.tensors().len(), 1);

    let t = file.get("linear.weight").unwrap();
    assert_eq!(t.dtype, "f32");
    assert_eq!(t.shape, vec![3, 4]);
    assert_eq!(t.numel(), 12);
    assert_eq!(t.data.len(), 48);

    let view = t.as_f32().unwrap();
    assert_eq!(view, &weight[..]);
}

#[test]
fn roundtrip_multi_tensor_on_disk() {
    let w1: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let w2: Vec<i64> = (0..4).collect();
    let w3: Vec<f32> = vec![0.1, 0.2, 0.3];

    let st = write_safetensors(
        &[
            ("layer1.weight", "f32", &[2, 4], &f32_bytes(&w1)),
            ("layer1.bias", "i64", &[4], &i64_bytes(&w2)),
            ("layer2.bias", "f32", &[3], &f32_bytes(&w3)),
        ],
        None,
    );

    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &st).unwrap();

    let file = SafetensorsFile::load(tmp.path()).unwrap();
    assert_eq!(file.tensors().len(), 3);

    let t1 = file.get("layer1.weight").unwrap();
    assert_eq!(t1.shape, vec![2, 4]);
    assert_eq!(t1.as_f32().unwrap(), &w1[..]);

    let t2 = file.get("layer1.bias").unwrap();
    assert_eq!(t2.dtype, "i64");
    assert_eq!(t2.data.len(), 32);

    let t3 = file.get("layer2.bias").unwrap();
    assert_eq!(t3.as_f32().unwrap(), &w3[..]);
}

#[test]
fn metadata_roundtrip() {
    let mut meta: BTreeMap<String, String> = BTreeMap::new();
    meta.insert("framework".into(), "pytorch".into());
    meta.insert("version".into(), "2.0".into());

    let bytes = f32_bytes(&[1.0, 2.0]);
    let st = write_safetensors(
        &[("w", "f32", &[2], &bytes)],
        Some(&meta),
    );
    let file = SafetensorsFile::from_bytes(&st).unwrap();
    assert_eq!(file.metadata().get("framework"), Some(&"pytorch".to_string()));
    assert_eq!(file.metadata().get("version"), Some(&"2.0".to_string()));
    assert_eq!(file.tensors().len(), 1);
}

#[test]
fn rejects_truncated_file() {
    let err = SafetensorsFile::from_bytes(&[0u8; 4]).unwrap_err();
    assert!(matches!(err, LoadError::Truncated));
}

#[test]
fn rejects_header_length_overflow() {
    // 8 bytes of header-length prefix claiming a huge header, but no
    // body follows.
    let mut bad = Vec::with_capacity(8);
    bad.extend_from_slice(&(u64::MAX).to_le_bytes());
    let err = SafetensorsFile::from_bytes(&bad).unwrap_err();
    assert!(matches!(err, LoadError::HeaderLengthOverflow(_, _)));
}

#[test]
fn rejects_unknown_dtype() {
    // Build a malformed header with a bogus dtype. We can't use
    // write_safetensors for this because it normalises known dtypes,
    // so craft the JSON by hand.
    let header_json = br#"{"w":{"dtype":"NONSENSE","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json);
    bytes.extend_from_slice(&[0u8; 8]); // payload
    let err = SafetensorsFile::from_bytes(&bytes).unwrap_err();
    match err {
        LoadError::UnsupportedDtype { name, dtype } => {
            assert_eq!(name, "w");
            assert_eq!(dtype, "NONSENSE");
        }
        other => panic!("expected UnsupportedDtype, got {other:?}"),
    }
}

#[test]
fn rejects_span_mismatch() {
    // Header says shape [2] of F32 (= 8 bytes) but data_offsets only
    // span 4 bytes.
    let header_json = br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json);
    bytes.extend_from_slice(&[0u8; 4]);
    let err = SafetensorsFile::from_bytes(&bytes).unwrap_err();
    match err {
        LoadError::SpanMismatch { name, span, expected } => {
            assert_eq!(name, "w");
            assert_eq!(span, 4);
            assert_eq!(expected, 8);
        }
        other => panic!("expected SpanMismatch, got {other:?}"),
    }
}

#[test]
fn rejects_offsets_oob() {
    // data_offsets end > payload length.
    let header_json = br#"{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,32]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json);
    bytes.extend_from_slice(&[0u8; 8]); // only 8 bytes of payload
    let err = SafetensorsFile::from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, LoadError::OffsetsOob { .. }));
}

#[test]
fn tensors_returned_in_file_order() {
    // Give them non-alphabetical names to confirm the ordering is by
    // offset (which is the insertion order in write_safetensors), not
    // lexicographic.
    let bytes = f32_bytes(&[1.0]);
    let st = write_safetensors(
        &[
            ("zulu", "f32", &[1], &bytes),
            ("alpha", "f32", &[1], &bytes),
            ("mike", "f32", &[1], &bytes),
        ],
        None,
    );
    let file = SafetensorsFile::from_bytes(&st).unwrap();
    let names: Vec<&str> = file.tensors().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["zulu", "alpha", "mike"]);
}

#[test]
fn large_tensor_10mb_roundtrip() {
    // 10 MB of f32 — representative of a mid-size weight matrix
    // (~2.5M parameters). Exercises mmap + linear-copy path.
    let n = 10 * 1024 * 1024 / 4;
    let weight: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let bytes = f32_bytes(&weight);

    let st = write_safetensors(
        &[("big", "f32", &[n], &bytes)],
        None,
    );
    let tmp = NamedTempFile::new().unwrap();
    std::fs::write(tmp.path(), &st).unwrap();

    let file = SafetensorsFile::load(tmp.path()).unwrap();
    let t = file.get("big").unwrap();
    assert_eq!(t.numel(), n);
    let view = t.as_f32().unwrap();
    assert_eq!(view.len(), n);
    assert_eq!(view[0], 0.0);
    assert_eq!(view[n - 1], (n - 1) as f32);
    // Spot-check a few interior elements.
    assert_eq!(view[12345], 12345.0);
    assert_eq!(view[n / 2], (n / 2) as f32);
}
