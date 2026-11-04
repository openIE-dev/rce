//! Dump Gemma 4 mmproj structure: vision + audio towers + projectors.
use rce_ingest_gguf::{GgufFile, GgufValue};
fn main() {
    let p = std::env::args().nth(1).unwrap_or_else(|| {
        "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/\
         gemma-4-E2B-mmproj-q8_0.gguf".into()
    });
    let b = std::fs::read(&p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    // Metadata keys
    for kv in &g.metadata {
        let k = &kv.key;
        let pick = k.starts_with("general.")
            || k.starts_with("clip")
            || k.contains("vision")
            || k.contains("audio")
            || k.contains("proj")
            || k.contains("image")
            || k.contains("patch")
            || k.contains("mean")
            || k.contains("std");
        if !pick { continue; }
        let v = match &kv.value {
            GgufValue::U32(x) => format!("U32({x})"),
            GgufValue::I32(x) => format!("I32({x})"),
            GgufValue::U64(x) => format!("U64({x})"),
            GgufValue::F32(x) => format!("F32({x})"),
            GgufValue::Bool(x) => format!("{x}"),
            GgufValue::String(s) => format!("{:?}", s),
            GgufValue::Array(a) => format!("Array[len={}] first={:?}", a.len(), &a[..a.len().min(4)]),
            o => format!("{o:?}"),
        };
        println!("{k} = {v}");
    }
    // Tensor names grouped by prefix.
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    for t in &g.tensors {
        let prefix: String = t.name.split('.').take(2).collect::<Vec<_>>().join(".");
        *groups.entry(prefix).or_insert(0) += 1;
    }
    println!("\n--- tensor groups ---");
    for (k, n) in &groups { println!("{k:40} ×{n}"); }
    // First few tensors per top-level prefix.
    let mut top_seen: BTreeMap<String, usize> = BTreeMap::new();
    println!("\n--- examples (first 4 per top prefix) ---");
    for t in &g.tensors {
        let top = t.name.split('.').next().unwrap_or("").to_string();
        let c = top_seen.entry(top.clone()).or_insert(0);
        if *c < 4 {
            println!("{:55} dims={:?} {:?}", t.name, t.dims, t.dtype);
            *c += 1;
        }
    }
    println!("\ntotal tensors = {}", g.tensors.len());
}
