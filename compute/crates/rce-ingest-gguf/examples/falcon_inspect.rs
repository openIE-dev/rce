// One-off inspector for Falcon3-Mamba GGUF metadata + tensor shapes.
// Run via: cargo run --release -p rce-ingest-gguf --example falcon_inspect

use rce_ingest_gguf::{GgufFile, GgufValue};
use std::fs::File;

fn main() {
    let path = std::env::args().nth(1).expect("usage: falcon_inspect <gguf>");
    let file = File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let gguf = GgufFile::parse(&mmap).unwrap();

    println!("─── metadata (ssm/mamba/tokenizer) ───");
    for kv in &gguf.metadata {
        let k = &kv.key;
        if k.starts_with("mamba.") || k.starts_with("ssm.") || k.starts_with("tokenizer.ggml.")
            || k.contains("vocab") || k.contains("bos_token") || k.contains("eos_token")
        {
            let preview = match &kv.value {
                GgufValue::U32(v) => format!("u32 = {v}"),
                GgufValue::I32(v) => format!("i32 = {v}"),
                GgufValue::F32(v) => format!("f32 = {v}"),
                GgufValue::Bool(v) => format!("bool = {v}"),
                GgufValue::String(s) => format!("string = {:?}", &s[..s.len().min(80)]),
                GgufValue::Array(a) => format!("array[{}]", a.len()),
                _ => "...".into(),
            };
            println!("  {k} = {preview}");
        }
    }

    println!("\n─── tensor shapes (blk.0 + top-level) ───");
    for t in gguf.tensors.iter().filter(|t| t.name.starts_with("blk.0.") || t.name == "token_embd.weight" || t.name == "output.weight" || t.name == "output_norm.weight") {
        println!("  {:<40} dims={:?}  dtype={:?}", t.name, t.dims, t.dtype);
    }

    // Find which blocks have attention.
    let attn_blocks: Vec<usize> = (0..256)
        .filter(|i| gguf.tensors.iter().any(|t| t.name == format!("blk.{i}.attn_q.weight")))
        .collect();
    if !attn_blocks.is_empty() {
        println!("\n─── attention block indices: {:?} ───", attn_blocks);
        let first = attn_blocks[0];
        println!("─── tensors for first attn block (blk.{first}) ───");
        for t in gguf.tensors.iter().filter(|t| t.name.starts_with(&format!("blk.{first}."))) {
            println!("  {:<40} dims={:?}  dtype={:?}", t.name, t.dims, t.dtype);
        }
    }

    println!("\n─── all non-tokenizer/general metadata ───");
    for kv in &gguf.metadata {
        let k = &kv.key;
        if !k.starts_with("tokenizer.") && !k.starts_with("general.") {
            let preview = match &kv.value {
                GgufValue::U32(v) => format!("u32 = {v}"),
                GgufValue::I32(v) => format!("i32 = {v}"),
                GgufValue::F32(v) => format!("f32 = {v}"),
                GgufValue::Bool(v) => format!("bool = {v}"),
                GgufValue::String(s) => format!("string = {:?}", &s[..s.len().min(80)]),
                GgufValue::Array(a) => format!("array[{}]", a.len()),
                _ => "...".into(),
            };
            println!("  {k} = {preview}");
        }
    }
}
