use rce_ingest_gguf::{GgufFile, Tokenizer};
use std::fs::File;

fn main() {
    let path = std::env::args().nth(1).expect("usage: tok_check <gguf>");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "The capital of France is".to_string());
    let file = File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let gguf = GgufFile::parse(&mmap).unwrap();
    let tk = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let ids = tk.encode_gpt2(&prompt, false);
    let decoded = tk.decode_sequence(&ids, false);
    println!("prompt   : {:?}", prompt);
    println!("ids      : {:?}", ids);
    println!("decoded  : {:?}", decoded);
    println!("match    : {}", prompt == decoded);
    println!();
    // Print per-token
    for (i, &id) in ids.iter().enumerate() {
        println!("  [{i}] {id} -> {:?}", tk.decode_sequence(&[id], false));
    }
}
