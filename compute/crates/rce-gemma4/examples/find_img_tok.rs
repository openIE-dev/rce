//! Probe Gemma 4's chat-template tokenization.
use rce_ingest_gguf::{GgufFile, GgufValue, Tokenizer};
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    let b = std::fs::read(p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    let tk = Tokenizer::from_gguf(&g).unwrap();
    let is_gpt2 = g.metadata.iter().find(|kv| kv.key == "tokenizer.ggml.model")
        .map(|kv| matches!(&kv.value, GgufValue::String(s) if s == "gpt2"))
        .unwrap_or(false);
    eprintln!("is_gpt2 = {is_gpt2}");
    let enc = |s: &str| {
        let ids = if is_gpt2 { tk.encode_gpt2(s, false) } else { tk.encode(s, false) };
        eprintln!("encode({s:?}) = {ids:?}");
        ids
    };
    enc("<start_of_turn>user\n");
    enc("<end_of_turn>\n");
    enc("<start_of_turn>model\n");
    enc("<|image|>");
    enc("Describe this image briefly.");
    enc("Hello");
    eprintln!("lookup(\"\\n\") = {:?}", tk.lookup_token("\n"));
}
