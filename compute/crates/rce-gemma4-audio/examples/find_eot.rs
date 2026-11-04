use rce_ingest_gguf::{GgufFile, GgufValue};
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-text-q8_0.gguf";
    let b = std::fs::read(p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    let tokens = g.metadata.iter().find(|kv| kv.key == "tokenizer.ggml.tokens").unwrap();
    let eot_kv = g.metadata.iter().find(|kv| kv.key.contains("eot") || kv.key.contains("end_of_turn") || kv.key.contains("eog"));
    println!("eot metadata: {:?}", eot_kv);
    if let GgufValue::Array(arr) = &tokens.value {
        for (i, v) in arr.iter().enumerate() {
            if let GgufValue::String(s) = v {
                if s == "<|turn>" || s == "<turn|>" || s == "<eos>" || s == "<bos>" || s == "<pad>"
                    || s == "<|tool>" || s == "<tool|>" || s.starts_with("[multi") {
                    println!("token {} = {:?}", i, s);
                }
            }
        }
    }
}
