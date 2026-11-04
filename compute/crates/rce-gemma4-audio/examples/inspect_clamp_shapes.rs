use rce_ingest_gguf::{GgufFile, read_tensor_f32};
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let b = std::fs::read(p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    for t in g.tensors.iter().filter(|t|
        t.name.contains("a.blk.0.attn_q")
            || t.name.contains("a.blk.0.conv_pw2")
            || t.name.contains("a.blk.0.attn_out"))
    {
        println!("{:50} dims={:?}  dtype={:?}  offset={}", t.name, t.dims, t.dtype, t.offset);
        if t.name.ends_with("_max") || t.name.ends_with("_min") {
            if let Ok(vs) = read_tensor_f32(&b, &g, &t.name) {
                println!("    values: {:?}", vs);
            }
        }
    }
}
