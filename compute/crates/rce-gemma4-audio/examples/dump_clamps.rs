use rce_ingest_gguf::GgufFile;
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let b = std::fs::read(p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    let names = [
        "a.input_projection",
        "a.blk.0.ffn_up", "a.blk.0.ffn_down",
        "a.blk.0.attn_q", "a.blk.0.attn_k", "a.blk.0.attn_v", "a.blk.0.attn_out",
        "a.blk.0.attn_k_rel",
        "a.blk.0.conv_pw1", "a.blk.0.conv_pw2",
        "a.blk.0.ffn_up_1", "a.blk.0.ffn_down_1",
        "a.pre_encode.out",
        "mm.a.input_projection",
    ];
    let bytes: &[u8] = &b;
    use rce_ingest_gguf::read_tensor_f32;
    for n in names {
        let imax = read_tensor_f32(bytes, &g, &format!("{n}.input_max")).ok().and_then(|v| v.first().copied());
        let imin = read_tensor_f32(bytes, &g, &format!("{n}.input_min")).ok().and_then(|v| v.first().copied());
        let omax = read_tensor_f32(bytes, &g, &format!("{n}.output_max")).ok().and_then(|v| v.first().copied());
        let omin = read_tensor_f32(bytes, &g, &format!("{n}.output_min")).ok().and_then(|v| v.first().copied());
        println!("{n:30}  in [{:?}, {:?}]  out [{:?}, {:?}]", imin, imax, omin, omax);
    }
}
