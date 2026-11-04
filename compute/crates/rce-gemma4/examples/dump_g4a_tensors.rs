//! Probe Gemma 4a audio tower tensors + audio hparams.
use rce_ingest_gguf::{GgufFile, GgufValue};
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let b = std::fs::read(p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    for kv in &g.metadata {
        let k = &kv.key;
        if k.contains("audio") || k.contains("conformer") || k.contains("sample_rate")
            || k.contains("mel") || k.contains("a.proj") || k.contains("frame")
            || k.contains("chunk") {
            let v = match &kv.value {
                GgufValue::U32(x) => format!("U32({x})"),
                GgufValue::I32(x) => format!("I32({x})"),
                GgufValue::F32(x) => format!("F32({x})"),
                GgufValue::Bool(x) => format!("{x}"),
                GgufValue::String(s) => format!("{:?}", s),
                GgufValue::Array(a) => format!("Array[len={}]", a.len()),
                o => format!("{o:?}"),
            };
            println!("{k} = {v}");
        }
    }
    println!("---key audio tensors---");
    let target_names = [
        "a.input_projection.weight",
        "a.pre_encode.0.weight",
        "a.pre_encode.1.weight",
        "a.conv1d.0.weight",
        "a.conv1d.1.weight",
        "a.blk.0.ln1.weight",
        "a.blk.0.attn_q.weight",
        "a.blk.0.attn_k.weight",
        "a.blk.0.attn_v.weight",
        "a.blk.0.attn_q_norm.weight",
        "a.blk.0.attn_k_norm.weight",
        "a.blk.0.attn_out.weight",
        "a.blk.0.attn_post_norm.weight",
        "a.blk.0.ln2.weight",
        "a.blk.0.ffn_up.weight",
        "a.blk.0.ffn_down.weight",
        "a.blk.0.ffn_gate.weight",
        "a.blk.0.ffn_post_norm.weight",
        "mm.a.input_projection.weight",
    ];
    for n in target_names {
        if let Some(t) = g.tensors.iter().find(|t| t.name == n) {
            println!("{:50} dims={:?} {:?}", t.name, t.dims, t.dtype);
        }
    }
    // Also list ALL a.blk.0.* tensor weight names to find missing structure.
    println!("---a.blk.0.* weight tensors (full list)---");
    let mut names: Vec<_> = g.tensors.iter()
        .filter(|t| t.name.starts_with("a.blk.0.") && t.name.ends_with(".weight"))
        .map(|t| (t.name.clone(), t.dims.clone()))
        .collect();
    names.sort();
    for (n, d) in &names {
        println!("{n:55} dims={d:?}");
    }
}
