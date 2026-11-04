//! Probe key dimensions of the Gemma 4v mmproj's vision tower:
//! position embed shape, n_merge candidates, image normalization.
use rce_ingest_gguf::{GgufFile, GgufValue};
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B-mmproj-q8_0.gguf";
    let b = std::fs::read(p).unwrap();
    let g = GgufFile::parse(&b).unwrap();
    // Print specific keys + tensor dims.
    for k in ["clip.vision.image_size", "clip.vision.patch_size", "clip.vision.block_count",
              "clip.vision.attention.head_count", "clip.vision.embedding_length",
              "clip.vision.projection_dim", "clip.vision.feed_forward_length",
              "clip.vision.projector.scale_factor", "clip.vision.projector_type",
              "clip.vision.attention.layer_norm_epsilon",
              "clip.vision.attention.rope_theta", "clip.vision.rope.theta",
              "clip.use_silu", "clip.use_gelu",
              "clip.vision.image_mean", "clip.vision.image_std"] {
        if let Some(v) = g.get_metadata(k) {
            let s = match v {
                GgufValue::U32(x) => format!("U32({x})"),
                GgufValue::F32(x) => format!("F32({x})"),
                GgufValue::Bool(x) => format!("{x}"),
                GgufValue::String(x) => format!("{:?}", x),
                GgufValue::Array(a) => format!("Array[{}]: {:?}", a.len(), &a[..a.len().min(3)]),
                o => format!("{o:?}"),
            };
            println!("{k} = {s}");
        }
    }
    println!("---key vision tensors---");
    for name in ["v.patch_embd.weight", "v.patch_embd.bias",
                 "v.position_embd.weight", "v.post_ln.weight", "v.post_ln.bias",
                 "v.pre_ln.weight",
                 "mm.input_projection.weight",
                 "mm.input_projection.bias",
                 "v.std_bias.weight", "v.std_scale.weight",
                 "v.blk.0.ln1.weight", "v.blk.0.ln1.bias",
                 "v.blk.0.attn_q.weight", "v.blk.0.attn_q.bias",
                 "v.blk.0.attn_k.weight", "v.blk.0.attn_v.weight",
                 "v.blk.0.attn_output.weight", "v.blk.0.attn_output.bias",
                 "v.blk.0.ln2.weight", "v.blk.0.ln2.bias",
                 "v.blk.0.ffn_up.weight", "v.blk.0.ffn_up.bias",
                 "v.blk.0.ffn_down.weight", "v.blk.0.ffn_down.bias",
                 "v.blk.0.ffn_gate.weight",
                 "v.blk.0.ffn_up.input_max", "v.blk.0.ffn_up.input_min",
                 "v.blk.0.ffn_up.output_max", "v.blk.0.ffn_up.output_min"] {
        if let Some(t) = g.tensors.iter().find(|t| t.name == name) {
            println!("{:55} dims={:?} {:?}", t.name, t.dims, t.dtype);
        } else {
            println!("{:55} ABSENT", name);
        }
    }
}
