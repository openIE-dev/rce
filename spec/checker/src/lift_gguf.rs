//! `rce lift gguf FILE STORE` (10.17): an RCE program from a GGUF file.
//!
//! The program's step is the model's forward pass, written as graph terms
//! over `std.tensor` (6.13); every weight enters a term as
//! `tensor.blob_E(DIGEST, SHAPE)`, its bytes put into the store STORE under
//! their digest (GT-9). Nothing of the file is read at run time but those
//! blobs, so the program names its weights by content, never by path.
//!
//! What is lifted (LIFT-1 to LIFT-4 of 10.17):
//! - the architecture `gpt2`: token and position embeddings, pre-norm
//!   blocks of causal multi-head attention and a `gelu_tanh` feed-forward
//!   layer with biases and residual sums, a final layer norm and the output
//!   projection (tied to the token embedding when the file has no
//!   `output.weight`). Any other architecture is refused with its name.
//! - weights of type F32 and F16 (as f32 tensors; every f16 value is an
//!   f32 value), Q8_0 and Q4_0 (as their quantized values, an `i8` tensor,
//!   and their block scales, an f32 tensor, exactly as the file holds them,
//!   for `matmul_q8_0` and `matmul_q4_0`). Any other type is refused.
//! - GGUF lists a tensor's lengths fastest first; a matrix of lengths
//!   (k, n) is n rows of k, the shape `[n, k]`, and `x W^T` is its product.
//! - The fused projection `attn_qkv` is split into its three row ranges,
//!   each its own blob, so no term slices a tensor.
//!
//! Every operation is at its strongest level: the quantization happened
//! when the file was written, so no lossy operation is in the program.
use rce_ingest_gguf::{GgufDtype, GgufFile, GgufTensorInfo, GgufValue, tensor_byte_size};

/// The value of an f16 bit pattern as f32, exactly.
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let mag = if e == 0 {
        // subnormal: m 2^-24
        (m as f32) * f32::from_bits(0x3380_0000)
    } else if e == 31 {
        f32::from_bits(0x7f80_0000 | (m << 13))
    } else {
        f32::from_bits((((e - 15 + 127) as u32) << 23) | (m << 13))
    };
    f32::from_bits(mag.to_bits() | s)
}

/// A weight as the program holds it.
enum Weight {
    /// an f32 tensor
    F32 { digest: String, shape: Vec<u64> },
    /// a block-quantized matrix: values (i8, [n, k]) and scales (f32, [n, k/32])
    Q { q4: bool, q: String, d: String, n: u64, k: u64 },
}

struct Lift<'a> {
    file: &'a GgufFile,
    bytes: &'a [u8],
    store: &'a rce_fgm::Store,
}

fn put(store: &rce_fgm::Store, b: &[u8]) -> Result<String, String> {
    store.put(b).map(|d| rce_fgm::hex(&d)).map_err(|e| format!("{}: {e}", store.dir().display()))
}

impl Lift<'_> {
    fn info(&self, name: &str) -> Result<&GgufTensorInfo, String> {
        self.file.tensors.iter().find(|t| t.name == name).ok_or_else(|| format!("the file has no tensor `{name}`"))
    }

    fn data(&self, t: &GgufTensorInfo) -> Result<&[u8], String> {
        let start = (self.file.data_offset + t.offset) as usize;
        let end = start + tensor_byte_size(t);
        self.bytes.get(start..end).ok_or_else(|| format!("the tensor `{}` runs past the end of the file", t.name))
    }

    /// The f32 values of an F32 or F16 tensor.
    fn floats(&self, t: &GgufTensorInfo) -> Result<Vec<f32>, String> {
        let b = self.data(t)?;
        match t.dtype {
            GgufDtype::F32 => Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
            GgufDtype::F16 => Ok(b.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()),
            d => Err(format!("the tensor `{}` is of type {d:?}; a vector or table is lifted from F32 or F16", t.name)),
        }
    }

    /// The rows `rows` of tensor `name` (all of them when None), as a weight.
    fn weight(&self, name: &str, rows: Option<(u64, u64)>) -> Result<Weight, String> {
        let t = self.info(name)?;
        let (k, n) = match t.dims.as_slice() {
            [k] => (*k, 1),
            [k, n] => (*k, *n),
            d => return Err(format!("the tensor `{name}` has the lengths {d:?}; one or two are lifted")),
        };
        let (r0, r1) = rows.unwrap_or((0, n));
        match t.dtype {
            GgufDtype::F32 | GgufDtype::F16 => {
                let v = self.floats(t)?;
                let v = &v[(r0 * k) as usize..(r1 * k) as usize];
                let shape = if t.dims.len() == 1 && rows.is_none() { vec![k] } else { vec![r1 - r0, k] };
                Ok(Weight::F32 { digest: put(self.store, &rce_fgm::f32_bytes(v))?, shape })
            }
            GgufDtype::Q8_0 | GgufDtype::Q4_0 => {
                let q4 = t.dtype == GgufDtype::Q4_0;
                let bb = if q4 { 18 } else { 34 };
                if k % 32 != 0 {
                    return Err(format!("the tensor `{name}` has rows of {k}, not a multiple of 32"));
                }
                let b = self.data(t)?;
                let (mut qs, mut ds) = (vec![], vec![]);
                for blk in b.chunks_exact(bb).skip((r0 * k / 32) as usize).take(((r1 - r0) * k / 32) as usize) {
                    ds.push(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
                    if q4 {
                        let lo: Vec<i8> = blk[2..18].iter().map(|x| (x & 0x0f) as i8 - 8).collect();
                        let hi: Vec<i8> = blk[2..18].iter().map(|x| (x >> 4) as i8 - 8).collect();
                        qs.extend(lo);
                        qs.extend(hi);
                    } else {
                        qs.extend(blk[2..34].iter().map(|x| *x as i8));
                    }
                }
                let q = put(self.store, &rce_fgm::i8_bytes(&qs))?;
                let d = put(self.store, &rce_fgm::f32_bytes(&ds))?;
                Ok(Weight::Q { q4, q, d, n: r1 - r0, k })
            }
            d => Err(format!("the tensor `{name}` is of type {d:?}; F32, F16, Q8_0 and Q4_0 are lifted")),
        }
    }

    fn meta_u(&self, key: &str) -> Result<u64, String> {
        match self.file.get_metadata(key) {
            Some(GgufValue::U32(v)) => Ok(*v as u64),
            Some(GgufValue::U64(v)) => Ok(*v),
            Some(GgufValue::I32(v)) if *v >= 0 => Ok(*v as u64),
            _ => Err(format!("the file has no unsigned integer `{key}`")),
        }
    }
}

fn shape(s: &[u64]) -> String {
    format!("[{}]", s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
}

fn blob(w: &Weight) -> String {
    match w {
        Weight::F32 { digest, shape: s } => format!("tensor.blob_f32(\"{digest}\", {})", shape(s)),
        Weight::Q { .. } => unreachable!("a quantized weight enters a product only"),
    }
}

/// `x W^T` for the weight W.
fn product(x: &str, w: &Weight) -> String {
    match w {
        Weight::F32 { .. } => format!("tensor.matmul({x}, tensor.transpose({}))", blob(w)),
        Weight::Q { q4, q, d, n, k } => format!(
            "tensor.matmul_{}({x}, tensor.blob_i8(\"{q}\", [{n}, {k}]), tensor.blob_f32(\"{d}\", [{n}, {}]))",
            if *q4 { "q4_0" } else { "q8_0" },
            k / 32
        ),
    }
}

/// A float literal of RCE source for an f32 value: its exact binary64
/// value, shortest digits, so rounding it to f32 gives the value back.
fn float_lit(x: f32) -> String {
    let s = format!("{:e}", x as f64);
    let (m, e) = s.split_once('e').unwrap_or((&s, "0"));
    let m = if m.contains('.') { m.to_string() } else { format!("{m}.0") };
    format!("{m}e{e}")
}

/// The program lifted from the GGUF bytes `bytes`, its weights put into
/// `store`.
pub fn lift(bytes: &[u8], store: &rce_fgm::Store) -> Result<String, String> {
    let file = GgufFile::parse(bytes).map_err(|e| format!("not a GGUF file: {e}"))?;
    let l = Lift { file: &file, bytes, store };
    let arch = file.architecture().unwrap_or("").to_string();
    if arch != "gpt2" {
        return Err(format!("the architecture `{arch}` is not lifted; `gpt2` is (10.17 LIFT-1)"));
    }
    let d = l.meta_u("gpt2.embedding_length")?;
    let layers = l.meta_u("gpt2.block_count")?;
    let heads = l.meta_u("gpt2.attention.head_count")?;
    let ctx = l.meta_u("gpt2.context_length")?;
    let eps = match file.get_metadata("gpt2.attention.layer_norm_epsilon") {
        Some(GgufValue::F32(e)) => *e,
        _ => return Err("the file has no f32 `gpt2.attention.layer_norm_epsilon`".into()),
    };
    if heads == 0 || d % heads != 0 {
        return Err(format!("{heads} heads do not divide the width {d}"));
    }
    let hd = d / heads;
    let vocab = l.info("token_embd.weight")?.dims.get(1).copied().ok_or("token_embd.weight is not a matrix")?;
    let eps = float_lit(eps);
    let f32w = |name: &str| -> Result<String, String> { Ok(blob(&l.weight(name, None)?)) };
    let mut out = String::new();
    let mut w = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    w(format!("# Lifted by `rce lift gguf` from a GGUF file of the architecture gpt2:"));
    w(format!("# {layers} blocks of width {d}, {heads} heads, a vocabulary of {vocab} and a context"));
    w(format!("# of {ctx}. Every weight is a blob of the store, named by its digest"));
    w("# (spec 6.13 GT-9); every operation is at its strongest level.".into());
    w("use std.tensor".into());
    w(String::new());
    let ids0: Vec<String> = (0..ctx).map(|i| (i % vocab).to_string()).collect();
    w("intent \"the logits of every position of a token sequence\"".into());
    w(format!("    example Tokens(ids: [{}]) -> seen == 1", ids0.join(", ")));
    w(String::new());
    w("state Model".into());
    w(format!("    logits: [[f32; {vocab}]; {ctx}] = [[0.0 as f32; {vocab}]; {ctx}]"));
    w("    seen: u64 = 0".into());
    w(String::new());
    w(format!("event Tokens(ids: [i32; {ctx}])"));
    w(String::new());
    w("step Model on Tokens(ids)".into());
    let pos: Vec<String> = (0..ctx).map(|i| i.to_string()).collect();
    w(format!("    let pos: [i32; {ctx}] = [{}]", pos.join(", ")));
    w(format!(
        "    let x0 = tensor.add(tensor.embed({}, ids), tensor.embed({}, pos))",
        f32w("token_embd.weight")?,
        f32w("position_embd.weight")?
    ));
    for i in 0..layers {
        let p = |s: &str| format!("blk.{i}.{s}");
        w(format!("    # block {i}"));
        w(format!(
            "    let a{i} = tensor.layernorm(x{i}, {}, {}, {eps})",
            f32w(&p("attn_norm.weight"))?,
            f32w(&p("attn_norm.bias"))?
        ));
        let qkv_b = l.floats(l.info(&p("attn_qkv.bias"))?)?;
        for (j, name) in ["q", "k", "v"].iter().enumerate() {
            let wt = l.weight(&p("attn_qkv.weight"), Some((j as u64 * d, (j as u64 + 1) * d)))?;
            let bias = put(store, &rce_fgm::f32_bytes(&qkv_b[j * d as usize..(j + 1) * d as usize]))?;
            w(format!(
                "    let {name}{i} = tensor.permute(tensor.reshape(tensor.add({}, tensor.blob_f32(\"{bias}\", [{d}])), [{ctx}, {heads}, {hd}]), [1, 0, 2])",
                product(&format!("a{i}"), &wt)
            ));
        }
        w(format!("    let o{i} = tensor.reshape(tensor.permute(tensor.attention_causal(q{i}, k{i}, v{i}), [1, 0, 2]), [{ctx}, {d}])"));
        w(format!(
            "    let h{i} = tensor.add(x{i}, tensor.add({}, {}))",
            product(&format!("o{i}"), &l.weight(&p("attn_output.weight"), None)?),
            f32w(&p("attn_output.bias"))?
        ));
        w(format!(
            "    let f{i} = tensor.layernorm(h{i}, {}, {}, {eps})",
            f32w(&p("ffn_norm.weight"))?,
            f32w(&p("ffn_norm.bias"))?
        ));
        let up = format!("tensor.add({}, {})", product(&format!("f{i}"), &l.weight(&p("ffn_up.weight"), None)?), f32w(&p("ffn_up.bias"))?);
        w(format!(
            "    let x{} = tensor.add(h{i}, tensor.add({}, {}))",
            i + 1,
            product(&format!("tensor.gelu_tanh({up})"), &l.weight(&p("ffn_down.weight"), None)?),
            f32w(&p("ffn_down.bias"))?
        ));
    }
    let out_w = if file.tensors.iter().any(|t| t.name == "output.weight") { "output.weight" } else { "token_embd.weight" };
    w(format!(
        "    logits = {}",
        product(
            &format!("tensor.layernorm(x{layers}, {}, {}, {eps})", f32w("output_norm.weight")?, f32w("output_norm.bias")?),
            &l.weight(out_w, None)?
        )
    ));
    w("    seen += 1".into());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_values_are_f32_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(float_lit(1e-5), "9.999999747378752e-6");
        assert_eq!(float_lit(2.0), "2.0e0");
    }
}
