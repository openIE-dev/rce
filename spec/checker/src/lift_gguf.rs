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
//!   `output.weight`).
//! - the architecture `llama`: a token embedding, pre-norm blocks of RMS
//!   norm, causal grouped-query attention (GQA: the query heads a multiple
//!   of the key and value heads) with the rotary position embedding of
//!   adjacent pairs (`tensor.rope`) on q and k, and a SwiGLU feed-forward
//!   layer (`silu(f Wg^T) * (f Wu^T)`, then `Wd`), residual sums, a final
//!   RMS norm and the output projection (tied when absent). The rotary
//!   frequencies `base^(-2i/hd)` are computed once by the lift in binary64
//!   and rounded once to f32, and enter the program as a blob.
//!   Any other architecture is refused with its name.
//! - weights of type F32 and F16 (as f32 tensors; every f16 value is an
//!   f32 value), Q8_0 and Q4_0 (as their quantized values, an `i8` tensor,
//!   and their block scales, an f32 tensor, exactly as the file holds them,
//!   for `matmul_q8_0` and `matmul_q4_0`), and Q4_K, Q5_K and Q6_K (as
//!   their values, an `i8` tensor in position order, and their sub-block
//!   scales `d sc` and offsets `dmin m`, f32 tensors whose every value is
//!   exact, for `matmul_q4_k`, `matmul_q5_k` and `matmul_q6_k`). Any other
//!   type is refused.
//! - GGUF lists a tensor's lengths fastest first; a matrix of lengths
//!   (k, n) is n rows of k, the shape `[n, k]`, and `x W^T` is its product.
//! - The fused projection `attn_qkv` is split into its three row ranges,
//!   each its own blob, so no term slices a tensor.
//!
//! Every operation is at its strongest level: the quantization happened
//! when the file was written, so no lossy operation is in the program.
use rce_vm_backend_cpu::ml::kquant::{self, KFmt};
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
    /// a K-quant matrix: values (i8, [n, k]), sub-block scales and (Q4_K,
    /// Q5_K) offsets (f32, [n, k/32], or [n, k/16] for Q6_K)
    K { f: KFmt, q: String, s: String, o: Option<String>, n: u64, k: u64 },
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
            GgufDtype::Q4K | GgufDtype::Q5K | GgufDtype::Q6K => {
                let f = match t.dtype {
                    GgufDtype::Q4K => KFmt::Q4K,
                    GgufDtype::Q5K => KFmt::Q5K,
                    _ => KFmt::Q6K,
                };
                if k % 256 != 0 {
                    return Err(format!("the tensor `{name}` has rows of {k}, not a multiple of 256"));
                }
                let b = self.data(t)?;
                let per = (k / 256) as usize;
                let (mut qs, mut ss, mut os) = (vec![], vec![], vec![]);
                for blk in b.chunks_exact(f.block_bytes()).skip(r0 as usize * per).take((r1 - r0) as usize * per) {
                    let kb = kquant::unpack(blk, f, f16_to_f32);
                    let (s, o) = kb.scales();
                    qs.extend(kb.q.iter().map(|v| *v as i8));
                    ss.extend(s);
                    os.extend(o);
                }
                let q = put(self.store, &rce_fgm::i8_bytes(&qs))?;
                let s = put(self.store, &rce_fgm::f32_bytes(&ss))?;
                let o = if f == KFmt::Q6K { None } else { Some(put(self.store, &rce_fgm::f32_bytes(&os))?) };
                Ok(Weight::K { f, q, s, o, n: r1 - r0, k })
            }
            d => Err(format!("the tensor `{name}` is of type {d:?}; F32, F16, Q8_0, Q4_0, Q4_K, Q5_K and Q6_K are lifted")),
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
        Weight::Q { .. } | Weight::K { .. } => unreachable!("a quantized weight enters a product only"),
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
        Weight::K { f, q, s, o, n, k } => {
            let sub = f.sub() as u64;
            let o = o.as_ref().map(|o| format!(", tensor.blob_f32(\"{o}\", [{n}, {}])", k / sub)).unwrap_or_default();
            format!("tensor.{}({x}, tensor.blob_i8(\"{q}\", [{n}, {k}]), tensor.blob_f32(\"{s}\", [{n}, {}]){o})", f.names().0, k / sub)
        }
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
    match arch.as_str() {
        "gpt2" => lift_gpt2(&l),
        "llama" => lift_llama(&l),
        _ => Err(format!("the architecture `{arch}` is not lifted; `gpt2` and `llama` are (10.17 LIFT-1)")),
    }
}

/// The rotary frequencies of a head of `hd`: base^(-2i/hd) for i in
/// 0..hd/2, each computed in binary64 and rounded once to f32.
pub fn rope_freqs(base: f32, hd: usize) -> Vec<f32> {
    (0..hd / 2).map(|i| (base as f64).powf(-((2 * i) as f64) / hd as f64) as f32).collect()
}

fn lift_llama(l: &Lift) -> Result<String, String> {
    let (file, store) = (l.file, l.store);
    let d = l.meta_u("llama.embedding_length")?;
    let layers = l.meta_u("llama.block_count")?;
    let heads = l.meta_u("llama.attention.head_count")?;
    let kvh = l.meta_u("llama.attention.head_count_kv").unwrap_or(heads);
    let ctx = l.meta_u("llama.context_length")?;
    let eps = match file.get_metadata("llama.attention.layer_norm_rms_epsilon") {
        Some(GgufValue::F32(e)) => *e,
        _ => return Err("the file has no f32 `llama.attention.layer_norm_rms_epsilon`".into()),
    };
    let base = match file.get_metadata("llama.rope.freq_base") {
        Some(GgufValue::F32(b)) => *b,
        None => 10000.0,
        _ => return Err("`llama.rope.freq_base` is not an f32".into()),
    };
    if heads == 0 || kvh == 0 || d % heads != 0 || heads % kvh != 0 {
        return Err(format!("{heads} query heads and {kvh} key and value heads over the width {d}: the heads divide the width and the key heads the query heads"));
    }
    let hd = d / heads;
    if hd % 2 != 0 {
        return Err(format!("a head of {hd} is not of even length; the rotary embedding rotates pairs"));
    }
    let vocab = l.info("token_embd.weight")?.dims.get(1).copied().ok_or("token_embd.weight is not a matrix")?;
    let eps = float_lit(eps);
    let f32w = |name: &str| -> Result<String, String> { Ok(blob(&l.weight(name, None)?)) };
    let fr = put(store, &rce_fgm::f32_bytes(&rope_freqs(base, hd as usize)))?;
    let mut out = String::new();
    let mut w = |s: String| {
        out.push_str(&s);
        out.push('\n');
    };
    w("# Lifted by `rce lift gguf` from a GGUF file of the architecture llama:".into());
    w(format!("# {layers} blocks of width {d}, {heads} query heads and {kvh} key and value heads,"));
    w(format!("# a vocabulary of {vocab} and a context of {ctx}. Every weight is a blob of"));
    w("# the store, named by its digest (spec 6.13 GT-9); every operation is at".into());
    w("# its strongest level.".into());
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
    w(format!("    let fr = tensor.blob_f32(\"{fr}\", [{}])", hd / 2));
    w(format!("    let x0 = tensor.embed({}, ids)", f32w("token_embd.weight")?));
    for i in 0..layers {
        let p = |s: &str| format!("blk.{i}.{s}");
        w(format!("    # block {i}"));
        w(format!("    let a{i} = tensor.rmsnorm(x{i}, {}, {eps})", f32w(&p("attn_norm.weight"))?));
        for (name, h, rope) in [("q", heads, true), ("k", kvh, true), ("v", kvh, false)] {
            let pr = product(&format!("a{i}"), &l.weight(&p(&format!("attn_{name}.weight")), None)?);
            let r = format!("tensor.reshape({pr}, [{ctx}, {h}, {hd}])");
            let r = if rope { format!("tensor.rope({r}, pos, fr)") } else { r };
            w(format!("    let {name}{i} = tensor.permute({r}, [1, 0, 2])"));
        }
        w(format!("    let o{i} = tensor.reshape(tensor.permute(tensor.attention_causal(q{i}, k{i}, v{i}), [1, 0, 2]), [{ctx}, {d}])"));
        w(format!("    let h{i} = tensor.add(x{i}, {})", product(&format!("o{i}"), &l.weight(&p("attn_output.weight"), None)?)));
        w(format!("    let f{i} = tensor.rmsnorm(h{i}, {}, {eps})", f32w(&p("ffn_norm.weight"))?));
        let gate = product(&format!("f{i}"), &l.weight(&p("ffn_gate.weight"), None)?);
        let up = product(&format!("f{i}"), &l.weight(&p("ffn_up.weight"), None)?);
        w(format!("    let g{i} = tensor.mul(tensor.silu({gate}), {up})"));
        w(format!("    let x{} = tensor.add(h{i}, {})", i + 1, product(&format!("g{i}"), &l.weight(&p("ffn_down.weight"), None)?)));
    }
    let out_w = if file.tensors.iter().any(|t| t.name == "output.weight") { "output.weight" } else { "token_embd.weight" };
    w(format!(
        "    logits = {}",
        product(&format!("tensor.rmsnorm(x{layers}, {}, {eps})", f32w("output_norm.weight")?), &l.weight(out_w, None)?)
    ));
    w("    seen += 1".into());
    Ok(out)
}

fn lift_gpt2(l: &Lift) -> Result<String, String> {
    let (file, store) = (l.file, l.store);
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
