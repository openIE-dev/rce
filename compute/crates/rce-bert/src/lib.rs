//! BERT encoder primitive (Wave-1 shared building block) + E5 sentence
//! embeddings. Post-norm Transformer encoder with WordPiece tokenization,
//! learned positions, GELU MLP, LayerNorm (eps 1e-12). Mean-pool + L2
//! norm gives E5 embeddings. First model: `intfloat/e5-small-v2`.

use rce_weight_loader::{LoadedTensor, SafetensorsFile};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::Path;

const D: usize = 384;
const N_LAYERS: usize = 12;
const N_HEADS: usize = 12;
const HD: usize = 32; // D / N_HEADS
const FF: usize = 1536;
const EPS: f32 = 1e-12;

#[inline] fn gelu(x: f32) -> f32 {
    // exact erf GELU (Abramowitz-Stegun erf, ~1.5e-7).
    let t = 1.0 / (1.0 + 0.327_591_1 * (x * std::f32::consts::FRAC_1_SQRT_2).abs());
    let z = x * std::f32::consts::FRAC_1_SQRT_2;
    let y = 1.0 - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6) * t * (-z * z).exp();
    let erf = if z < 0.0 { -y } else { y };
    0.5 * x * (1.0 + erf)
}

fn tf32(t: &LoadedTensor) -> Vec<f32> {
    t.as_f32().unwrap_or_else(|| panic!("tensor {} not f32", t.name)).to_vec()
}

fn linear(x: &[f32], t: usize, w: &[f32], dout: usize, din: usize, b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; t * dout];
    out.par_chunks_mut(dout).enumerate().for_each(|(i, orow)| {
        let xr = &x[i * din..(i + 1) * din];
        for o in 0..dout {
            let wr = &w[o * din..(o + 1) * din];
            let mut acc = b[o];
            for k in 0..din { acc += xr[k] * wr[k]; }
            orow[o] = acc;
        }
    });
    out
}

/// LayerNorm with weight+bias over last dim `d`, per row, in place.
fn layernorm(x: &mut [f32], d: usize, w: &[f32], b: &[f32]) {
    x.par_chunks_mut(d).for_each(|row| {
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| { let z = v - mean; z * z }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for ((v, g), bb) in row.iter_mut().zip(w).zip(b) { *v = (*v - mean) * inv * *g + *bb; }
    });
}

struct Layer {
    qw: Vec<f32>, qb: Vec<f32>, kw: Vec<f32>, kb: Vec<f32>, vw: Vec<f32>, vb: Vec<f32>,
    ao_w: Vec<f32>, ao_b: Vec<f32>, aln_w: Vec<f32>, aln_b: Vec<f32>,
    iw: Vec<f32>, ib: Vec<f32>, ow: Vec<f32>, ob: Vec<f32>, oln_w: Vec<f32>, oln_b: Vec<f32>,
}

pub struct Bert {
    word: Vec<f32>, pos: Vec<f32>, tok_type: Vec<f32>, emb_ln_w: Vec<f32>, emb_ln_b: Vec<f32>,
    layers: Vec<Layer>,
    vocab: HashMap<String, i32>,
    cls: i32, sep: i32, unk: i32,
}

impl Bert {
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, String> {
        let dir = dir.as_ref();
        let f = SafetensorsFile::load(dir.join("model.safetensors")).map_err(|e| format!("load: {e:?}"))?;
        let g = |n: &str| tf32(f.get(n).unwrap_or_else(|| panic!("missing {n}")));
        let layers = (0..N_LAYERS).map(|i| {
            let p = format!("encoder.layer.{i}");
            Layer {
                qw: g(&format!("{p}.attention.self.query.weight")), qb: g(&format!("{p}.attention.self.query.bias")),
                kw: g(&format!("{p}.attention.self.key.weight")), kb: g(&format!("{p}.attention.self.key.bias")),
                vw: g(&format!("{p}.attention.self.value.weight")), vb: g(&format!("{p}.attention.self.value.bias")),
                ao_w: g(&format!("{p}.attention.output.dense.weight")), ao_b: g(&format!("{p}.attention.output.dense.bias")),
                aln_w: g(&format!("{p}.attention.output.LayerNorm.weight")), aln_b: g(&format!("{p}.attention.output.LayerNorm.bias")),
                iw: g(&format!("{p}.intermediate.dense.weight")), ib: g(&format!("{p}.intermediate.dense.bias")),
                ow: g(&format!("{p}.output.dense.weight")), ob: g(&format!("{p}.output.dense.bias")),
                oln_w: g(&format!("{p}.output.LayerNorm.weight")), oln_b: g(&format!("{p}.output.LayerNorm.bias")),
            }
        }).collect();
        let (vocab, cls, sep, unk) = load_vocab(&dir.join("tokenizer.json"))?;
        Ok(Self {
            word: g("embeddings.word_embeddings.weight"), pos: g("embeddings.position_embeddings.weight"),
            tok_type: g("embeddings.token_type_embeddings.weight"),
            emb_ln_w: g("embeddings.LayerNorm.weight"), emb_ln_b: g("embeddings.LayerNorm.bias"),
            layers, vocab, cls, sep, unk,
        })
    }

    /// Tokenize → encode → mean-pool over tokens → L2-normalize.
    /// E5 expects `"query: ..."` / `"passage: ..."` prefixes (caller adds).
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let ids = self.tokenize(text);
        let t = ids.len();
        // embeddings = LN(word[id] + pos[i] + tok_type[0])
        let mut x = vec![0.0f32; t * D];
        for (i, &id) in ids.iter().enumerate() {
            let w = &self.word[id as usize * D..(id as usize + 1) * D];
            let p = &self.pos[i * D..(i + 1) * D];
            for d in 0..D { x[i * D + d] = w[d] + p[d] + self.tok_type[d]; }
        }
        layernorm(&mut x, D, &self.emb_ln_w, &self.emb_ln_b);

        let scale = 1.0 / (HD as f32).sqrt();
        for layer in &self.layers {
            // self-attention (bidirectional)
            let q = linear(&x, t, &layer.qw, D, D, &layer.qb);
            let k = linear(&x, t, &layer.kw, D, D, &layer.kb);
            let v = linear(&x, t, &layer.vw, D, D, &layer.vb);
            let mut ctx = vec![0.0f32; t * D];
            ctx.par_chunks_mut(D).enumerate().for_each(|(i, crow)| {
                for h in 0..N_HEADS {
                    let qh = &q[i * D + h * HD..i * D + h * HD + HD];
                    let mut sc = vec![0.0f32; t];
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..t {
                        let kj = &k[j * D + h * HD..j * D + h * HD + HD];
                        let mut s = 0.0f32; for d in 0..HD { s += qh[d] * kj[d]; }
                        s *= scale; sc[j] = s; if s > mx { mx = s; }
                    }
                    let mut sum = 0.0f32;
                    for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
                    let inv = 1.0 / sum;
                    let oh = &mut crow[h * HD..(h + 1) * HD];
                    for j in 0..t {
                        let wj = sc[j] * inv;
                        let vj = &v[j * D + h * HD..j * D + h * HD + HD];
                        for d in 0..HD { oh[d] += wj * vj[d]; }
                    }
                }
            });
            // attention output dense + residual + LN
            let mut ao = linear(&ctx, t, &layer.ao_w, D, D, &layer.ao_b);
            for i in 0..ao.len() { ao[i] += x[i]; }
            layernorm(&mut ao, D, &layer.aln_w, &layer.aln_b);
            // FFN + residual + LN
            let mut inter = linear(&ao, t, &layer.iw, FF, D, &layer.ib);
            for z in inter.iter_mut() { *z = gelu(*z); }
            let mut out = linear(&inter, t, &layer.ow, D, FF, &layer.ob);
            for i in 0..out.len() { out[i] += ao[i]; }
            layernorm(&mut out, D, &layer.oln_w, &layer.oln_b);
            x = out;
        }
        // mean-pool over tokens
        let mut emb = vec![0.0f32; D];
        for i in 0..t { for d in 0..D { emb[d] += x[i * D + d]; } }
        for d in 0..D { emb[d] /= t as f32; }
        // L2 normalize
        let n = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        for v in emb.iter_mut() { *v /= n; }
        emb
    }

    /// BERT WordPiece (uncased): lowercase, split on whitespace+punct,
    /// greedy longest-match subwords (`##` continuation), [CLS]…[SEP].
    fn tokenize(&self, text: &str) -> Vec<i32> {
        let mut ids = vec![self.cls];
        for word in basic_tokens(text) {
            let chars: Vec<char> = word.chars().collect();
            let mut start = 0;
            let mut pieces = Vec::new();
            let mut ok = true;
            while start < chars.len() {
                let mut end = chars.len();
                let mut found: Option<i32> = None;
                while end > start {
                    let sub: String = chars[start..end].iter().collect();
                    let cand = if start == 0 { sub } else { format!("##{sub}") };
                    if let Some(&id) = self.vocab.get(&cand) { found = Some(id); break; }
                    end -= 1;
                }
                match found { Some(id) => { pieces.push(id); start = end; } None => { ok = false; break; } }
            }
            if ok { ids.extend(pieces); } else { ids.push(self.unk); }
        }
        ids.push(self.sep);
        ids
    }
}

/// Lowercase, then split into word/punctuation tokens (whitespace splits,
/// each ASCII punct becomes its own token). Adequate for English.
fn basic_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.to_lowercase().chars() {
        if ch.is_whitespace() { if !cur.is_empty() { out.push(std::mem::take(&mut cur)); } }
        else if ch.is_ascii_punctuation() { if !cur.is_empty() { out.push(std::mem::take(&mut cur)); } out.push(ch.to_string()); }
        else { cur.push(ch); }
    }
    if !cur.is_empty() { out.push(cur); }
    out
}

fn load_vocab(path: &Path) -> Result<(HashMap<String, i32>, i32, i32, i32), String> {
    let txt = std::fs::read_to_string(path).map_err(|e| format!("tokenizer.json: {e}"))?;
    let json: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("parse: {e}"))?;
    let mut vocab = HashMap::new();
    if let Some(obj) = json.get("model").and_then(|m| m.get("vocab")).and_then(|v| v.as_object()) {
        for (tok, id) in obj { if let Some(i) = id.as_i64() { vocab.insert(tok.clone(), i as i32); } }
    }
    if vocab.is_empty() { return Err("empty vocab".into()); }
    let get = |k: &str| *vocab.get(k).unwrap_or(&0);
    Ok((vocab.clone(), get("[CLS]"), get("[SEP]"), get("[UNK]")))
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum() // both already L2-normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    fn model_dir() -> Option<String> {
        for b in ["macos_4TB_external", "macos_4TB_external_1", "macos_4TB_external_2"] {
            let d = format!("/Volumes/{b}/neuronexus_ai/model_library/intfloat_e5-small-v2");
            if std::path::Path::new(&d).join("model.safetensors").exists() { return Some(d); }
        }
        None
    }

    #[test]
    fn e5_embeddings_rank_correctly() {
        let Some(dir) = model_dir() else { eprintln!("e5: not mounted — skipping"); return };
        let m = Bert::load(&dir).expect("load");
        let q = m.embed("query: how much protein should a female eat");
        let related = m.embed("passage: The CDC recommends that adult women consume about 46 grams of protein per day, more if pregnant or athletic.");
        let unrelated = m.embed("passage: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.");
        let s_rel = cosine(&q, &related);
        let s_unrel = cosine(&q, &unrelated);
        eprintln!("e5 cosine: related={s_rel:.4}  unrelated={s_unrel:.4}");
        assert!(q.len() == D);
        assert!(s_rel > s_unrel + 0.05, "related ({s_rel:.4}) should clearly exceed unrelated ({s_unrel:.4})");
    }
}
