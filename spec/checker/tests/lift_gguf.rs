//! `rce lift gguf` (10.17) on a GGUF file written here: a two-block gpt2
//! model of width 32 with weights from a fixed seed, its products in Q8_0
//! and Q4_0. The lifted program is admitted, its test and trace are the
//! same bytes on every run, D records the blobs it loaded, and each of its
//! operations is judged against the kernels RCE's host adapters run
//! (`rce-ingest-gguf`, `rce-op-qmm`, `rce-op-nn-prims`, `rce-op-attention`),
//! fed the reference's own inputs, at the operation's level.
use rce_ingest_gguf::parser::{build_gguf, encode_f32_value, encode_string_value, encode_u32_value};
use rce_ingest_gguf::{GgufDtype, GgufFile};
use rce_spec_check::host::{Host, MemStore, replay_report};
use rce_spec_check::{cli, graph, parse::parse};
use rce_vm_backend_cpu::ml::nn;
use rce_vm_backend_cpu::{HostTensor, OpDispatch, ScalarValue, Value};
use std::path::PathBuf;

const D: usize = 32;
const HEADS: usize = 2;
const HD: usize = D / HEADS;
const FF: usize = 64;
const VOCAB: usize = 32;
const CTX: usize = 8;
const LAYERS: usize = 2;
const EPS: f32 = 1e-5;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let st = cli::main(&a, &mut std::io::empty(), &mut out, &mut err);
    (st, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-lift-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Uniform values in [-a, a) from a fixed linear congruential sequence.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, a: f32) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
    }
    fn vec(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n).map(|_| self.next(a)).collect()
    }
}

/// The binary16 bits of an f32 that is a binary16 value.
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let a = x.abs();
    if a == 0.0 {
        return s;
    }
    if a < 2f32.powi(-14) {
        return s | (a / 2f32.powi(-24)) as u16;
    }
    let e = ((a.to_bits() >> 23) as i32) - 127 + 15;
    s | ((e as u16) << 10) | ((a.to_bits() >> 13) & 0x3ff) as u16
}

/// GGUF bytes of an n x k matrix (rows of k) in Q8_0 or Q4_0, quantized by
/// the reference quantizer of the block formats.
fn quantize(w: &[f32], q4: bool) -> Vec<u8> {
    let mut out = vec![];
    for blk in w.chunks(32) {
        let (q, d) = nn::quantize_block(blk, q4);
        out.extend_from_slice(&f16_bits(d).to_le_bytes());
        if q4 {
            out.extend((0..16).map(|t| ((q[t] + 8) as u8) | (((q[t + 16] + 8) as u8) << 4)));
        } else {
            out.extend(q.iter().map(|v| *v as i8 as u8));
        }
    }
    out
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The toy model's GGUF file.
fn toy_gguf() -> Vec<u8> {
    let mut r = Lcg(0x5eed_0001);
    let mut tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)> = vec![];
    let mut f32t = |name: &str, dims: Vec<u64>, v: Vec<f32>| tensors.push((name.into(), dims, 0, f32_bytes(&v)));
    f32t("token_embd.weight", vec![D as u64, VOCAB as u64], r.vec(D * VOCAB, 1.0));
    // the position table in F16: its values are binary16 values
    let pos: Vec<f32> = r.vec(D * CTX, 0.5).iter().map(|x| nn::round_to_f16(*x)).collect();
    let pos16: Vec<u8> = pos.iter().flat_map(|x| f16_bits(*x).to_le_bytes()).collect();
    tensors.push(("position_embd.weight".into(), vec![D as u64, CTX as u64], 1, pos16));
    for i in 0..LAYERS {
        let q4 = i == 1;
        let qt = if q4 { 2 } else { 8 };
        let mut quant = |name: String, k: usize, n: usize, q4: bool, r: &mut Lcg| {
            let w = r.vec(n * k, 0.3);
            tensors.push((name, vec![k as u64, n as u64], if q4 { 2 } else { 8 }, quantize(&w, q4)));
        };
        quant(format!("blk.{i}.attn_qkv.weight"), D, 3 * D, q4, &mut r);
        quant(format!("blk.{i}.attn_output.weight"), D, D, false, &mut r);
        quant(format!("blk.{i}.ffn_up.weight"), D, FF, q4, &mut r);
        quant(format!("blk.{i}.ffn_down.weight"), FF, D, q4, &mut r);
        let _ = qt;
        for (name, n, a, off) in [("attn_norm.weight", D, 0.2, 1.0), ("attn_norm.bias", D, 0.1, 0.0), ("attn_qkv.bias", 3 * D, 0.1, 0.0), ("attn_output.bias", D, 0.1, 0.0), ("ffn_norm.weight", D, 0.2, 1.0), ("ffn_norm.bias", D, 0.1, 0.0), ("ffn_up.bias", FF, 0.1, 0.0), ("ffn_down.bias", D, 0.1, 0.0)] {
            let v: Vec<f32> = r.vec(n, a).iter().map(|x| x + off).collect();
            tensors.push((format!("blk.{i}.{name}"), vec![n as u64], 0, f32_bytes(&v)));
        }
    }
    let v: Vec<f32> = r.vec(D, 0.2).iter().map(|x| x + 1.0).collect();
    tensors.push(("output_norm.weight".into(), vec![D as u64], 0, f32_bytes(&v)));
    tensors.push(("output_norm.bias".into(), vec![D as u64], 0, f32_bytes(&r.vec(D, 0.1))));
    let w = r.vec(VOCAB * D, 0.3);
    tensors.push(("output.weight".into(), vec![D as u64, VOCAB as u64], 8, quantize(&w, false)));
    let meta: Vec<(&str, u32, Vec<u8>)> = vec![
        ("general.architecture", 8, encode_string_value("gpt2")),
        ("gpt2.context_length", 4, encode_u32_value(CTX as u32)),
        ("gpt2.embedding_length", 4, encode_u32_value(D as u32)),
        ("gpt2.feed_forward_length", 4, encode_u32_value(FF as u32)),
        ("gpt2.block_count", 4, encode_u32_value(LAYERS as u32)),
        ("gpt2.attention.head_count", 4, encode_u32_value(HEADS as u32)),
        ("gpt2.attention.layer_norm_epsilon", 6, encode_f32_value(EPS)),
    ];
    let m: Vec<(&str, u32, &[u8])> = meta.iter().map(|(k, t, v)| (*k, *t, v.as_slice())).collect();
    let t: Vec<(&str, &[u64], u32, &[u8])> = tensors.iter().map(|(n, d, ty, b)| (n.as_str(), d.as_slice(), *ty, b.as_slice())).collect();
    build_gguf(&m, &t)
}

/// The lifted program and its store, in a fresh directory: (dir, program path).
fn lifted(name: &str) -> (PathBuf, String) {
    let dir = tmp(name);
    let g = dir.join("toy.gguf");
    std::fs::write(&g, toy_gguf()).unwrap();
    let (st, out, err) = rce(&["lift", "gguf", g.to_str().unwrap(), dir.join("blobs").to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    let p = dir.join("toy.rce");
    std::fs::write(&p, out).unwrap();
    (dir, p.to_str().unwrap().to_string())
}

const IDS: [[i32; CTX]; 2] = [[3, 17, 5, 0, 31, 9, 9, 22], [1, 2, 3, 4, 5, 6, 7, 8]];

fn events() -> String {
    IDS.iter().map(|ids| format!("Tokens(ids: {ids:?})")).collect::<Vec<_>>().join(", ")
}

#[test]
fn the_lifted_program_is_admitted_and_its_runs_are_the_same_bytes() {
    let (dir, p) = lifted("admit");
    let (st, out, _) = rce(&["check", &p]);
    assert_eq!(st, 0, "{out}");
    let tests: Vec<(i32, String, String)> = (0..2).map(|_| rce(&["test", &p])).collect();
    assert_eq!(tests[0].0, 0, "{}", tests[0].1);
    assert!(tests[0].1.contains("(1 examples, 0 checks)"), "{}", tests[0].1);
    assert_eq!(tests[0], tests[1], "rce test differs between runs");
    let ev = events();
    let traces: Vec<(i32, String, String)> = (0..2).map(|_| rce(&["trace", &p, &ev])).collect();
    assert_eq!(traces[0].0, 0, "{}", traces[0].2);
    assert_eq!(traces[0], traces[1], "rce trace differs between runs");
    // the program names its weights by digest only: no path, no file name
    let src = std::fs::read_to_string(&p).unwrap();
    assert!(!src.contains("toy.gguf") && !src.contains("blobs/"));
    let m = parse(&src).unwrap();
    let ds = graph::blob_digests(&m);
    assert_eq!(ds.len(), std::fs::read_dir(dir.join("blobs")).unwrap().count(), "every blob of the store is named, and only those");
    // D records the digests the host loaded, before any input; replay
    // re-derives every state
    let mut st = MemStore::default();
    let mut h = Host::start(&m, Box::new(&mut st)).unwrap();
    for ids in IDS {
        h.admit(&src, &format!("Tokens(ids: {ids:?})")).unwrap();
    }
    let fin = h.state_hash().to_string();
    drop(h);
    let blobs: Vec<&String> = st.lines.iter().filter(|l| l.contains("\"host.Blobs\"")).collect();
    assert_eq!(blobs.len(), 1);
    for d in &ds {
        assert!(blobs[0].contains(d.as_str()), "D names {d}");
    }
    let d: String = st.lines.iter().map(|l| format!("{l}\n")).collect();
    assert_eq!(replay_report(&m, &d).unwrap().state_hash, fin);
    // an architecture that is not lifted is refused with its name (LIFT-1)
    let arch = encode_string_value("llama");
    let other = build_gguf(&[("general.architecture", 8, arch.as_slice())], &[]);
    let g = dir.join("other.gguf");
    std::fs::write(&g, other).unwrap();
    let (st, _, err) = rce(&["lift", "gguf", g.to_str().unwrap(), dir.join("other").to_str().unwrap()]);
    assert_eq!(st, 2);
    assert!(err.contains("the architecture `llama` is not lifted; `gpt2` is"), "{err}");
}

/// Units in the last place between two f32 values.
fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN.wrapping_sub(i) as i64 } else { i as i64 }
    };
    (key(a) - key(b)).unsigned_abs()
}

fn t(shape: Vec<usize>, v: &[f32]) -> Value {
    Value::Tensor(HostTensor::from_f32(shape, v).unwrap())
}

fn vals(v: Value) -> Vec<f32> {
    match v {
        Value::Tensor(h) => h.as_f32().unwrap(),
        _ => panic!("not a tensor"),
    }
}

fn sc(x: ScalarValue) -> Value {
    Value::Scalar(x)
}

/// The largest distance in ulps each operation of the forward pass had
/// from the kernel of a host adapter, by operation.
#[derive(Default, Debug)]
struct Judged(std::collections::BTreeMap<&'static str, (u64, usize)>);
impl Judged {
    fn judge(&mut self, op: &'static str, reference: &[f32], kernel: &[f32]) {
        assert_eq!(reference.len(), kernel.len(), "{op}");
        let e = self.0.entry(op).or_default();
        for (a, b) in reference.iter().zip(kernel) {
            e.0 = e.0.max(ulps(*a, *b));
            e.1 += 1;
        }
    }
}

struct Model {
    f: GgufFile,
    bytes: Vec<u8>,
}

impl Model {
    fn data(&self, name: &str) -> (&rce_ingest_gguf::GgufTensorInfo, &[u8]) {
        let t = self.f.tensors.iter().find(|t| t.name == name).unwrap();
        let s = (self.f.data_offset + t.offset) as usize;
        (t, &self.bytes[s..s + rce_ingest_gguf::tensor_byte_size(t)])
    }
    fn floats(&self, name: &str) -> Vec<f32> {
        let (t, b) = self.data(name);
        rce_ingest_gguf::dequantize_to_f32(b, t.dtype, t.numel() as usize).unwrap()
    }
    /// The reference's `x W^T` for rows `rows` of W, and the same by
    /// rce-op-qmm's kernel on the file's bytes.
    fn product(&self, name: &str, x: &[f32], rows: std::ops::Range<usize>, j: &mut Judged) -> Vec<f32> {
        let (t, b) = self.data(name);
        let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
        let q4 = t.dtype == GgufDtype::Q4_0;
        let bb = if q4 { 18 } else { 34 };
        let (mut qs, mut ds) = (vec![], vec![]);
        for blk in b.chunks_exact(bb).skip(rows.start * k / 32).take(rows.len() * k / 32) {
            ds.push(rce_spec_check::lift_gguf::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
            if q4 {
                qs.extend(blk[2..18].iter().map(|v| (v & 15) as i32 - 8));
                qs.extend(blk[2..18].iter().map(|v| (v >> 4) as i32 - 8));
            } else {
                qs.extend(blk[2..34].iter().map(|v| *v as i8 as i32));
            }
        }
        let m = x.len() / k;
        let args = [
            t_(vec![m, k], x),
            Value::Tensor(HostTensor::from_i32(vec![rows.len(), k], &qs).unwrap()),
            t_(vec![rows.len(), k / 32], &ds),
        ];
        let r = vals(nn::matmul_q(0, &args, q4).unwrap());
        let w = rce_op_qmm::QuantWeight::from_owned_bytes(b.to_vec(), t.dtype, n, k);
        let mut kern = vec![];
        for row in x.chunks(k) {
            let mut o = vec![0f32; n];
            rce_op_qmm::mm_1d(row, &w, &mut o);
            kern.extend_from_slice(&o[rows.clone()]);
        }
        j.judge(if q4 { "matmul_q4_0" } else { "matmul_q8_0" }, &r, &kern);
        r
    }
    fn layernorm(&self, x: &[f32], w: &str, b: &str, j: &mut Judged) -> Vec<f32> {
        let (w, b) = (self.floats(w), self.floats(b));
        let rows = x.len() / D;
        let args = [t_(vec![rows, D], x), t_(vec![D], &w), t_(vec![D], &b), sc(ScalarValue::I32(rows as i32)), sc(ScalarValue::F32(EPS))];
        let r = vals(nn::layer_norm(0, &args).unwrap());
        let tile = |v: &[f32]| -> Vec<f32> { v.iter().cycle().take(x.len()).copied().collect() };
        let kargs = [t_(vec![rows, D], x), t_(vec![rows * D], &tile(&w)), t_(vec![rows * D], &tile(&b)), sc(ScalarValue::I32(rows as i32)), sc(ScalarValue::F32(EPS))];
        let kern = vals(rce_op_nn_prims::LayerNormLeaf.execute(0, "layer_norm", &kargs).unwrap());
        j.judge("layer_norm", &r, &kern);
        r
    }
}

fn t_(shape: Vec<usize>, v: &[f32]) -> Value {
    t(shape, v)
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter().cycle()).map(|(x, y)| x + y).collect()
}

/// The forward pass of the toy model on `ids`, by the reference kernels in
/// the order of the lifted program, each operation judged against the
/// kernel of a host adapter on the same inputs.
fn forward(m: &Model, ids: &[i32], j: &mut Judged) -> Vec<f32> {
    // embeddings: the reference's lookup against rce-ingest-gguf's rows
    let tok = m.floats("token_embd.weight");
    let pos = m.floats("position_embd.weight");
    let look = |table: &[f32], ids: &[i32]| -> Vec<f32> {
        let a = [t(vec![table.len() / D, D], table), Value::Tensor(HostTensor::from_i32(vec![ids.len()], ids).unwrap())];
        vals(nn::embedding_lookup(0, &a).unwrap())
    };
    let pos_ids: Vec<i32> = (0..CTX as i32).collect();
    let (te, pe) = (look(&tok, ids), look(&pos, &pos_ids));
    let rows = |table: &[f32], ids: &[i32]| -> Vec<f32> { ids.iter().flat_map(|i| table[*i as usize * D..(*i as usize + 1) * D].to_vec()).collect() };
    j.judge("embedding_lookup", &te, &rows(&tok, ids));
    j.judge("embedding_lookup", &pe, &rows(&pos, &pos_ids));
    let mut x = add(&te, &pe);
    for i in 0..LAYERS {
        let p = |s: &str| format!("blk.{i}.{s}");
        let a = m.layernorm(&x, &p("attn_norm.weight"), &p("attn_norm.bias"), j);
        let qkv_b = m.floats(&p("attn_qkv.bias"));
        // q, k, v: [CTX, D] -> heads first [HEADS, CTX, HD]
        let heads = |v: &[f32]| -> Vec<f32> { (0..HEADS).flat_map(|h| (0..CTX).flat_map(move |s| v[s * D + h * HD..s * D + (h + 1) * HD].to_vec())).collect() };
        let proj: Vec<Vec<f32>> = (0..3).map(|c| heads(&add(&m.product(&p("attn_qkv.weight"), &a, c * D..(c + 1) * D, j), &qkv_b[c * D..(c + 1) * D]))).collect();
        let scale = (1.0 / (HD as f64).sqrt()) as f32;
        let args = [t(vec![HEADS, CTX, HD], &proj[0]), t(vec![HEADS, CTX, HD], &proj[1]), t(vec![HEADS, CTX, HD], &proj[2]), sc(ScalarValue::F32(scale)), sc(ScalarValue::I32(0)), sc(ScalarValue::Bool(true))];
        let att = vals(nn::attention(0, &args).unwrap());
        // rce-op-attention: one head, one query row over the keys 0..=row
        let mut kern = vec![];
        for h in 0..HEADS {
            for s in 0..CTX {
                let at = |v: &[f32], r: std::ops::Range<usize>| v[(h * CTX + r.start) * HD..(h * CTX + r.end) * HD].to_vec();
                let a = [t(vec![1, HD], &at(&proj[0], s..s + 1)), t(vec![s + 1, HD], &at(&proj[1], 0..s + 1)), t(vec![s + 1, HD], &at(&proj[2], 0..s + 1)), sc(ScalarValue::F32(scale)), sc(ScalarValue::I32(0))];
                kern.extend(vals(rce_op_attention::AttentionLeaf.execute(0, "attention", &a).unwrap()));
            }
        }
        j.judge("attention", &att, &kern);
        let mut o = vec![];
        for s in 0..CTX {
            for h in 0..HEADS {
                o.extend_from_slice(&att[(h * CTX + s) * HD..(h * CTX + s + 1) * HD]);
            }
        }
        let h = add(&x, &add(&m.product(&p("attn_output.weight"), &o, 0..D, j), &m.floats(&p("attn_output.bias"))));
        let f = m.layernorm(&h, &p("ffn_norm.weight"), &p("ffn_norm.bias"), j);
        let up = add(&m.product(&p("ffn_up.weight"), &f, 0..FF, j), &m.floats(&p("ffn_up.bias")));
        let g = vals(nn::unary(0, "gelu", &[t(vec![CTX, FF], &up)]).unwrap());
        let kern = vals(rce_op_nn_prims::GeluLeaf.execute(0, "gelu", &[t(vec![CTX, FF], &up)]).unwrap());
        j.judge("gelu", &g, &kern);
        x = add(&h, &add(&m.product(&p("ffn_down.weight"), &g, 0..D, j), &m.floats(&p("ffn_down.bias"))));
    }
    let n = m.layernorm(&x, "output_norm.weight", "output_norm.bias", j);
    m.product("output.weight", &n, 0..VOCAB, j)
}

/// The logits of the state `rce run` prints, as f32 values.
fn run_logits(out: &str) -> Vec<f32> {
    let line = out.lines().last().unwrap();
    let at = line.find("logits = ").unwrap() + "logits = ".len();
    let end = line[at..].find("]], ").unwrap() + at + 2;
    line[at..end].replace(['[', ']'], "").split(", ").map(|v| v.parse::<f64>().unwrap() as f32).collect()
}

#[test]
fn each_operation_equals_the_kernels_of_the_host_adapters_at_its_level() {
    let (_dir, p) = lifted("judge");
    let bytes = toy_gguf();
    let m = Model { f: GgufFile::parse(&bytes).unwrap(), bytes: bytes.clone() };
    let mut j = Judged::default();
    for ids in IDS {
        let reference = forward(&m, &ids, &mut j);
        // the forward pass here is the lifted program's meaning, bit for bit
        let (st, out, err) = rce(&["run", &p, &format!("Tokens(ids: {ids:?})")]);
        assert_eq!(st, 0, "{err}");
        let got = run_logits(&out);
        assert_eq!(got.len(), CTX * VOCAB);
        assert!(got.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()), "the lifted program's logits are the reference forward pass");
    }
    eprintln!("ulps from the host adapters' kernels, by operation (max, elements): {:?}", j.0);
    // every operation of the lifted program is exact (GT-4); the kernels
    // that compute with RCE's own arithmetic reproduce the reference's bits
    // on every element, so each is a realisation at that level
    for op in ["embedding_lookup", "layer_norm", "matmul_q8_0", "matmul_q4_0"] {
        assert_eq!(j.0[op].0, 0, "{op}: {:?}", j.0[op]);
        assert!(j.0[op].1 >= 1024, "{op}");
    }
    // rce-op-attention and rce-op-nn-prims' gelu take exp and tanh from the
    // platform: they are judged element by element and are realisations at
    // `exact` only where they reproduce every bit; their distance is the N
    // a program would declare with `tensor.within(N, ...)` to admit them
    // (10.17 LIFT-4 states the values measured on this model)
    for op in ["attention", "gelu"] {
        assert!(j.0[op].1 >= 1024, "{op}");
    }
}
