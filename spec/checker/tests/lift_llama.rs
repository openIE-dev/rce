//! `rce lift gguf` (10.17 LIFT-5) on a llama GGUF file written here: two
//! blocks of width 256, 4 query heads over 2 key and value heads (GQA),
//! RMS norm, the rotary embedding and SwiGLU, with weights from a fixed
//! seed in Q4_K, Q5_K, Q6_K, Q8_0 and F16. Nothing is downloaded. The
//! lifted program is admitted, its test and trace are the same bytes on
//! every run, its logits are a forward pass written here in Rust bit for
//! bit, the values it holds for every K-quant weight are those
//! `rce-ingest-gguf` reads from the file, exactly, and each K-quant product
//! is judged against `rce-op-qmm`.
use rce_ingest_gguf::{GgufDtype, GgufFile};

fn code(d: GgufDtype) -> u32 {
    match d {
        GgufDtype::Q4K => 12,
        GgufDtype::Q5K => 13,
        GgufDtype::Q6K => 14,
        GgufDtype::Q8_0 => 8,
        _ => 1,
    }
}
use rce_spec_check::cli;
use rce_spec_check::lift_gguf::{f16_to_f32, rope_freqs};
use rce_vm_backend_cpu::ml::kquant::{self, KFmt};
use rce_vm_backend_cpu::ml::nn;
use rce_vm_backend_cpu::{HostTensor, ScalarValue, Value};
use std::path::PathBuf;

#[path = "support/toy_llama.rs"]
mod toy_llama;
use toy_llama::*;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let st = cli::main(&a, &mut std::io::empty(), &mut out, &mut err);
    (st, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn lifted(name: &str) -> (PathBuf, String) {
    let dir = std::env::temp_dir().join(format!("rce-lift-llama-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let g = dir.join("toy.gguf");
    std::fs::write(&g, toy_gguf()).unwrap();
    let (st, out, err) = rce(&["lift", "gguf", g.to_str().unwrap(), dir.join("blobs").to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    let p = dir.join("toy.rce");
    std::fs::write(&p, out).unwrap();
    (dir, p.to_str().unwrap().to_string())
}

const IDS: [[i32; CTX]; 2] = [[3, 17, 5, 0, 31, 9, 9, 22], [1, 2, 3, 4, 5, 6, 7, 8]];

fn t(shape: Vec<usize>, v: &[f32]) -> Value {
    Value::Tensor(HostTensor::from_f32(shape, v).unwrap())
}
fn vals(v: Value) -> Vec<f32> {
    match v {
        Value::Tensor(h) => h.as_f32().unwrap(),
        _ => panic!("not a tensor"),
    }
}
fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN.wrapping_sub(i) as i64 } else { i as i64 }
    };
    (key(a) - key(b)).unsigned_abs()
}

struct Model {
    f: GgufFile,
    bytes: Vec<u8>,
}

#[derive(Default, Debug)]
struct Judged(std::collections::BTreeMap<&'static str, (u64, usize)>);

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
    /// `x W^T` by the reference kernel of W's format, judged (K-quants)
    /// against rce-op-qmm's serial and dispatched kernels.
    fn product(&self, name: &str, x: &[f32], j: &mut Judged) -> Vec<f32> {
        let (t, b) = self.data(name);
        let (k, n) = (t.dims[0] as usize, t.dims[1] as usize);
        let m = x.len() / k;
        let kf = match code(t.dtype) {
            12 => Some(KFmt::Q4K),
            13 => Some(KFmt::Q5K),
            14 => Some(KFmt::Q6K),
            _ => None,
        };
        if let Some(f) = kf {
            let (mut qs, mut ss, mut os) = (vec![], vec![], vec![]);
            for blk in b.chunks_exact(f.block_bytes()) {
                let kb = kquant::unpack(blk, f, f16_to_f32);
                let (s, o) = kb.scales();
                qs.extend(kb.q.clone());
                ss.extend(s);
                os.extend(o);
            }
            let ns = k / f.sub();
            let mut args = vec![t_(vec![m, k], x), Value::Tensor(HostTensor::from_i32(vec![n, k], &qs).unwrap()), t_(vec![n, ns], &ss)];
            if f != KFmt::Q6K {
                args.push(t_(vec![n, ns], &os));
            }
            let r = vals(kquant::matmul_k(0, &args, f).unwrap());
            let w = rce_op_qmm::QuantWeight::from_owned_bytes(b.to_vec(), t.dtype, n, k);
            let (mut ser, mut dis) = (vec![], vec![]);
            for row in x.chunks(k) {
                let mut o = vec![0f32; n];
                rce_op_qmm::mm_1d_serial(row, &w, &mut o);
                ser.extend_from_slice(&o);
                let mut o = vec![0f32; n];
                rce_op_qmm::mm_1d(row, &w, &mut o);
                dis.extend_from_slice(&o);
            }
            let name = f.names().0;
            for (label, kern) in [("serial", &ser), ("dispatched", &dis)] {
                let key: &'static str = Box::leak(format!("{name} {label}").into_boxed_str());
                let e = j.0.entry(key).or_default();
                for (a, c) in r.iter().zip(kern.iter()) {
                    e.0 = e.0.max(ulps(*a, *c));
                    e.1 += 1;
                }
            }
            r
        } else if code(t.dtype) == 8 {
            let mut qs = vec![];
            let mut ds = vec![];
            for blk in b.chunks_exact(34) {
                ds.push(f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])));
                qs.extend(blk[2..34].iter().map(|v| *v as i8 as i32));
            }
            let args = [t_(vec![m, k], x), Value::Tensor(HostTensor::from_i32(vec![n, k], &qs).unwrap()), t_(vec![n, k / 32], &ds)];
            vals(nn::matmul_q(0, &args, false).unwrap())
        } else {
            // F16: the f32 matrix product of x and W^T, k increasing from +0.0
            let w = self.floats(name);
            let mut o = vec![0f32; m * n];
            for i in 0..m {
                for c in 0..n {
                    let mut s = 0.0f32;
                    for p in 0..k {
                        s += x[i * k + p] * w[c * k + p];
                    }
                    o[i * n + c] = s;
                }
            }
            o
        }
    }
    fn rmsnorm(&self, x: &[f32], w: &str) -> Vec<f32> {
        let w = self.floats(w);
        let rows = x.len() / D;
        let args = [t_(vec![rows, D], x), t_(vec![D], &w), Value::Scalar(ScalarValue::I32(rows as i32)), Value::Scalar(ScalarValue::F32(EPS))];
        vals(nn::rms_norm(0, &args).unwrap())
    }
}

fn t_(shape: Vec<usize>, v: &[f32]) -> Value {
    t(shape, v)
}

/// The rotary embedding of adjacent pairs, written out: [CTX, h, HD].
fn rope(x: &[f32], h: usize, fr: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    for s in 0..CTX {
        for hh in 0..h {
            for (i, f) in fr.iter().enumerate() {
                let a = s as f32 * *f;
                let (c, n) = (nn::e_cos(a), nn::e_sin(a));
                let at = (s * h + hh) * HD + 2 * i;
                let (x0, x1) = (x[at], x[at + 1]);
                y[at] = x0 * c - x1 * n;
                y[at + 1] = x0 * n + x1 * c;
            }
        }
    }
    y
}

/// [CTX, h, HD] to heads first [h, CTX, HD], and back.
fn heads_first(v: &[f32], h: usize) -> Vec<f32> {
    (0..h).flat_map(|hh| (0..CTX).flat_map(move |s| v[(s * h + hh) * HD..(s * h + hh + 1) * HD].to_vec())).collect()
}

fn forward(m: &Model, ids: &[i32], j: &mut Judged) -> Vec<f32> {
    let tok = m.floats("token_embd.weight");
    let mut x: Vec<f32> = ids.iter().flat_map(|i| tok[*i as usize * D..(*i as usize + 1) * D].to_vec()).collect();
    let fr = rope_freqs(BASE, HD);
    for i in 0..LAYERS {
        let p = |s: &str| format!("blk.{i}.{s}");
        let a = m.rmsnorm(&x, &p("attn_norm.weight"));
        let q = heads_first(&rope(&m.product(&p("attn_q.weight"), &a, j), HEADS, &fr), HEADS);
        let k = heads_first(&rope(&m.product(&p("attn_k.weight"), &a, j), KV_HEADS, &fr), KV_HEADS);
        let v = heads_first(&m.product(&p("attn_v.weight"), &a, j), KV_HEADS);
        let scale = (1.0 / (HD as f64).sqrt()) as f32;
        let args = [t(vec![HEADS, CTX, HD], &q), t(vec![KV_HEADS, CTX, HD], &k), t(vec![KV_HEADS, CTX, HD], &v), Value::Scalar(ScalarValue::F32(scale)), Value::Scalar(ScalarValue::I32(0)), Value::Scalar(ScalarValue::Bool(true))];
        let att = vals(nn::attention(0, &args).unwrap());
        let mut o = vec![];
        for s in 0..CTX {
            for h in 0..HEADS {
                o.extend_from_slice(&att[(h * CTX + s) * HD..(h * CTX + s + 1) * HD]);
            }
        }
        let h: Vec<f32> = x.iter().zip(m.product(&p("attn_output.weight"), &o, j)).map(|(a, b)| a + b).collect();
        let f = m.rmsnorm(&h, &p("ffn_norm.weight"));
        let g: Vec<f32> = m.product(&p("ffn_gate.weight"), &f, j).iter().zip(m.product(&p("ffn_up.weight"), &f, j)).map(|(a, b)| nn::silu(*a) * b).collect();
        x = h.iter().zip(m.product(&p("ffn_down.weight"), &g, j)).map(|(a, b)| a + b).collect();
    }
    let n = m.rmsnorm(&x, "output_norm.weight");
    m.product("output.weight", &n, j)
}

fn run_logits(out: &str) -> Vec<f32> {
    let line = out.lines().last().unwrap();
    let at = line.find("logits = ").unwrap() + "logits = ".len();
    let end = line[at..].find("]], ").unwrap() + at + 2;
    line[at..end].replace(['[', ']'], "").split(", ").map(|v| v.parse::<f64>().unwrap() as f32).collect()
}

#[test]
fn the_lifted_llama_is_admitted_and_its_runs_are_the_same_bytes() {
    let (_dir, p) = lifted("admit");
    let (st, out, _) = rce(&["check", &p]);
    assert_eq!(st, 0, "{out}");
    let src = std::fs::read_to_string(&p).unwrap();
    for op in ["tensor.rope(", "tensor.rmsnorm(", "tensor.silu(", "tensor.matmul_q4_k(", "tensor.matmul_q5_k(", "tensor.matmul_q6_k(", "tensor.matmul_q8_0("] {
        assert!(src.contains(op), "{op}");
    }
    let tests: Vec<(i32, String, String)> = (0..2).map(|_| rce(&["test", &p])).collect();
    assert_eq!(tests[0].0, 0, "{}", tests[0].1);
    assert_eq!(tests[0], tests[1]);
    let ev = IDS.iter().map(|ids| format!("Tokens(ids: {ids:?})")).collect::<Vec<_>>().join(", ");
    let traces: Vec<(i32, String, String)> = (0..2).map(|_| rce(&["trace", &p, &ev])).collect();
    assert_eq!(traces[0].0, 0, "{}", traces[0].2);
    assert_eq!(traces[0], traces[1]);
}

#[test]
fn the_values_held_for_each_k_quant_weight_are_the_files_exactly() {
    let bytes = toy_gguf();
    let m = Model { f: GgufFile::parse(&bytes).unwrap(), bytes: bytes.clone() };
    let mut seen = 0;
    for t in &m.f.tensors {
        let f = match code(t.dtype) {
            12 => KFmt::Q4K,
            13 => KFmt::Q5K,
            14 => KFmt::Q6K,
            _ => continue,
        };
        let (_, b) = m.data(&t.name);
        let held: Vec<f32> = b.chunks_exact(f.block_bytes()).flat_map(|blk| kquant::unpack(blk, f, f16_to_f32).values(f)).collect();
        let file = m.floats(&t.name);
        assert!(held.iter().zip(&file).all(|(a, c)| a.to_bits() == c.to_bits()), "{}", t.name);
        seen += 1;
    }
    assert!(seen >= 8, "{seen}");
}

#[test]
fn the_lifted_llama_is_the_rust_forward_pass_bit_for_bit() {
    let (_dir, p) = lifted("judge");
    let bytes = toy_gguf();
    let m = Model { f: GgufFile::parse(&bytes).unwrap(), bytes: bytes.clone() };
    let mut j = Judged::default();
    for ids in IDS {
        let reference = forward(&m, &ids, &mut j);
        let (st, out, err) = rce(&["run", &p, &format!("Tokens(ids: {ids:?})")]);
        assert_eq!(st, 0, "{err}");
        let got = run_logits(&out);
        assert_eq!(got.len(), CTX * VOCAB);
        assert!(got.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits()), "the lifted program's logits are the forward pass");
    }
    eprintln!("ulps from rce-op-qmm, by operation (max, elements): {:?}", j.0);
    // the serial row kernels of rce-op-qmm sum in the reference's order
    for op in ["matmul_q4_k serial", "matmul_q5_k serial", "matmul_q6_k serial"] {
        assert_eq!(j.0[op].0, 0, "{op}: {:?}", j.0[op]);
        assert!(j.0[op].1 >= 256, "{op}");
    }
}
