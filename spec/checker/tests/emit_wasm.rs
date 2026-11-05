//! `rce emit wasm` and `rce graph run wasm` (spec 10.18, EW-1 to EW-6).
//!
//! Each graph term of `compute/examples/dense.rce`, `block.rce` and the
//! lifted toy gpt2 (`support/toy_gpt2.rs`) runs through its program module
//! and the kernel library under wasmi, on operands from a fixed seed, and
//! every node is judged against the reference evaluation's value of that
//! node at the node's level (exact: bit for bit; within N: N units in the
//! last place; lossy: its type).
use rce_graph::term::Level;
use rce_spec_check::{cli, graph, wasm_term};
use rce_vm_backend_cpu::{DType, HostTensor};
use std::path::PathBuf;

#[path = "support/toy_gpt2.rs"]
mod toy_gpt2;
use toy_gpt2::*;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let st = cli::main(&a, &mut std::io::empty(), &mut out, &mut err);
    (st, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rce-emit-wasm-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn example(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../compute/examples").join(name)
}

fn toy(name: &str) -> PathBuf {
    let dir = tmp(name);
    let g = dir.join("toy.gguf");
    std::fs::write(&g, toy_gguf()).unwrap();
    let (st, out, err) = rce(&["lift", "gguf", g.to_str().unwrap(), dir.join("blobs").to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    let p = dir.join("toy.rce");
    std::fs::write(&p, out).unwrap();
    graph::add_store(&dir.join("blobs"));
    p
}

/// Operands of the term's parameter types: floats in [-1, 1), integers in
/// 0..8 (an index of every table of the programs here).
fn operands(t: &graph::Term, seed: u64) -> Vec<rce_spec_check::interp::V> {
    let mut r = Lcg(seed);
    t.types
        .params
        .iter()
        .map(|p| {
            let tt = graph::TTy::of_spec(p).unwrap();
            let n = tt.numel().max(1) as usize;
            let json = if p.dtype.starts_with('f') {
                serde_json::json!(r.vec(n, 1.0).iter().map(|x| *x as f64).collect::<Vec<_>>())
            } else {
                serde_json::json!((0..n).map(|_| ((r.next(1.0) + 1.0) * 4.0) as i64 % 8).collect::<Vec<_>>())
            };
            let nested = nest(&json, &tt.shape);
            graph::value_of_json(&nested, &tt).unwrap()
        })
        .collect()
}

fn nest(flat: &serde_json::Value, shape: &[u64]) -> serde_json::Value {
    let xs = flat.as_array().unwrap();
    fn go(xs: &[serde_json::Value], shape: &[u64]) -> serde_json::Value {
        match shape.split_first() {
            None => xs[0].clone(),
            Some((d, rest)) => {
                let step = xs.len() / *d as usize;
                serde_json::Value::Array((0..*d as usize).map(|i| go(&xs[i * step..(i + 1) * step], rest)).collect())
            }
        }
    }
    go(xs, shape)
}

fn ulps64(a: f64, b: f64) -> u64 {
    let k = |x: f64| {
        let b = x.to_bits() as i64;
        if b < 0 { i64::MIN - b } else { b }
    };
    k(a).abs_diff(k(b))
}

fn ulps32(a: f32, b: f32) -> u64 {
    let k = |x: f32| {
        let b = x.to_bits() as i32 as i64;
        if b < 0 { i32::MIN as i64 - b } else { b }
    };
    k(a).abs_diff(k(b))
}

/// The largest distance in units in the last place between two tensors of
/// one type (0 for integers that are equal).
fn distance(a: &HostTensor, b: &HostTensor) -> u64 {
    assert_eq!((a.dtype, &a.shape), (b.dtype, &b.shape));
    match a.dtype {
        DType::F64 => a.as_f64().unwrap().iter().zip(b.as_f64().unwrap()).map(|(x, y)| ulps64(*x, y)).max().unwrap_or(0),
        DType::F32 => a.as_f32().unwrap().iter().zip(b.as_f32().unwrap()).map(|(x, y)| ulps32(*x, y)).max().unwrap_or(0),
        _ => (a.data != b.data) as u64 * u64::MAX,
    }
}

/// Judge every term of the program at `p` on two operand sets; the count
/// of nodes judged.
fn judge(p: &PathBuf) -> usize {
    let src = std::fs::read_to_string(p).unwrap();
    let m = rce_spec_check::program(&src).unwrap();
    let ts = graph::terms_of(&m);
    assert!(!ts.is_empty(), "{} has graph terms", p.display());
    let mut judged = 0;
    for t in &ts {
        for seed in [1u64, 2] {
            let vals = operands(t, seed);
            let (rr, mut rn) = graph::reference_nodes(&t.graph, &t.types, &vals).unwrap();
            assert_eq!(rr, graph::evaluate(t, &vals), "the node-by-node reference is the reference");
            rn.sort_by_key(|x| x.0);
            let w = wasm_term::run(&t.key, &t.graph, &t.types, &vals).unwrap();
            assert_eq!(w.result, rr, "term {} of {}", t.key, p.display());
            assert_eq!(w.nodes.iter().map(|x| x.0).collect::<Vec<_>>(), rn.iter().map(|x| x.0).collect::<Vec<_>>(), "the same nodes completed");
            for ((n, a), (k, b)) in w.nodes.iter().zip(&rn) {
                assert_eq!(n, k);
                let level = t.types.levels.iter().find(|l| l.node as usize == *n).map(|l| l.level).unwrap_or(Level::Exact);
                match level {
                    Level::Exact => assert_eq!(a, b, "node {n} of {} is exact", t.key),
                    Level::Within { ulps } => assert!(distance(a, b) <= ulps as u64, "node {n} within {ulps}"),
                    Level::Lossy => assert_eq!((a.dtype, &a.shape), (b.dtype, &b.shape), "node {n} keeps its type"),
                }
                judged += 1;
            }
        }
    }
    judged
}

#[test]
fn every_node_of_the_examples_is_the_reference_at_its_level() {
    assert!(judge(&example("dense.rce")) > 0);
    assert!(judge(&example("block.rce")) > 0);
}

#[test]
fn every_node_of_the_lifted_toy_gpt2_is_the_reference_at_its_level() {
    assert!(judge(&toy("judge")) > 0);
}

#[test]
fn the_kernel_library_is_pinned_and_the_program_links_to_it_only() {
    let got = rce_fgm::hex(&rce_fgm::digest(rce_emit_wasm::KERNELS));
    assert_eq!(got, rce_emit_wasm::KERNELS_DIGEST.trim(), "the kernel library is the pinned bytes (EW-2)");
    let (st, wat, err) = rce(&["emit", "wasm", example("block.rce").to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    assert!(wat.contains(&format!("\"rce-kernels/{got}\"")), "the program imports the library by its digest");
    assert!(!wat.contains("fast_exp"), "no approximate exponential is reached (EW-3)");
    assert!(wat.contains("(@custom \"rce.levels\""));
    // the same text on every run
    assert_eq!(rce(&["emit", "wasm", example("block.rce").to_str().unwrap()]).1, wat);
}

#[test]
fn a_blob_whose_bytes_are_not_its_digest_is_refused() {
    let p = toy("blob");
    let m = rce_spec_check::program(&std::fs::read_to_string(&p).unwrap()).unwrap();
    let t = graph::terms_of(&m).into_iter().find(|t| !rce_emit_wasm::needs(&t.graph).blobs.is_empty()).expect("a term with a blob");
    let vals = operands(&t, 1);
    let bad = |d: &str| {
        let mut b = graph::blob_bytes(d)?.to_vec();
        b[0] ^= 1;
        Ok(b.into())
    };
    let r = wasm_term::run_with(&t.key, &t.graph, &t.types, &vals, &bad).unwrap();
    let e = r.result.unwrap_err();
    assert!(e.contains("the blob bytes hash to"), "{e}");
}

#[test]
fn the_commands() {
    let dir = tmp("cli");
    let ex = example("dense.rce");
    let (st, _, err) = rce(&["graph", "terms", ex.to_str().unwrap(), dir.to_str().unwrap()]);
    assert_eq!(st, 0, "{err}");
    let fg = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
    let inputs = "[[[0.5,-0.5,0.5],[-1.0,1.0,-1.0]],[1.0,2.0,3.0],[0.5,-0.25]]";
    let a = rce(&["graph", "run", fg.to_str().unwrap(), inputs]);
    let b = rce(&["graph", "run", "wasm", fg.to_str().unwrap(), inputs]);
    assert_eq!((a.0, &a.1), (0, &"[1.5,0.0]\n".to_string()));
    assert_eq!(a, b, "graph run wasm prints what graph run prints (EW-6)");
    let big = "[[[0.5,-0.5,0.5],[-1.0,1.0,-1.0]],[1.0e308,-1.0e308,1.0e308],[0.5,-0.25]]";
    assert_eq!(rce(&["graph", "run", "wasm", fg.to_str().unwrap(), big]).1, "trap: non-finite float\n");
    // a program without graph terms requires change
    let plain = dir.join("plain.rce");
    std::fs::write(&plain, "state S\n    n: u64 = 0\n").unwrap();
    let (st, _, err) = rce(&["emit", "wasm", plain.to_str().unwrap()]);
    assert_eq!(st, 1, "{err}");
    assert!(err.contains("has none"), "{err}");
}
