//! Scale on owned hardware, not paid compute. The Law of Coding says reuse beats recompute,
//! so scale comes from (a) federated training across the owned fleet (no raw data leaves a
//! node, no GPU bill) and (b) corpus mining / crystallization rather than brute force.
//! Deterministic federated averaging of the proposer seed; CPU-only.

use crate::train::Scorer;

/// Train each fleet shard locally, then average weights deterministically.
/// Shards can run on separate owned nodes; only model weights are combined.
pub fn federated_train(shards: &[Vec<(Vec<f64>, bool)>], epochs: usize, lr: f64) -> Scorer {
    let models: Vec<Scorer> = shards.iter().map(|s| Scorer::train(s, epochs, lr)).collect();
    let n = models.len().max(1) as f64;
    let dim = models.first().map(|m| m.w.len()).unwrap_or(0);
    let mut w = vec![0.0; dim];
    let mut b = 0.0;
    for m in &models {
        for (i, x) in m.w.iter().enumerate() {
            w[i] += x / n;
        }
        b += m.b / n;
    }
    Scorer { w, b }
}

/// Holdout accuracy of a scorer.
pub fn accuracy(s: &Scorer, hold: &[(Vec<f64>, bool)]) -> f64 {
    if hold.is_empty() {
        return 0.0;
    }
    let c = hold.iter().filter(|(x, y)| (s.score(x) > 0.5) == *y).count();
    c as f64 / hold.len() as f64
}

/// Split a corpus into k contiguous shards (one per fleet node).
pub fn shard(corpus: &[(Vec<f64>, bool)], k: usize) -> Vec<Vec<(Vec<f64>, bool)>> {
    let k = k.max(1);
    let per = corpus.len() / k;
    (0..k)
        .map(|i| {
            let lo = i * per;
            let hi = if i == k - 1 { corpus.len() } else { (i + 1) * per };
            corpus[lo..hi].to_vec()
        })
        .collect()
}

/// One shard's placement on a hybrid fabric.
#[derive(Clone, Debug)]
pub struct HybridAssign {
    pub shard: usize,
    pub fabric: String,
    pub est_uj: f64,
    pub est_us: f64,
}

/// Per-row latency proxy (µs) by fabric kind — a modelled throughput prior, not a measurement.
fn row_us(kind: &crate::fabric::Kind) -> f64 {
    use crate::fabric::Kind::*;
    match kind {
        Tensor => 0.02,
        Dataflow => 0.03,
        Gpu => 0.05,
        Wasm => 2.0,
        _ => 1.0,
    }
}

/// Mosaic-style hybrid partitioning with admission control: assign each shard (row count)
/// to the cheapest fabric that fits the memory cap and keeps that fabric's cumulative
/// latency within `budget_us`. Refuse (None) on overload rather than silently miss a
/// deadline — the MosaicSched admission-control lineage, on our joule cost model.
pub fn hybrid_plan(rows: &[usize], budget_us: f64, cap_kib: u64) -> Option<Vec<HybridAssign>> {
    let mut fabrics = crate::fabric::builtin_fabrics();
    fabrics.sort_by(|a, b| {
        crate::fabric::energy_muladd(a)
            .partial_cmp(&crate::fabric::energy_muladd(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut load: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut out = Vec::new();
    for (i, n) in rows.iter().enumerate() {
        let mut placed = false;
        for f in &fabrics {
            if crate::fabric::memory_kib(&f.name) > cap_kib {
                continue;
            }
            let us = *n as f64 * row_us(&f.kind);
            let cur = load.get(f.name).copied().unwrap_or(0.0);
            if cur + us > budget_us {
                continue;
            }
            load.insert(f.name.to_string(), cur + us);
            out.push(HybridAssign {
                shard: i,
                fabric: f.name.to_string(),
                est_uj: crate::fabric::energy_muladd(f) * *n as f64,
                est_us: us,
            });
            placed = true;
            break;
        }
        if !placed {
            return Option::None;
        }
    }
    Option::Some(out)
}

/// Train the proposer on a hybrid fabric path: a native BLAS kernel (Accelerate / AMX on
/// Apple silicon, vecLib elsewhere) compiled with clang, instead of the scalar Rust loop.
/// Returns the trained scorer; falls back to Err if no native toolchain is present.
pub fn train_native(samples: &[(Vec<f64>, bool)], epochs: usize, lr: f64) -> Result<Scorer, String> {
    let dim = samples.first().map(|(x, _)| x.len()).unwrap_or(0);
    let n = samples.len();
    if dim == 0 || n == 0 {
        return Err("empty corpus".into());
    }
    let mut input = String::new();
    input.push_str(&format!("{n} {dim} {epochs} {lr}\n"));
    for (x, y) in samples {
        for v in x {
            input.push_str(&format!("{v} "));
        }
        input.push_str(if *y { "1\n" } else { "0\n" });
    }
    let dir = std::env::temp_dir().join(format!("rce-native-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let csrc = dir.join("gd.c");
    let bin = dir.join("gd");
    let inf = dir.join("in.txt");
    std::fs::write(&csrc, NATIVE_GD).map_err(|e| e.to_string())?;
    std::fs::write(&inf, input).map_err(|e| e.to_string())?;
    let cc = std::process::Command::new("clang")
        .args(["-O2", "-framework", "Accelerate"])
        .arg(&csrc)
        .arg("-o")
        .arg(&bin)
        .output()
        .map_err(|e| e.to_string())?;
    if !cc.status.success() {
        return Err(String::from_utf8_lossy(&cc.stderr).into_owned());
    }
    let run = std::process::Command::new(&bin)
        .stdin(std::fs::File::open(&inf).map_err(|e| e.to_string())?)
        .output()
        .map_err(|e| e.to_string())?;
    if !run.status.success() {
        return Err(String::from_utf8_lossy(&run.stderr).into_owned());
    }
    let out = String::from_utf8_lossy(&run.stdout);
    let nums: Vec<f64> = out.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    if nums.len() != dim + 1 {
        return Err(format!("native kernel returned {} numbers, want {}", nums.len(), dim + 1));
    }
    Ok(Scorer { w: nums[..dim].to_vec(), b: nums[dim] })
}

const NATIVE_GD: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <Accelerate/Accelerate.h>
int main(void){
  int n,d,epochs; double lr;
  if(scanf("%d %d %d %lf",&n,&d,&epochs,&lr)!=4) return 1;
  double *X=malloc(sizeof(double)*n*d), *y=malloc(sizeof(double)*n);
  for(int i=0;i<n;i++){ for(int j=0;j<d;j++) scanf("%lf",&X[i*d+j]); scanf("%lf",&y[i]); }
  double *w=calloc(d,sizeof(double)), b=0;
  double *z=malloc(sizeof(double)*n), *e=malloc(sizeof(double)*n), *g=calloc(d,sizeof(double));
  for(int ep=0;ep<epochs;ep++){
    cblas_dgemv(CblasRowMajor,CblasNoTrans,n,d,1.0,X,d,w,1,0.0,z,1);
    for(int i=0;i<n;i++){ z[i]+=b; e[i]=y[i]-1.0/(1.0+exp(-z[i])); }
    cblas_dgemv(CblasRowMajor,CblasTrans,n,d,1.0,X,d,e,1,0.0,g,1);
    double ones=1.0; double gb=cblas_ddot(n,e,1,&ones,0);
    cblas_daxpy(d,lr/n,g,1,w,1);
    b+=lr/n*gb;
  }
  for(int j=0;j<d;j++) printf("%.17g ",w[j]);
  printf("%.17g\n",b);
  return 0;
}
"#;
