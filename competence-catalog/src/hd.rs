//! Compute-light, cross-fabric learning (not old brute-force backprop):
//!  * HD/VSA — random hypervector encoding + one-shot bundling of class prototypes;
//!    learning is a single pass of bind/bundle, no epochs, no backprop. The ops are
//!    elementwise, so the SAME math runs on CPU, GPU, ANE or FPGA.
//!  * ELM/reservoir — frozen random hidden layer + a single closed-form ridge readout
//!    (one small linear solve), no iterative training.
//! A CUDA source is emitted as data so a cloud GPU runs the identical kernel; a native
//! Accelerate/AMX kernel runs it on this host. Cross-fabric by construction.

/// Hypervector dimension.
pub const DIM: usize = 1024;

fn basis(j: usize, d: usize) -> f64 {
    let mut h = (j as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (d as u64).wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58476D1CE4E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D049BB133111EB);
    h ^= h >> 31;
    if h & 1 == 0 { 1.0 } else { -1.0 }
}

/// Encode a feature vector as a hypervector (bundle of value-scaled random basis vectors).
pub fn encode(x: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; DIM];
    for (j, xj) in x.iter().enumerate() {
        for d in 0..DIM {
            v[d] += xj * basis(j, d);
        }
    }
    v
}

/// Bind (elementwise product) — the VSA composition operator.
pub fn bind(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// One-shot HD model: a prototype hypervector per class (bundle of exemplars).
pub struct HdModel {
    pub prototypes: Vec<Vec<f64>>,
}

impl HdModel {
    /// Single pass: bundle exemplars per class. No epochs, no backprop.
    pub fn train(samples: &[(Vec<f64>, bool)]) -> HdModel {
        let mut pos = vec![0.0; DIM];
        let mut neg = vec![0.0; DIM];
        for (x, y) in samples {
            let e = encode(x);
            let target = if *y { &mut pos } else { &mut neg };
            for d in 0..DIM {
                target[d] += e[d];
            }
        }
        HdModel { prototypes: vec![neg, pos] }
    }

    pub fn predict(&self, x: &[f64]) -> bool {
        let e = encode(x);
        cosine(&e, &self.prototypes[1]) >= cosine(&e, &self.prototypes[0])
    }
}

/// Frozen random hidden layer (reservoir/ELM) + closed-form ridge readout: one linear solve.
pub struct Elm {
    pub w: Vec<Vec<f64>>,
    pub readout: Vec<f64>,
    pub hidden: usize,
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        for k in i + 1..n {
            if a[k][i].abs() > a[piv][i].abs() {
                piv = k;
            }
        }
        if a[piv][i].abs() < 1e-12 {
            return Option::None;
        }
        a.swap(i, piv);
        b.swap(i, piv);
        for k in i + 1..n {
            let f = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= f * a[i][j];
            }
            b[k] -= f * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in i + 1..n {
            s -= a[i][j] * x[j];
        }
        x[i] = s / a[i][i];
    }
    Option::Some(x)
}

impl Elm {
    pub fn train(samples: &[(Vec<f64>, bool)], hidden: usize, ridge: f64) -> Elm {
        let dim = samples.first().map(|(x, _)| x.len()).unwrap_or(0);
        let w: Vec<Vec<f64>> = (0..dim)
            .map(|j| (0..hidden).map(|h| basis(j, h + 7919) * 0.5).collect())
            .collect();
        let feats: Vec<Vec<f64>> = samples
            .iter()
            .map(|(x, _)| {
                let mut f = vec![1.0];
                for h in 0..hidden {
                    let s: f64 = x.iter().enumerate().map(|(j, xj)| xj * w[j][h]).sum();
                    f.push(s.tanh());
                }
                f
            })
            .collect();
        let m = hidden + 1;
        let mut ata = vec![vec![0.0; m]; m];
        let mut atb = vec![0.0; m];
        for (f, (_, y)) in feats.iter().zip(samples) {
            for i in 0..m {
                atb[i] += f[i] * if *y { 1.0 } else { 0.0 };
                for j in 0..m {
                    ata[i][j] += f[i] * f[j];
                }
            }
        }
        for i in 0..m {
            ata[i][i] += ridge;
        }
        let readout = solve(ata, atb).unwrap_or_else(|| vec![0.0; m]);
        Elm { w, readout, hidden }
    }

    pub fn predict(&self, x: &[f64]) -> bool {
        let mut s = self.readout[0];
        for h in 0..self.hidden {
            let z: f64 = x.iter().enumerate().map(|(j, xj)| xj * self.w[j][h]).sum();
            s += self.readout[h + 1] * z.tanh();
        }
        s > 0.5
    }
}

/// The identical HD encode+cosine kernel as CUDA source (data), for a cloud GPU.
pub fn cuda_source() -> &'static str {
    r#"
__global__ void hd_encode(const double* x, int dim, double* out, int D){
  int d = blockIdx.x*blockDim.x + threadIdx.x;
  if(d>=D) return;
  double acc=0;
  for(int j=0;j<dim;j++){
    unsigned long long h = (unsigned long long)j*0x9E3779B97F4A7C15ULL ^ (unsigned long long)d*0xBF58476D1CE4E5B9ULL;
    h ^= h>>30; h*=0xBF58476D1CE4E5B9ULL; h ^= h>>27; h*=0x94D049BB133111EBULL; h ^= h>>31;
    acc += x[j] * ((h&1ULL)==0ULL ? 1.0 : -1.0);
  }
  out[d]=acc;
}
"#
}

/// Run the HD encode+cosine on a native Accelerate/AMX kernel and return the prediction,
/// proving the same math executes on a second fabric on this host.
pub fn predict_native(model: &HdModel, x: &[f64]) -> Result<bool, String> {
    let dir = std::env::temp_dir().join(format!("rce-hd-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let csrc = dir.join("hd.c");
    let bin = dir.join("hd");
    let inf = dir.join("in.txt");
    let mut input = String::new();
    input.push_str(&format!("{} {}\n", x.len(), DIM));
    for v in x {
        input.push_str(&format!("{v} "));
    }
    input.push('\n');
    for p in &model.prototypes {
        for v in p {
            input.push_str(&format!("{v} "));
        }
        input.push('\n');
    }
    std::fs::write(&csrc, NATIVE_HD).map_err(|e| e.to_string())?;
    std::fs::write(&inf, input).map_err(|e| e.to_string())?;
    let cc = std::process::Command::new("clang")
        .args(["-O2", "-framework", "Accelerate"])
        .arg(&csrc).arg("-o").arg(&bin)
        .output().map_err(|e| e.to_string())?;
    if !cc.status.success() {
        return Err(String::from_utf8_lossy(&cc.stderr).into_owned());
    }
    let run = std::process::Command::new(&bin)
        .stdin(std::fs::File::open(&inf).map_err(|e| e.to_string())?)
        .output().map_err(|e| e.to_string())?;
    if !run.status.success() {
        return Err(String::from_utf8_lossy(&run.stderr).into_owned());
    }
    let out = String::from_utf8_lossy(&run.stdout);
    Ok(out.trim() == "1")
}

const NATIVE_HD: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <Accelerate/Accelerate.h>
static double basis(int j,int d){
  unsigned long long h=(unsigned long long)j*0x9E3779B97F4A7C15ULL ^ (unsigned long long)d*0xBF58476D1CE4E5B9ULL;
  h^=h>>30; h*=0xBF58476D1CE4E5B9ULL; h^=h>>27; h*=0x94D049BB133111EBULL; h^=h>>31;
  return (h&1ULL)==0ULL?1.0:-1.0;
}
int main(void){
  int dim,D; if(scanf("%d %d",&dim,&D)!=2) return 1;
  double *x=malloc(sizeof(double)*dim);
  for(int j=0;j<dim;j++) scanf("%lf",&x[j]);
  double *e=calloc(D,sizeof(double));
  for(int j=0;j<dim;j++) for(int d=0;d<D;d++) e[d]+=x[j]*basis(j,d);
  double *p0=malloc(sizeof(double)*D), *p1=malloc(sizeof(double)*D);
  for(int d=0;d<D;d++) scanf("%lf",&p0[d]);
  for(int d=0;d<D;d++) scanf("%lf",&p1[d]);
  double d0=cblas_ddot(D,e,1,p0,1), d1=cblas_ddot(D,e,1,p1,1);
  double n0=sqrt(cblas_ddot(D,p0,1,p0,1)), n1=sqrt(cblas_ddot(D,p1,1,p1,1));
  double ne=sqrt(cblas_ddot(D,e,1,e,1));
  double c0=(ne*n0)>0?d0/(ne*n0):0, c1=(ne*n1)>0?d1/(ne*n1):0;
  printf("%d\n", c1>=c0?1:0);
  return 0;
}
"#;
