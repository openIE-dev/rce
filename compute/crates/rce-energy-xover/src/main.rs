//! RCE energy crossover. Stage-A calibration harness.
//!
//! Samples whole-SoC energy per square f32 matmul on CPU (op-blas
//! AMX/Accelerate sgemm) and GPU (op-mps `MPSMatrixMultiplication`, forced
//! on for every shape via `with_min_flops(0)`). The meter is
//! `open_default_backend()`. On macOS that backend is `fallback-tdp`
//! (`est`) unless a counter backend is wired in. On Linux it is RAPL
//! (`reported`) when a readable powercap zone exists, and `fallback-tdp`
//! otherwise. Figures follow the printed meter name. `macos_ioreport.rs`
//! is not in this repository; that backend is planned. The name
//! `macos-ioreport` is not returned by `open_default_backend`.
//!
//! The cost-table lines print `basis`, the meter's D.11 basis. They do
//! not print `measured = true`.
//!
//! Methodology (the meter's limits, design doc section 4):
//!   * Run a batch of K matmuls per sample so the window is about 250 ms.
//!   * Subtract a same-duration idle window.
//!   * Take the median of M repeats.
//!   * CPU samples use the Cpu channel. GPU samples use Gpu and Dram.
//!
//! On Linux RAPL and the TDP fallback the counter is cumulative since
//! open, so an interval is `s_after.total_uj_since(&s_before)`. A planned
//! IOReport backend would return a since-previous delta. This file calls
//! `snapshot()` and reads the channels of the second snapshot, which is
//! the interval for a since-previous backend and the cumulative total for
//! RAPL and the TDP fallback. `rce-placement-adapter::SubstrateEnergyMeter`
//! switches on `backend.name()` for the interval.

use std::time::{Duration, Instant};

use rce_op_mps::MpsMatmul;
use rce_vm_backend_cpu::{DType, HostTensor, OpDispatch, Value};
use rce_energy::{Channel, EnergyTelemetry, open_default_backend};

/// Square matmul sweep — denser than the original to bracket the crossover.
const SQUARE: &[usize] = &[64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072];
/// Non-square cases exercise the FLOPs+bytes feature model (§4.4): cells where
/// FLOPs are similar but byte-touch (transfer cost) differs — separating the
/// `a` and `b` coefficients in `pJ ≈ a·FLOPs + b·bytes + c`.
const NON_SQUARE: &[(usize, usize, usize)] = &[
    (1024, 256, 1024),
    (512, 2048, 512),
    (4096, 128, 4096),
];
const TARGET_MS: u128 = 250;
const M: usize = 5;
const FLOOR_UJ: f64 = 1.0; // µJ/op below this is lost in the noise — refuse to report a number

type Meter = Box<dyn EnergyTelemetry>;
type Chans = std::collections::BTreeMap<Channel, u64>;

/// Per-channel µJ consumed over `work` (the snapshot taken *after* a reset
/// snapshot; on Apple IOReport a snapshot is energy-since-previous-snapshot).
fn window<F: FnOnce()>(meter: &mut Meter, work: F) -> Chans {
    let _ = meter.snapshot(); // reset the interval start
    work();
    meter.snapshot().map(|s| s.channels).unwrap_or_default()
}

fn sum_ch(m: &Chans, chans: &[Channel]) -> u64 {
    chans.iter().map(|c| m.get(c).copied().unwrap_or(0)).sum()
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

/// Ordinary least squares fit: y = a·x1 + b·x2 + c, by Gauss-eliminating the
/// normal equations. Returns `None` if singular or fewer than 3 samples.
/// Used for the cost-model feature fit (§4.4): x1 = FLOPs, x2 = bytes_touched,
/// y = sampled pJ; a = pJ/FLOP, b = pJ/byte, c = fixed overhead pJ.
fn ols3(samples: &[(f64, f64, f64)]) -> Option<(f64, f64, f64)> {
    if samples.len() < 3 { return None; }
    let (mut s1, mut s2, mut sy) = (0.0, 0.0, 0.0);
    let (mut s11, mut s12, mut s22) = (0.0, 0.0, 0.0);
    let (mut s1y, mut s2y) = (0.0, 0.0);
    let n = samples.len() as f64;
    for &(x1, x2, y) in samples {
        s1 += x1; s2 += x2; sy += y;
        s11 += x1 * x1; s12 += x1 * x2; s22 += x2 * x2;
        s1y += x1 * y; s2y += x2 * y;
    }
    let mut m = [
        [s11, s12, s1, s1y],
        [s12, s22, s2, s2y],
        [s1,  s2,  n,  sy ],
    ];
    for i in 0..3 {
        let mut pivot = i;
        for r in i + 1..3 { if m[r][i].abs() > m[pivot][i].abs() { pivot = r; } }
        m.swap(i, pivot);
        if m[i][i].abs() < 1e-12 { return None; }
        for r in i + 1..3 {
            let f = m[r][i] / m[i][i];
            for c in i..4 { m[r][c] -= f * m[i][c]; }
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = m[i][3];
        for j in i + 1..3 { s -= m[i][j] * x[j]; }
        x[i] = s / m[i][i];
    }
    Some((x[0], x[1], x[2]))
}

/// Measure active µJ-per-op for `run_once`. Per the design doc §4: each active
/// window is **paired** with an immediately-following idle window of equal wall
/// time; their difference cancels common background (slow drift), leaving the
/// matmul's *marginal* energy. Median of M, with p90 so noise is visible — on a
/// contended machine the spread blows up, which is the honest signal that the
/// run needs a quiesced host.
fn measure<F: FnMut()>(meter: &mut Meter, chans: &[Channel], mut run_once: F) -> (f64, f64) {
    for _ in 0..3 { run_once(); } // warm caches / pipelines / GPU resource cache
    let t0 = Instant::now();
    run_once();
    let per_us = t0.elapsed().as_micros().max(1);
    let k = ((TARGET_MS * 1000) / per_us).clamp(1, 4_000_000) as usize;

    let mut samples = Vec::with_capacity(M);
    for _ in 0..M {
        let w0 = Instant::now();
        let active = window(meter, || { for _ in 0..k { run_once(); } });
        let dur = w0.elapsed();
        let idle = window(meter, || std::thread::sleep(dur));
        let marginal = sum_ch(&active, chans).saturating_sub(sum_ch(&idle, chans)) as f64;
        samples.push(marginal / k as f64); // µJ per matmul, on the engine's channels
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (percentile(&samples, 0.5), percentile(&samples, 0.9))
}

fn rand_f32(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*seed >> 33) as f32) / ((1u64 << 31) as f32) - 1.0
        })
        .collect()
}

fn f32_tensor(data: &[f32], shape: Vec<usize>) -> Value {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    Value::Tensor(HostTensor::new(shape, DType::F32, bytes).expect("host tensor"))
}

fn meter_basis(name: &str) -> &'static str {
    match name {
        "linux-rapl" | "macos-ioreport" => "reported",
        _ => "est",
    }
}

fn main() {
    let mut meter = open_default_backend();
    let cal = meter.calibration().clone();
    let basis = meter_basis(meter.name());
    println!("# RCE energy crossover: CPU(op-blas) vs GPU(op-mps) matmul");
    println!("# meter: {} on {}", meter.name(), if cal.model.is_empty() { "unknown" } else { &cal.model });
    println!("# basis: {basis} (D.11). fallback-tdp is est. linux-rapl is reported. measured is a wall or shunt meter, which this program does not open.");

    let mps = match MpsMatmul::with_min_flops(0) {
        Some(m) => m, // min_flops=0 ⇒ GPU forced for every shape (no BLAS fallback)
        None => {
            eprintln!("MPS unavailable on this machine; cannot run the GPU path.");
            std::process::exit(1);
        }
    };

    // Whole-SoC power over a ~200 ms window, as a contention indicator: this
    // measurement is only trustworthy on a quiesced host (design doc §4.6).
    let bw = sum_ch(&window(&mut meter, || std::thread::sleep(Duration::from_millis(200))), &[Channel::Total]);
    let watts = bw as f64 / 200.0 / 1000.0;
    println!("# whole-SoC power now: {:.1} W  ({})", watts,
        if watts > 30.0 { "CONTENDED — results will be noisy; rerun on a quiesced host" } else { "quiet enough" });
    println!();
    println!("{:>14}  {:>9}  {:>12}  {:>12}  {:>10}  {:>10}  {:>8}  {:>7}",
        "M×K×N", "GFLOPs", "CPU µJ/op", "GPU µJ/op", "CPU pJ/F", "GPU pJ/F", "winner", "ratio");
    println!("{}", "-".repeat(98));

    // (m, k, n) tuple list: square sweep + non-square stress cases.
    let mut shapes: Vec<(usize, usize, usize)> =
        SQUARE.iter().map(|&n| (n, n, n)).collect();
    shapes.extend_from_slice(NON_SQUARE);

    let mut cpu_pj_per_flop: Vec<f64> = Vec::new();
    let mut gpu_pj_per_flop: Vec<f64> = Vec::new();
    // (FLOPs, bytes, µJ) samples for the OLS fit (§4.4 feature model).
    let mut cpu_xy: Vec<(f64, f64, f64)> = Vec::new();
    let mut gpu_xy: Vec<(f64, f64, f64)> = Vec::new();
    let mut sanity_done = false;

    let mut seed = 0x9e3779b97f4a7c15u64;
    for (mi, ki, ni) in shapes {
        let (m, k, n) = (mi, ki, ni);
        let a = rand_f32(m * k, &mut seed);
        let b = rand_f32(k * n, &mut seed);
        let mut c = vec![0.0f32; m * n];

        // GPU args built once (no per-call host clone); reused each iteration.
        let args = [f32_tensor(&a, vec![m, k]), f32_tensor(&b, vec![k, n])];
        if !sanity_done {
            if mps.execute(0, "matmul", &args).is_err() {
                eprintln!("MPS matmul returned an error on first shape; aborting.");
                std::process::exit(1);
            }
            sanity_done = true;
        }

        // CPU on the Cpu channel; GPU on Gpu+Dram (compute + memory/transfer).
        let (cpu, _) = measure(&mut meter, &[Channel::Cpu], || {
            rce_op_blas::sgemm_into(&a, &b, &mut c, m, k, n);
            std::hint::black_box(c[0]);
        });
        let (gpu, _) = measure(&mut meter, &[Channel::Gpu, Channel::Dram], || {
            std::hint::black_box(mps.execute(0, "matmul", &args).is_ok());
        });

        let flops = 2u64.saturating_mul(m as u64).saturating_mul(k as u64).saturating_mul(n as u64);
        // Bytes touched (the `b` axis of the feature model): A + B + C, f32.
        let bytes = 4u64 * (m as u64 * k as u64 + k as u64 * n as u64 + m as u64 * n as u64);
        let gflops = flops as f64 / 1e9;
        let cpu_pjf = if cpu >= FLOOR_UJ && flops > 0 { cpu * 1e6 / flops as f64 } else { f64::NAN };
        let gpu_pjf = if gpu >= FLOOR_UJ && flops > 0 { gpu * 1e6 / flops as f64 } else { f64::NAN };
        if cpu_pjf.is_finite() {
            cpu_pj_per_flop.push(cpu_pjf);
            // pJ = µJ * 1e6
            cpu_xy.push((flops as f64, bytes as f64, cpu * 1e6));
        }
        if gpu_pjf.is_finite() {
            gpu_pj_per_flop.push(gpu_pjf);
            gpu_xy.push((flops as f64, bytes as f64, gpu * 1e6));
        }

        let (winner, ratio) = if cpu < FLOOR_UJ || gpu < FLOOR_UJ {
            ("(noise)".to_string(), String::new())
        } else if cpu <= gpu {
            ("CPU".to_string(), format!("{:.2}x", gpu / cpu))
        } else {
            ("GPU".to_string(), format!("{:.2}x", cpu / gpu))
        };
        let shape_str = format!("{}x{}x{}", m, k, n);
        let fmt = |v: f64| if v.is_finite() { format!("{:>10.2}", v) } else { "     —    ".to_string() };
        println!("{:>14}  {:>9.3}  {:>12.2}  {:>12.2}  {}  {}  {:>8}  {:>7}",
            shape_str, gflops, cpu, gpu, fmt(cpu_pjf), fmt(gpu_pjf), winner, ratio);
    }

    // Median pJ/FLOP per backend across rows where the signal was above noise.
    let med = |mut v: Vec<f64>| -> Option<f64> {
        if v.is_empty() { return None; }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[v.len() / 2])
    };
    println!();
    println!("# cost-model coefficients (seed of the CostTable §4.3):");
    match med(cpu_pj_per_flop) {
        Some(p) => println!("#   CPU (op-blas sgemm, AMX): median {:.2} pJ/FLOP", p),
        None    => println!("#   CPU: all rows below noise floor — rerun on a quiesced host"),
    }
    match med(gpu_pj_per_flop) {
        Some(p) => println!("#   GPU (op-mps MPSMatrixMul):  median {:.2} pJ/FLOP", p),
        None    => println!("#   GPU: all rows below noise floor — rerun on a quiesced host"),
    }
    // OLS fit: pJ ≈ a·FLOPs + b·bytes + c — the feature-model the placement
    // pass consumes. `a` is the pJ-per-FLOP, `b` is the pJ-per-byte transfer,
    // `c` is the per-call fixed overhead (dispatch + kernel launch).
    println!();
    println!("# OLS feature-model fit  pJ ≈ a·FLOPs + b·bytes_touched + c:");
    match ols3(&cpu_xy) {
        Some((a, b, c)) => println!(
            "#   CPU:  a = {:>8.3} pJ/FLOP   b = {:>8.3} pJ/byte   c = {:>9.2} µJ", a, b, c / 1e6
        ),
        None => println!("#   CPU: insufficient samples for OLS (need ≥3 resolved shapes)"),
    }
    match ols3(&gpu_xy) {
        Some((a, b, c)) => println!(
            "#   GPU:  a = {:>8.3} pJ/FLOP   b = {:>8.3} pJ/byte   c = {:>9.2} µJ", a, b, c / 1e6
        ),
        None => println!("#   GPU: insufficient samples for OLS (need ≥3 resolved shapes)"),
    }

    // Persisted CostTable in TOML — drops straight into rce-placement::CostTable
    // via a small loader (§4.5 of the design doc closes the calibration loop).
    // Emitted only when both fits succeeded; on a noisy run nothing is written.
    if let (Some((ca, cb, cc)), Some((ga, gb, gc))) = (ols3(&cpu_xy), ols3(&gpu_xy)) {
        println!();
        println!("# ----- cost_table.toml (paste into a file; load at scheduler init) -----");
        println!("[[cells]]");
        println!("op = \"matmul\"");
        println!("backend = \"cpu:op-blas\"");
        println!("a_pj_per_flop = {ca:.6}");
        println!("b_pj_per_byte = {cb:.6}");
        println!("c_pj_overhead = {cc:.3}");
        println!("basis = \"{basis}\"");
        println!();
        println!("[[cells]]");
        println!("op = \"matmul\"");
        println!("backend = \"gpu:op-mps\"");
        println!("a_pj_per_flop = {ga:.6}");
        println!("b_pj_per_byte = {gb:.6}");
        println!("c_pj_overhead = {gc:.3}");
        println!("basis = \"{basis}\"");
        println!("# ----- end cost_table.toml -----");
    }

    println!();
    println!("# A min-joule placement selector would pick CPU where CPU µJ/matmul is lower");
    println!("# and GPU where GPU is lower — the crossover is the routing decision boundary.");
    println!("# (vs op-mps's hardcoded 10-GFLOP *latency* threshold; this is the *energy* one.)");
}
