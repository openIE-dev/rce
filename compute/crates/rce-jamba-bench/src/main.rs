//! `rce-jamba-bench` — Phase W. Real-GGUF Jamba prefill latency
//! sweep with CSV / JSON output.
//!
//! ## What this measures
//!
//! End-to-end wall clock of [`rce_jamba::JambaRuntime::prefill_logits`]
//! on a real Jamba-Reasoning-3B GGUF, across a configurable set of
//! prompt lengths, with a configurable warmup count + iteration count
//! per length. Reports per-iteration wall ms and ms/token, plus a
//! summary (min / median / mean / p95).
//!
//! ## Why a separate crate
//!
//! Previous bench-scaffold landings lived inside `rce-jamba/src/lib.rs`
//! as `#[ignore]` tests. That coupling means every bench iteration
//! requires editing the jamba crate, which conflicts with in-flight
//! attention-refactor work. A standalone binary depends on
//! `rce-jamba`'s public API only, so the bench harness can evolve
//! freely without touching jamba's tree.
//!
//! ## Configuration (env vars)
//!
//! - `JAMBA_GGUF` — required. Path to the GGUF file. Bench exits
//!   cleanly with a skip message if unset or pointing to a missing
//!   file.
//! - `JAMBA_BENCH_PROMPT_LENS` — comma-separated prompt lengths to
//!   sweep. Default `32,64,128,256`.
//! - `JAMBA_BENCH_ITERS` — timed iterations per prompt length.
//!   Default `5`.
//! - `JAMBA_BENCH_WARMUP` — warmup iterations per prompt length
//!   (not recorded). Default `1`.
//! - `JAMBA_BENCH_FORMAT` — `csv` (default) or `json`. CSV writes
//!   one row per iteration to stdout; JSON writes a single document
//!   with `runs`, `summary`, and `meta`.
//!
//! ## Output formats
//!
//! CSV header (always written first):
//!
//! ```text
//! prompt_len,iter,wall_ms,ms_per_token
//! ```
//!
//! JSON shape:
//!
//! ```json
//! {
//!   "meta": { "model_path": "...", "load_ms": 4321.0, "iters": 5, "warmup": 1 },
//!   "runs": [
//!     { "prompt_len": 32, "iter": 0, "wall_ms": 412.7, "ms_per_token": 12.9 },
//!     ...
//!   ],
//!   "summary": [
//!     { "prompt_len": 32, "min_ms": 410.1, "median_ms": 412.7, "mean_ms": 413.2, "p95_ms": 418.4 },
//!     ...
//!   ]
//! }
//! ```
//!
//! No external JSON dependency — output is hand-emitted (the schema
//! is small and stable).

use rce_jamba::JambaRuntime;
use std::env;
use std::path::PathBuf;
use std::time::Instant;

const DEFAULT_PROMPT_LENS: &[usize] = &[32, 64, 128, 256];
const DEFAULT_ITERS: usize = 5;
const DEFAULT_WARMUP: usize = 1;

fn parse_prompt_lens(raw: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for tok in raw.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let n: usize = tok
            .parse()
            .map_err(|e| format!("JAMBA_BENCH_PROMPT_LENS: bad number `{tok}`: {e}"))?;
        if n == 0 {
            return Err(format!("JAMBA_BENCH_PROMPT_LENS: 0 is not a valid prompt length"));
        }
        out.push(n);
    }
    if out.is_empty() {
        return Err("JAMBA_BENCH_PROMPT_LENS: empty after parse".into());
    }
    Ok(out)
}

#[derive(Clone, Copy)]
enum Format {
    Csv,
    Json,
}

impl Format {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.to_ascii_lowercase().as_str() {
            "csv" => Ok(Format::Csv),
            "json" => Ok(Format::Json),
            other => Err(format!("JAMBA_BENCH_FORMAT: expected csv or json, got `{other}`")),
        }
    }
}

#[derive(Clone, Copy)]
struct Run {
    prompt_len: usize,
    iter: usize,
    wall_ms: f64,
    ms_per_token: f64,
}

struct Summary {
    prompt_len: usize,
    min_ms: f64,
    median_ms: f64,
    mean_ms: f64,
    p95_ms: f64,
}

fn summarize(prompt_len: usize, runs: &[Run]) -> Summary {
    let mut ms: Vec<f64> = runs.iter().map(|r| r.wall_ms).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms.len();
    let min_ms = ms[0];
    let median_ms = if n % 2 == 1 {
        ms[n / 2]
    } else {
        0.5 * (ms[n / 2 - 1] + ms[n / 2])
    };
    let mean_ms = ms.iter().sum::<f64>() / n as f64;
    // Linear interpolation between adjacent ranks; clamps to last
    // sample for tiny n. For n=5 this puts p95 ≈ max(ms).
    let rank = 0.95 * (n as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    let p95_ms = ms[lo] + frac * (ms[hi] - ms[lo]);
    Summary { prompt_len, min_ms, median_ms, mean_ms, p95_ms }
}

fn synth_prompt(prompt_len: usize) -> Vec<i32> {
    // Deterministic synthetic prompt — vocab=0..32K is safe for any
    // jamba checkpoint. Pattern keeps tokens distinct so any
    // prefix-cache short-circuit can't accidentally inflate the
    // measurement.
    (0..prompt_len).map(|i| ((i as i32 * 17 + 3) & 0x7FFF) + 1).collect()
}

fn emit_csv(runs: &[Run]) {
    println!("prompt_len,iter,wall_ms,ms_per_token");
    for r in runs {
        println!(
            "{},{},{:.3},{:.3}",
            r.prompt_len, r.iter, r.wall_ms, r.ms_per_token,
        );
    }
}

fn emit_json(
    model_path: &str,
    load_ms: f64,
    iters: usize,
    warmup: usize,
    runs: &[Run],
    summaries: &[Summary],
) {
    println!("{{");
    println!(
        "  \"meta\": {{ \"model_path\": \"{}\", \"load_ms\": {:.3}, \"iters\": {}, \"warmup\": {} }},",
        escape_json(model_path), load_ms, iters, warmup,
    );
    println!("  \"runs\": [");
    for (i, r) in runs.iter().enumerate() {
        let comma = if i + 1 < runs.len() { "," } else { "" };
        println!(
            "    {{ \"prompt_len\": {}, \"iter\": {}, \"wall_ms\": {:.3}, \"ms_per_token\": {:.3} }}{}",
            r.prompt_len, r.iter, r.wall_ms, r.ms_per_token, comma,
        );
    }
    println!("  ],");
    println!("  \"summary\": [");
    for (i, s) in summaries.iter().enumerate() {
        let comma = if i + 1 < summaries.len() { "," } else { "" };
        println!(
            "    {{ \"prompt_len\": {}, \"min_ms\": {:.3}, \"median_ms\": {:.3}, \"mean_ms\": {:.3}, \"p95_ms\": {:.3} }}{}",
            s.prompt_len, s.min_ms, s.median_ms, s.mean_ms, s.p95_ms, comma,
        );
    }
    println!("  ]");
    println!("}}");
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn main() {
    let path = match env::var("JAMBA_GGUF") {
        Ok(p) => p,
        Err(_) => {
            eprintln!(
                "JAMBA_GGUF not set; skipping bench. Set it to a Jamba-Reasoning-3B GGUF path \
                 and re-run."
            );
            return;
        }
    };
    let path_buf = PathBuf::from(&path);
    if !path_buf.exists() {
        eprintln!("JAMBA_GGUF = `{path}` does not exist; skipping bench.");
        return;
    }

    let prompt_lens = match env::var("JAMBA_BENCH_PROMPT_LENS") {
        Ok(raw) => match parse_prompt_lens(&raw) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        },
        Err(_) => DEFAULT_PROMPT_LENS.to_vec(),
    };
    let iters: usize = match env::var("JAMBA_BENCH_ITERS") {
        Ok(s) => match s.parse() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!("JAMBA_BENCH_ITERS: expected positive integer, got `{s}`");
                std::process::exit(2);
            }
        },
        Err(_) => DEFAULT_ITERS,
    };
    let warmup: usize = match env::var("JAMBA_BENCH_WARMUP") {
        Ok(s) => match s.parse() {
            Ok(n) => n,
            Err(_) => {
                eprintln!("JAMBA_BENCH_WARMUP: expected non-negative integer, got `{s}`");
                std::process::exit(2);
            }
        },
        Err(_) => DEFAULT_WARMUP,
    };
    let format = match env::var("JAMBA_BENCH_FORMAT") {
        Ok(s) => match Format::parse(&s) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(2);
            }
        },
        Err(_) => Format::Csv,
    };

    eprintln!("rce-jamba-bench: loading {path}…");
    let t0 = Instant::now();
    let rt = match JambaRuntime::load(&path_buf) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("rce-jamba-bench: load failed: {e}");
            std::process::exit(1);
        }
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!("rce-jamba-bench: loaded in {load_ms:.0} ms");

    let mut runs: Vec<Run> = Vec::with_capacity(prompt_lens.len() * iters);
    for &prompt_len in &prompt_lens {
        let prompt = synth_prompt(prompt_len);
        eprintln!(
            "rce-jamba-bench: prompt_len={prompt_len} warmup={warmup} iters={iters}",
        );
        for _ in 0..warmup {
            let mut state = rt.new_state();
            let _ = rt.prefill_logits(&mut state, &prompt);
        }
        for iter in 0..iters {
            let mut state = rt.new_state();
            let t0 = Instant::now();
            let _ = rt.prefill_logits(&mut state, &prompt);
            let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let ms_per_token = wall_ms / prompt_len as f64;
            runs.push(Run { prompt_len, iter, wall_ms, ms_per_token });
            eprintln!(
                "rce-jamba-bench:   iter={iter} wall_ms={wall_ms:.1} ms/tok={ms_per_token:.2}",
            );
        }
    }

    let summaries: Vec<Summary> = prompt_lens
        .iter()
        .map(|&pl| {
            let subset: Vec<Run> = runs.iter().copied().filter(|r| r.prompt_len == pl).collect();
            summarize(pl, &subset)
        })
        .collect();

    match format {
        Format::Csv => emit_csv(&runs),
        Format::Json => emit_json(&path, load_ms, iters, warmup, &runs, &summaries),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_prompt_lens_handles_normal_cases() {
        assert_eq!(parse_prompt_lens("32").unwrap(), vec![32]);
        assert_eq!(parse_prompt_lens("32,64,128").unwrap(), vec![32, 64, 128]);
        assert_eq!(parse_prompt_lens(" 32 , 64 , 128 ").unwrap(), vec![32, 64, 128]);
        assert_eq!(parse_prompt_lens("32,,64").unwrap(), vec![32, 64]);
    }

    #[test]
    fn parse_prompt_lens_rejects_bad_input() {
        assert!(parse_prompt_lens("").is_err());
        assert!(parse_prompt_lens(",").is_err());
        assert!(parse_prompt_lens("abc").is_err());
        assert!(parse_prompt_lens("0").is_err());
        assert!(parse_prompt_lens("32,0").is_err());
    }

    #[test]
    fn format_parse_accepts_csv_json_case_insensitive() {
        assert!(matches!(Format::parse("csv").unwrap(), Format::Csv));
        assert!(matches!(Format::parse("CSV").unwrap(), Format::Csv));
        assert!(matches!(Format::parse("json").unwrap(), Format::Json));
        assert!(matches!(Format::parse("JSON").unwrap(), Format::Json));
        assert!(Format::parse("yaml").is_err());
    }

    #[test]
    fn synth_prompt_is_deterministic_and_correct_length() {
        let a = synth_prompt(64);
        let b = synth_prompt(64);
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        // All token IDs in [1, 32768] — safe for any Jamba vocab.
        for tok in &a {
            assert!(*tok > 0);
            assert!(*tok <= 0x7FFF + 1);
        }
        // Distinct enough to avoid trivial-prefix optimisation: at
        // least half the positions are unique values.
        let mut sorted = a.clone();
        sorted.sort();
        sorted.dedup();
        assert!(sorted.len() >= a.len() / 2, "prompt should be reasonably diverse");
    }

    #[test]
    fn summarize_handles_single_value() {
        let r = Run { prompt_len: 32, iter: 0, wall_ms: 100.0, ms_per_token: 3.125 };
        let s = summarize(32, std::slice::from_ref(&r));
        assert_eq!(s.prompt_len, 32);
        assert!((s.min_ms - 100.0).abs() < 1e-9);
        assert!((s.median_ms - 100.0).abs() < 1e-9);
        assert!((s.mean_ms - 100.0).abs() < 1e-9);
        assert!((s.p95_ms - 100.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_computes_known_stats() {
        // Five samples: 100, 110, 120, 130, 140 ms.
        // min=100, median=120, mean=120, p95 = ms[3] + 0.8*(ms[4]-ms[3]) = 130 + 0.8*10 = 138.
        let runs: Vec<Run> = [100.0f64, 110.0, 120.0, 130.0, 140.0]
            .iter()
            .enumerate()
            .map(|(i, &ms)| Run { prompt_len: 64, iter: i, wall_ms: ms, ms_per_token: ms / 64.0 })
            .collect();
        let s = summarize(64, &runs);
        assert!((s.min_ms - 100.0).abs() < 1e-9);
        assert!((s.median_ms - 120.0).abs() < 1e-9);
        assert!((s.mean_ms - 120.0).abs() < 1e-9);
        assert!((s.p95_ms - 138.0).abs() < 1e-9, "p95={}", s.p95_ms);
    }

    #[test]
    fn escape_json_handles_special_chars() {
        assert_eq!(escape_json("plain"), "plain");
        assert_eq!(escape_json("a\"b"), "a\\\"b");
        assert_eq!(escape_json("a\\b"), "a\\\\b");
        assert_eq!(escape_json("a\nb"), "a\\nb");
        assert_eq!(escape_json("a\tb"), "a\\tb");
        assert_eq!(escape_json("a\x01b"), "a\\u0001b");
    }
}
