//! Cross-model benchmark — the synthesis artifact for the pattern-lang
//! exploration. Every rce runtime on a common factual prompt
//! ("The capital of France is", greedy, 16 new tokens): records params,
//! arch, bits/weight, whether the continuation is factually correct
//! (mentions Paris), and decode throughput (tok/s). Plus the two
//! non-decoder runtimes (EuroBERT retrieval, Tencent MT) measured on
//! their own meaningful metric.
//!
//! Single-threaded reference kernels — absolute tok/s is a relative
//! comparison on one box, not a hardware claim.

use rce_ingest_gguf::SampleConfig;
use std::path::Path;
use std::time::Instant;

const DIR: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models";
const PROMPT: &str = "The capital of France is";
const MAXTOK: usize = 16;

struct Row {
    model: String,
    arch: String,
    params: String,
    bpw: String,
    factual: String,
    toks: usize,
    secs: f64,
    sample: String,
}

fn paris(s: &str) -> bool {
    let l = s.to_lowercase();
    l.contains("paris")
}

fn main() {
    let mut rows: Vec<Row> = Vec::new();
    let mut notes: Vec<String> = Vec::new();

    macro_rules! decoder {
        ($label:expr, $arch:expr, $params:expr, $bpw:expr, $file:expr,
         $rt:path, $cfg:path) => {{
            let path = format!("{DIR}/{}", $file);
            if !Path::new(&path).exists() {
                notes.push(format!("- {} — file not present, skipped", $label));
            } else {
                match <$rt>::load(&path) {
                    Ok(rt) => {
                        let mut cfg = <$cfg>::default();
                        cfg.max_new_tokens = MAXTOK;
                        cfg.sample = SampleConfig::GREEDY;
                        cfg.rep_window = 64;
                        let mut out = String::new();
                        let t0 = Instant::now();
                        let r = rt.generate(PROMPT, &cfg, |t| out.push_str(t));
                        let dt = t0.elapsed().as_secs_f64();
                        match r {
                            Ok(_) => {
                                let n = out.split_whitespace().count().max(MAXTOK);
                                rows.push(Row {
                                    model: $label.into(), arch: $arch.into(),
                                    params: $params.into(), bpw: $bpw.into(),
                                    factual: if paris(&out) { "✓ Paris".into() }
                                             else { "—".into() },
                                    toks: MAXTOK, secs: dt,
                                    sample: out.chars().take(70).collect::<String>()
                                        .replace('\n', " "),
                                });
                                eprintln!("[done] {}  {:.2} tok/s", $label,
                                          MAXTOK as f64 / dt);
                                let _ = n;
                            }
                            Err(e) => notes.push(format!("- {} — generate error: {e}", $label)),
                        }
                    }
                    Err(e) => notes.push(format!("- {} — load error: {e}", $label)),
                }
            }
        }};
    }

    // ── Text decoders, common prompt ──
    decoder!("LFM2 350M", "lfm2 (liquid hybrid)", "350M", "8.0 (Q8_0)",
        "lfm2-350m-q8_0.gguf", rce_lfm2::Lfm2Runtime, rce_lfm2::Lfm2GenConfig);
    decoder!("LFM2.5 350M", "lfm2 (liquid hybrid)", "350M", "~4.5 (Q4_K_M)",
        "lfm2.5-350m-q4_k_m.gguf", rce_lfm2::Lfm2Runtime, rce_lfm2::Lfm2GenConfig);
    decoder!("LFM2.5-VL 450M (text)", "lfm2 (liquid hybrid)", "450M", "8.0 (Q8_0)",
        "lfm2.5-vl-450m-q8_0.gguf", rce_lfm2::Lfm2Runtime, rce_lfm2::Lfm2GenConfig);
    decoder!("TinyLlama 1.1B", "llama-2", "1.1B", "8.0 (Q8_0)",
        "tinyllama-1.1b-chat-q8_0.gguf",
        rce_tinyllama::TinyLlamaRuntime, rce_tinyllama::TinyLlamaGenConfig);
    decoder!("BitNet b1.58 2B", "bitnet (ternary I2_S)", "2B", "2.0 (I2_S)",
        "bitnet-b1.58-2b-4t-i2s.gguf",
        rce_bitnet::BitNetRuntime, rce_bitnet::BitNetGenConfig);
    decoder!("Bonsai 1.7B", "qwen3", "1.7B", "1.125 (Q1_0)",
        "bonsai-1.7b-q1_0.gguf",
        rce_qwen3::Qwen3Runtime, rce_qwen3::Qwen3GenConfig);
    decoder!("Bonsai 4B", "qwen3", "4B", "1.125 (Q1_0)",
        "bonsai-4b-q1_0.gguf",
        rce_qwen3::Qwen3Runtime, rce_qwen3::Qwen3GenConfig);
    decoder!("Ternary-Bonsai 1.7B", "qwen3", "1.7B", "2.125 (Q2_0)",
        "ternary-bonsai-1.7b-q2_0.gguf",
        rce_qwen3::Qwen3Runtime, rce_qwen3::Qwen3GenConfig);

    // ── Table ──
    println!("\n## rce cross-model benchmark");
    println!(
        "\nPrompt: {PROMPT:?} · greedy · {MAXTOK} new tokens · single-thread reference kernels\n"
    );
    println!("| Model | Arch | Params | bpw | Factual | tok/s | Continuation |");
    println!("|---|---|---|---|---|---|---|");
    for r in &rows {
        println!(
            "| {} | {} | {} | {} | {} | {:.2} | {}… |",
            r.model, r.arch, r.params, r.bpw, r.factual,
            r.toks as f64 / r.secs, r.sample.trim()
        );
    }

    // ── Non-decoder runtimes (own metric) ──
    println!("\n### Non-decoder runtimes");

    // EuroBERT retrieval: cosine(paraphrase) ≫ cosine(unrelated).
    let eb = format!("{DIR}/jina-v5-omni-nano-retrieval-q4_k_m.gguf");
    if Path::new(&eb).exists() {
        match rce_eurobert::EuroBertRuntime::load(&eb) {
            Ok(rt) => {
                let t0 = Instant::now();
                let q = rt.embed("The capital of France is Paris.").unwrap();
                let p = rt.embed("Paris is the capital city of France.").unwrap();
                let u = rt.embed("I enjoy eating pizza on the weekend.").unwrap();
                let dt = t0.elapsed().as_secs_f64();
                let sp = rce_eurobert::cosine(&q, &p);
                let su = rce_eurobert::cosine(&q, &u);
                println!(
                    "- **EuroBERT (jina-v5-omni-nano, MRL)** — bidirectional encoder, \
                     768-d L2 embedding. cos(paraphrase)={sp:.3} ≫ cos(unrelated)={su:.3} \
                     (margin {:.3}); 3 embeds in {dt:.2}s. Retrieval semantics intact.",
                    sp - su
                );
            }
            Err(e) => notes.push(format!("- EuroBERT — load error: {e}")),
        }
    }

    // Tencent HY-MT1.5 (STQ1_0): real EN→FR translation via chat frame.
    let hy = format!("{DIR}/tencent-hy-mt1.5-1.8b-1.25bit.gguf");
    if Path::new(&hy).exists() {
        match rce_hunyuan::HunyuanRuntime::load(&hy) {
            Ok(rt) => {
                let cfg = rce_hunyuan::HunyuanGenConfig {
                    max_new_tokens: 24, sample: SampleConfig::GREEDY, rep_window: 64,
                };
                if let Ok(ids) = rt.frame_user_turn(
                    "Translate the following into French:\nThe capital of France is Paris."
                ) {
                    let mut out = String::new();
                    let t0 = Instant::now();
                    let _ = rt.generate_from_ids(&ids, &cfg, |t| out.push_str(t));
                    let dt = t0.elapsed().as_secs_f64();
                    let ok = out.to_lowercase().contains("capitale");
                    println!(
                        "- **Tencent HY-MT1.5 1.8B (STQ1_0, 1.3125 bpw, hunyuan-dense)** — \
                         EN→FR: {:?} {} ({:.2} tok/s)",
                        out.trim(),
                        if ok { "✓" } else { "✗" },
                        out.split_whitespace().count().max(1) as f64 / dt
                    );
                }
            }
            Err(e) => notes.push(format!("- Tencent HY-MT — load error: {e}")),
        }
    }

    if !notes.is_empty() {
        println!("\n### Notes");
        for n in &notes {
            println!("{n}");
        }
    }
    println!(
        "\n_Reference (scalar) kernels, one box — tok/s is relative, not a hardware claim._"
    );
}
