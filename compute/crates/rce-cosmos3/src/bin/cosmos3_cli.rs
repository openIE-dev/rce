//! `cosmos3-cli` — interactive CLI for the NVIDIA Cosmos 3 Nano reasoner on rce.
//!
//! Two modes:
//!   - `text "<prompt>"`              — text continuation
//!   - `describe <image> [prompt]`    — multimodal description
//!
//! Reasoner runs locally on CPU with KV cache. Default Cosmos 3 Nano path is
//! `/Volumes/macos_4TB_external_1/.../Cosmos3-Nano` — override with
//! `--model-root <dir>`.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use rce_cosmos3::{
    image_preproc::preprocess_image_file,
    vision::VisionEncoder,
    Cosmos3Reasoner,
};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const DEFAULT_MODEL_ROOT: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/Cosmos3-Nano";
const IMAGE_PAD_ID: u32 = 151_655;
const VISION_START_ID: u32 = 151_652;
const VISION_END_ID: u32 = 151_653;
const IM_START_ID: u32 = 151_644;
const IM_END_ID: u32 = 151_645;
const EOS_IDS: [u32; 2] = [151_643, 151_645];

#[derive(Parser)]
#[command(
    name = "cosmos3-cli",
    about = "NVIDIA Cosmos 3 Nano (8B reasoner) — on rce",
    version = env!("CARGO_PKG_VERSION")
)]
struct Cli {
    /// Path to the Cosmos 3 Nano checkpoint (folder with `model.safetensors.index.json`).
    #[arg(long, default_value = DEFAULT_MODEL_ROOT)]
    model_root: PathBuf,

    /// Max new tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    max_new_tokens: usize,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Text continuation.
    Text {
        /// The prompt to extend.
        prompt: String,
    },
    /// Describe an image (default question: "Describe this image.").
    Describe {
        /// Path to the image (PNG / JPEG).
        image: PathBuf,
        /// Optional question after the image.
        #[arg(default_value = "Describe this image.")]
        question: String,
    },
}

fn load_tokenizer(root: &std::path::Path) -> Result<Tokenizer> {
    let p = root.join("tokenizer.json");
    Tokenizer::from_file(&p).map_err(|e| anyhow!("load {}: {e}", p.display()))
}

fn encode_text(tok: &Tokenizer, s: &str) -> Result<Vec<u32>> {
    Ok(tok.encode(s, false).map_err(|e| anyhow!("encode: {e}"))?.get_ids().to_vec())
}

fn decode_ids(tok: &Tokenizer, ids: &[u32]) -> String {
    tok.decode(ids, false).unwrap_or_else(|_| String::from("<decode-error>"))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = cli.model_root.clone();
    if !root.join("model.safetensors.index.json").exists() {
        return Err(anyhow!(
            "model not found at {}: missing model.safetensors.index.json",
            root.display()
        ));
    }

    eprintln!("[cosmos3-cli] loading tokenizer…");
    let tok = load_tokenizer(&root)?;

    eprintln!("[cosmos3-cli] loading reasoner (~16 GB BF16; ~10 s)…");
    let t0 = Instant::now();
    let m = Cosmos3Reasoner::load(&root).map_err(|e| anyhow!("{e}"))?;
    eprintln!("[cosmos3-cli] loaded in {:?}", t0.elapsed());

    match cli.cmd {
        Cmd::Text { prompt } => run_text(&m, &tok, &prompt, cli.max_new_tokens),
        Cmd::Describe { image, question } =>
            run_describe(&m, &tok, &root, &image, &question, cli.max_new_tokens),
    }
}

fn run_text(m: &Cosmos3Reasoner, tok: &Tokenizer, prompt: &str, max_new: usize) -> Result<()> {
    let ids = encode_text(tok, prompt)?;
    eprintln!("[cosmos3-cli] prompt: {} tokens", ids.len());

    print!("{prompt}");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut acc: Vec<u32> = Vec::new();
    let mut printed_len = 0usize;
    let out_ids = m.generate_greedy_cached_streaming(&ids, max_new, &EOS_IDS, |tid| {
        acc.push(tid);
        let full = decode_ids(tok, &acc);
        if full.len() > printed_len {
            print!("{}", &full[printed_len..]);
            std::io::stdout().flush().ok();
            printed_len = full.len();
        }
        true
    });
    let dt = t0.elapsed();
    println!();
    eprintln!("[cosmos3-cli] generated {} tok in {dt:?} ({:.1} ms/token)",
              out_ids.len(),
              if !out_ids.is_empty() { dt.as_millis() as f32 / out_ids.len() as f32 } else { 0.0 });
    Ok(())
}

fn run_describe(
    m: &Cosmos3Reasoner,
    tok: &Tokenizer,
    root: &std::path::Path,
    image: &std::path::Path,
    question: &str,
    max_new: usize,
) -> Result<()> {
    eprintln!("[cosmos3-cli] preprocessing image {}", image.display());
    let (patches, (gt, gh, gw)) = preprocess_image_file(image)
        .map_err(|e| anyhow!("preprocess: {e}"))?;
    let merged_h = gh / 2;
    let merged_w = gw / 2;
    let n_merged = gt * merged_h * merged_w;
    eprintln!("[cosmos3-cli] image grid = ({gt},{gh},{gw}) → {} patches → {n_merged} merged tokens",
              patches.len() / 1536);

    eprintln!("[cosmos3-cli] loading vision encoder…");
    let vision_path = root.join("vision_encoder/model.safetensors");
    let vis = VisionEncoder::load(&vision_path).map_err(|e| anyhow!("{e}"))
        .context("vision encoder load")?;

    eprintln!("[cosmos3-cli] running vision encoder…");
    let t_v = Instant::now();
    let (last_vit, deepstack) = vis.forward(&patches, (gt, gh, gw));
    let merged = vis.merge(&last_vit);
    eprintln!("[cosmos3-cli] vision forward in {:?}", t_v.elapsed());

    // Build chat-format prompt:
    //   <|im_start|>user
    //   <|vision_start|>{N×<|image_pad|>}<|vision_end|>{question}<|im_end|>
    //   <|im_start|>assistant
    //
    // We hand-stitch the special tokens to keep this independent of the
    // tokenizer's chat-template behavior.
    let user_text = format!("\nuser\n");
    let mid_text  = format!("{question}");
    let asst_text = format!("\nassistant\n");

    let mut ids: Vec<u32> = Vec::new();
    ids.push(IM_START_ID);
    ids.extend(encode_text(tok, &user_text)?);
    ids.push(VISION_START_ID);
    for _ in 0..n_merged { ids.push(IMAGE_PAD_ID); }
    ids.push(VISION_END_ID);
    ids.extend(encode_text(tok, &mid_text)?);
    ids.push(IM_END_ID);
    ids.push(IM_START_ID);
    ids.extend(encode_text(tok, &asst_text)?);
    eprintln!("[cosmos3-cli] prompt: {} tokens ({} text + {} image_pad)",
              ids.len(), ids.len() - n_merged, n_merged);

    let (embeds, mask) = m.build_multimodal_inputs(&ids, IMAGE_PAD_ID, &merged);

    eprintln!("[cosmos3-cli] generating…");
    println!("Image: {}", image.display());
    println!("Q: {question}");
    print!("A: ");
    std::io::stdout().flush().ok();
    let t0 = Instant::now();
    let mut acc: Vec<u32> = Vec::new();
    let mut printed_len = 0usize;
    let out_ids = m.multimodal_generate_greedy_cached_streaming(
        embeds, &mask, &deepstack, max_new, &EOS_IDS, |tid| {
            acc.push(tid);
            let full = decode_ids(tok, &acc);
            if full.len() > printed_len {
                print!("{}", &full[printed_len..]);
                std::io::stdout().flush().ok();
                printed_len = full.len();
            }
            true
        });
    let dt = t0.elapsed();
    println!();
    eprintln!("[cosmos3-cli] generated {} tok in {dt:?} ({:.1} ms/token)",
              out_ids.len(),
              if !out_ids.is_empty() { dt.as_millis() as f32 / out_ids.len() as f32 } else { 0.0 });
    Ok(())
}
