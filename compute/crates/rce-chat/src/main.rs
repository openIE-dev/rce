//! `rce-chat` — speech-in → text-answer chat CLI on rce.
//!
//! Pipeline:
//!   1. wav2vec2-base-960h transcribes the WAV file (any SR, mono PCM-16).
//!   2. The transcription is wrapped in a Qwen-style chat template
//!      `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`.
//!   3. cosmos3 Nano (Qwen3-VL-8B reasoner) greedy-generates the answer.
//!   4. The answer streams to stdout one token at a time.
//!
//! Both models are loaded at startup (~15 s for cosmos3, ~100 ms for wav2vec2).

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const COSMOS3_ROOT: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/Cosmos3-Nano";
const WAV2VEC2_MODEL: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/wav2vec2-base-960h/model.safetensors";
const MOONSHINE_DIR: &str =
    "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/moonshine";
const IM_START_ID: u32 = 151_644;
const IM_END_ID: u32 = 151_645;
const EOS_IDS: [u32; 2] = [151_643, 151_645];

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Asr { Moonshine, Wav2vec2 }

#[derive(Parser)]
#[command(
    name = "rce-chat",
    about = "Speech-in → text-answer chat (wav2vec2 ASR + Cosmos 3 Nano reasoner)",
    version = env!("CARGO_PKG_VERSION")
)]
struct Cli {
    /// Optional system prompt for the assistant.
    #[arg(long, default_value = "You are a helpful assistant.", global = true)]
    system: String,

    /// Max new tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64, global = true)]
    max_new_tokens: usize,

    /// Which ASR to use (Moonshine is faster and produces proper case + punctuation).
    #[arg(long, value_enum, default_value_t = Asr::Moonshine, global = true)]
    asr: Asr,

    /// Cosmos 3 Nano checkpoint root.
    #[arg(long, default_value = COSMOS3_ROOT, global = true)]
    cosmos3_root: PathBuf,

    /// wav2vec2 safetensors checkpoint (used when --asr=wav2vec2).
    #[arg(long, default_value = WAV2VEC2_MODEL, global = true)]
    wav2vec2_model: PathBuf,

    /// Moonshine model directory (used when --asr=moonshine).
    #[arg(long, default_value = MOONSHINE_DIR, global = true)]
    moonshine_dir: PathBuf,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Audio in → AI text answer.
    Audio {
        /// PCM-16 mono WAV (any SR; auto-resampled to 16 kHz).
        wav: PathBuf,
    },
    /// Direct text in → AI text answer (skips ASR loading).
    Text {
        prompt: String,
    },
    /// Interactive REPL — read turns from stdin; cosmos3 stays loaded between turns.
    /// `/quit` or empty line exits.  Each turn appends to the running conversation
    /// (re-tokenized + re-forwarded each step — no KV cache across turns yet).
    Repl,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if !cli.cosmos3_root.join("model.safetensors.index.json").exists() {
        return Err(anyhow!("cosmos3 model not found at {}", cli.cosmos3_root.display()));
    }

    match &cli.cmd {
        Cmd::Text { prompt } => {
            println!("USER: {prompt}");
            run_assistant(&cli, prompt)
        }
        Cmd::Audio { wav } => {
            let transcript = transcribe_audio(&cli, wav)?;
            run_assistant(&cli, &transcript)
        }
        Cmd::Repl => run_repl(&cli),
    }
}

fn run_repl(cli: &Cli) -> Result<()> {
    use std::io::BufRead;

    eprintln!("[rce-chat] loading cosmos3 (~16 GB BF16)…");
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(cli.cosmos3_root.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let m = rce_cosmos3::Cosmos3Reasoner::load(&cli.cosmos3_root).map_err(|e| anyhow!("{e}"))?;
    eprintln!("[rce-chat] cosmos3 ready in {:?}", t0.elapsed());
    // Persistent KV-cache across turns.
    let mut state = m.new_gen_state();
    let mut first_turn = true;
    let mut total_cached = 0usize;
    let mut system_prompt = cli.system.clone();
    let mut history: Vec<(String, String)> = Vec::new();   // (user, assistant) for /save
    let mut moonshine_cached: Option<rce_moonshine::Moonshine> = None;   // lazy-loaded on first /audio
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();

    print_help();

    loop {
        print!("\n> ");
        std::io::stdout().flush().ok();
        let Some(line) = lines.next() else { break; };
        let raw_input = line.map_err(|e| anyhow!("stdin: {e}"))?;
        let raw_trim = raw_input.trim();
        if raw_trim.is_empty() || raw_trim == "/quit" || raw_trim == "/q" { break; }
        if raw_trim == "/help" || raw_trim == "/h" { print_help(); continue; }
        if let Some(new_sys) = raw_trim.strip_prefix("/system ") {
            system_prompt = new_sys.trim().to_string();
            state = m.new_gen_state();
            first_turn = true;
            total_cached = 0;
            history.clear();
            eprintln!("[rce-chat] system prompt → {system_prompt:?} (context cleared)");
            continue;
        }
        if raw_trim == "/reset" || raw_trim == "/clear" {
            state = m.new_gen_state();
            first_turn = true;
            total_cached = 0;
            history.clear();
            eprintln!("[rce-chat] context cleared; next turn rebuilds from scratch");
            continue;
        }
        if raw_trim == "/cache" {
            eprintln!("[rce-chat] cache holds {total_cached} tokens, {} turns saved (system prompt: {system_prompt:?})",
                      history.len());
            continue;
        }
        if let Some(path) = raw_trim.strip_prefix("/save ") {
            let path = std::path::PathBuf::from(path.trim());
            save_history(&path, &system_prompt, &history)?;
            eprintln!("[rce-chat] saved {} turn(s) to {}", history.len(), path.display());
            continue;
        }

        // Resolve the user's turn: either typed text or `/audio <wav>` → transcribed text.
        let user_owned: String = if let Some(rest) = raw_trim.strip_prefix("/audio ") {
            let wav_path = PathBuf::from(rest.trim());
            if !wav_path.exists() {
                eprintln!("[rce-chat] audio not found: {}", wav_path.display());
                continue;
            }
            if moonshine_cached.is_none() {
                if !cli.moonshine_dir.join("model.safetensors").exists() {
                    eprintln!("[rce-chat] moonshine model dir missing — set --moonshine-dir");
                    continue;
                }
                eprintln!("[rce-chat] loading moonshine (first /audio call)…");
                let t0 = Instant::now();
                let asr = rce_moonshine::Moonshine::load(&cli.moonshine_dir).map_err(|e| anyhow!(e))?;
                eprintln!("[rce-chat] moonshine ready in {:?}", t0.elapsed());
                moonshine_cached = Some(asr);
            }
            let asr = moonshine_cached.as_ref().unwrap();
            let audio = rce_wav2vec2::read_wav_mono_16k(&wav_path).map_err(|e| anyhow!("{e}"))?;
            let t0 = Instant::now();
            let text = asr.transcribe(&audio, 256);
            eprintln!("[rce-chat] transcribed in {:?}: {text:?}", t0.elapsed());
            text
        } else {
            raw_trim.to_string()
        };
        let user = user_owned.trim();
        if user.is_empty() { continue; }

        // Build *new* ids to extend the cache with.
        //   First turn: <im_start>system\n{sys}<im_end> <im_start>user\n{q}<im_end> <im_start>assistant\n
        //   Later:        <im_start>user\n{q}<im_end> <im_start>assistant\n
        // (The previous turn's assistant <im_end> is already in cache because
        // `continue_greedy_cached_streaming` forwards the EOS before stopping.)
        let mut new_ids: Vec<u32> = Vec::new();
        if first_turn {
            push_block(&mut new_ids, &tok, &format!("system\n{system_prompt}"), true)?;
            first_turn = false;
        }
        push_block(&mut new_ids, &tok, &format!("user\n{user}"), true)?;
        push_block(&mut new_ids, &tok, "assistant\n", false)?;
        total_cached += new_ids.len();
        eprintln!("[rce-chat] +{} tokens (cache holds {} tokens)", new_ids.len(), total_cached);

        let t_prefill = Instant::now();
        let logits = m.forward_ids_into(&mut state, &new_ids);
        let dt_prefill = t_prefill.elapsed();
        eprintln!("[rce-chat] prefill {dt_prefill:?}");

        print!("ASSISTANT: ");
        std::io::stdout().flush().ok();
        let mut acc: Vec<u32> = Vec::new();
        let mut printed_len = 0usize;
        let t0 = Instant::now();
        let out_ids = m.continue_greedy_cached_streaming(
            &mut state, logits, cli.max_new_tokens, &EOS_IDS, |tid| {
                acc.push(tid);
                let full = tok.decode(&acc, false).unwrap_or_default();
                if full.len() > printed_len {
                    print!("{}", &full[printed_len..]);
                    std::io::stdout().flush().ok();
                    printed_len = full.len();
                }
                true
            });
        let dt = t0.elapsed();
        println!();
        total_cached += out_ids.len() + 1;   // +1 for the EOS just stored
        eprintln!("[rce-chat] {} tok in {dt:?} ({:.1} ms/tok); cache now {} tokens",
                  out_ids.len(),
                  if !out_ids.is_empty() { dt.as_millis() as f32 / out_ids.len() as f32 } else { 0.0 },
                  total_cached);
        let asst_text = tok.decode(&out_ids, false).unwrap_or_default();
        history.push((user.to_string(), asst_text));
    }
    eprintln!("[rce-chat] goodbye");
    Ok(())
}

fn print_help() {
    eprintln!("[rce-chat] REPL commands:");
    eprintln!("    /help  | /h               show this help");
    eprintln!("    /quit  | /q               exit (also: empty line)");
    eprintln!("    /audio <path>             transcribe a WAV file as the user message");
    eprintln!("    /system <prompt>          change the system prompt and clear the cache");
    eprintln!("    /reset | /clear           clear the KV cache; rebuild context next turn");
    eprintln!("    /cache                    print cache size + current system prompt");
    eprintln!("    /save  <path>             write the conversation transcript as Markdown");
    eprintln!("    <anything else>           treated as typed user message");
}

fn save_history(
    path: &std::path::Path,
    system_prompt: &str,
    history: &[(String, String)],
) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)
        .map_err(|e| anyhow!("create {}: {e}", path.display()))?;
    writeln!(f, "# rce-chat conversation transcript")?;
    writeln!(f, "")?;
    writeln!(f, "**System:** {system_prompt}")?;
    writeln!(f, "")?;
    for (i, (u, a)) in history.iter().enumerate() {
        writeln!(f, "## Turn {}", i + 1)?;
        writeln!(f, "")?;
        writeln!(f, "**User:** {u}")?;
        writeln!(f, "")?;
        writeln!(f, "**Assistant:** {a}")?;
        writeln!(f, "")?;
    }
    Ok(())
}

fn push_block(ids: &mut Vec<u32>, tok: &Tokenizer, block: &str, terminate: bool) -> Result<()> {
    ids.push(IM_START_ID);
    let enc = tok.encode(block, false).map_err(|e| anyhow!("encode: {e}"))?;
    ids.extend(enc.get_ids());
    if terminate { ids.push(IM_END_ID); }
    Ok(())
}

fn transcribe_audio(cli: &Cli, audio_path: &PathBuf) -> Result<String> {
    if !audio_path.exists() {
        return Err(anyhow!("audio not found at {}", audio_path.display()));
    }
    let transcript = match cli.asr {
        Asr::Moonshine => {
            if !cli.moonshine_dir.join("model.safetensors").exists() {
                return Err(anyhow!("moonshine model not found at {}", cli.moonshine_dir.display()));
            }
            eprintln!("[rce-chat] loading moonshine…");
            let t0 = Instant::now();
            let asr = rce_moonshine::Moonshine::load(&cli.moonshine_dir).map_err(|e| anyhow!(e))?;
            eprintln!("[rce-chat] moonshine ready in {:?}", t0.elapsed());
            let audio = rce_wav2vec2::read_wav_mono_16k(audio_path).map_err(|e| anyhow!("{e}"))?;
            let t0 = Instant::now();
            let text = asr.transcribe(&audio, 256);
            eprintln!("[rce-chat] transcribed in {:?}", t0.elapsed());
            text
        }
        Asr::Wav2vec2 => {
            if !cli.wav2vec2_model.exists() {
                return Err(anyhow!("wav2vec2 model not found at {}", cli.wav2vec2_model.display()));
            }
            eprintln!("[rce-chat] loading wav2vec2…");
            let t0 = Instant::now();
            let asr = rce_wav2vec2::Wav2Vec2::load(&cli.wav2vec2_model).map_err(|e| anyhow!("{e}"))?;
            eprintln!("[rce-chat] wav2vec2 ready in {:?}", t0.elapsed());
            let t0 = Instant::now();
            let raw = asr.transcribe_wav(audio_path).map_err(|e| anyhow!("{e}"))?;
            eprintln!("[rce-chat] transcribed in {:?}", t0.elapsed());
            format_transcript(&raw)
        }
    };
    println!("USER (audio → text): {transcript}");
    Ok(transcript)
}

fn run_assistant(cli: &Cli, transcript: &str) -> Result<()> {
    eprintln!("[rce-chat] loading cosmos3 (~16 GB BF16)…");
    let t0 = Instant::now();
    let tok = Tokenizer::from_file(cli.cosmos3_root.join("tokenizer.json"))
        .map_err(|e| anyhow!("tokenizer: {e}"))?;
    let m = rce_cosmos3::Cosmos3Reasoner::load(&cli.cosmos3_root).map_err(|e| anyhow!("{e}"))?;
    eprintln!("[rce-chat] cosmos3 ready in {:?}", t0.elapsed());

    let mut ids: Vec<u32> = Vec::new();
    let sys_block = format!("system\n{}", cli.system);
    let user_block = format!("user\n{transcript}");
    let asst_block = "assistant\n";
    for (block, terminate) in [
        (sys_block.as_str(),  true),
        (user_block.as_str(), true),
        (asst_block,          false),
    ] {
        ids.push(IM_START_ID);
        let enc = tok.encode(block, false).map_err(|e| anyhow!("encode: {e}"))?;
        ids.extend(enc.get_ids());
        if terminate { ids.push(IM_END_ID); }
    }
    eprintln!("[rce-chat] prompt: {} tokens", ids.len());

    print!("ASSISTANT: ");
    std::io::stdout().flush().ok();
    let mut acc: Vec<u32> = Vec::new();
    let mut printed_len = 0usize;
    let t0 = Instant::now();
    let out_ids = m.generate_greedy_cached_streaming(&ids, cli.max_new_tokens, &EOS_IDS, |tid| {
        acc.push(tid);
        let full = tok.decode(&acc, false).unwrap_or_default();
        if full.len() > printed_len {
            print!("{}", &full[printed_len..]);
            std::io::stdout().flush().ok();
            printed_len = full.len();
        }
        true
    });
    let dt = t0.elapsed();
    println!();
    eprintln!("[rce-chat] generated {} tok in {dt:?} ({:.1} ms/token)",
              out_ids.len(),
              if !out_ids.is_empty() { dt.as_millis() as f32 / out_ids.len() as f32 } else { 0.0 });
    Ok(())
}

/// wav2vec2-base-960h outputs UPPERCASE word-separated text; clean it to a
/// normal-looking sentence for the LLM prompt.
fn format_transcript(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.is_empty() { return String::new(); }
    // Lowercase everything, then capitalize the first letter, add a period if missing.
    let lower: String = trimmed.chars().map(|c| c.to_ascii_lowercase()).collect();
    let mut chars = lower.chars();
    let first = chars.next().unwrap().to_ascii_uppercase();
    let rest: String = chars.collect();
    let mut out: String = std::iter::once(first).chain(rest.chars()).collect();
    if !out.ends_with(|c: char| c == '.' || c == '?' || c == '!') {
        out.push('?');   // the audio is most often a question
    }
    out
}
