//! `rce-serve` — OpenAI-compatible HTTP front-end for the rce
//! Jamba runtime.
//!
//! Endpoints:
//! - `POST /v1/chat/completions` — streaming (`"stream": true`, SSE) or
//!   non-streaming. Accepts the usual `model`, `messages`,
//!   `max_tokens`, `temperature`, `top_p`, `seed`, plus an extension
//!   field `thinking_prefix` (default true) that toggles the
//!   Jamba-Reasoning thinking preamble.
//! - `GET /v1/models` — lists the single loaded model.
//! - `GET /health` — liveness probe.
//!
//! Every token goes through `rce_jamba::JambaRuntime::chat`, i.e.
//! rce `OpDispatch` dispatch on real GGUF weights: matmul→BLAS,
//! conv1d→Conv1dCausalLeaf, SSM→SsmScanMamba1Leaf, attention→
//! AttentionLeaf, silu/rms_norm/split/swiglu→nn-prims.
//!
//! Usage: `rce-serve [--model <gguf>] [--addr 127.0.0.1:8080]`
//! (model also via `$RCE_JAMBA_GGUF`). Inference is CPU-bound and
//! synchronous (~6 s/token on M3 Ultra) — requests run on
//! `spawn_blocking` so the async runtime stays responsive, but the
//! runtime is single-flighted (one inference at a time).

use axum::{
    extract::State,
    response::{sse::Event, IntoResponse, Sse},
    routing::{get, post},
    Json, Router,
};
use rce_bamba::{BambaGenConfig, BambaRuntime};
use rce_ingest_gguf::{GgufFile, GgufValue, SampleConfig};
use rce_jamba::{GenConfig as JambaGenConfig, JambaRuntime};
use rce_mamba1::{Mamba1GenConfig, Mamba1Runtime};
use rce_mamba2::{Mamba2GenConfig, Mamba2Runtime};
use rce_nemotron::{NemotronGenConfig, NemotronRuntime};
use futures::stream::{self, Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

// ── Shared state ───────────────────────────────────────────────────

struct AppState {
    /// The loaded runtime, in one of three flavors. Each runtime owns
    /// f32 weight buffers + parsed metadata; the OpDispatch leaves are
    /// created per forward pass. Each chat call allocates its own
    /// recurrent state so concurrent requests don't share it — they
    /// just compete for CPU.
    rt: AnyRuntime,
    model_id: String,
    arch: &'static str,
}

/// Multi-arch runtime dispatch. The three variants are heterogeneous —
/// different state types, different chat-template flavors. The HTTP
/// handler picks the right one via `match`.
enum AnyRuntime {
    Jamba(Arc<JambaRuntime>),
    Nemotron(Arc<NemotronRuntime>),
    Mamba2(Arc<Mamba2Runtime>),
    Mamba1(Arc<Mamba1Runtime>),
    Bamba(Arc<BambaRuntime>),
}

impl AnyRuntime {
    fn detect_and_load(path: &str) -> Result<(Self, &'static str), String> {
        // Peek at `general.architecture` without committing to a runtime.
        let file = std::fs::File::open(path).map_err(|e| format!("open: {e}"))?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
        let gguf = GgufFile::parse(&mmap).map_err(|e| format!("parse: {e:?}"))?;
        let arch = gguf
            .metadata
            .iter()
            .find(|kv| kv.key == "general.architecture")
            .and_then(|kv| match &kv.value {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .ok_or("missing general.architecture")?
            .to_string();
        drop(mmap);

        if arch.starts_with("nemotron_h") {
            let rt = NemotronRuntime::load(path).map_err(|e| format!("nemotron load: {e}"))?;
            Ok((AnyRuntime::Nemotron(Arc::new(rt)), "nemotron_h"))
        } else if arch == "mamba2" {
            let rt = Mamba2Runtime::load(path).map_err(|e| format!("mamba2 load: {e}"))?;
            Ok((AnyRuntime::Mamba2(Arc::new(rt)), "mamba2"))
        } else if arch == "mamba" || arch == "falcon_mamba" {
            let rt = Mamba1Runtime::load(path).map_err(|e| format!("mamba1 load: {e}"))?;
            Ok((AnyRuntime::Mamba1(Arc::new(rt)), "mamba"))
        } else if arch == "bamba" {
            let rt = BambaRuntime::load(path).map_err(|e| format!("bamba load: {e}"))?;
            Ok((AnyRuntime::Bamba(Arc::new(rt)), "bamba"))
        } else if arch == "jamba" {
            let rt = JambaRuntime::load(path).map_err(|e| format!("jamba load: {e}"))?;
            Ok((AnyRuntime::Jamba(Arc::new(rt)), "jamba"))
        } else {
            Err(format!("unsupported arch: {arch} (need jamba / nemotron_h* / mamba / mamba2 / bamba)"))
        }
    }
}

// ── OpenAI wire types ──────────────────────────────────────────────

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    /// Extension: toggle the Jamba thinking prefix (default true).
    #[serde(default = "default_true")]
    thinking_prefix: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ChunkChoice {
    index: usize,
    delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct ChatCompletionChunk {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChunkChoice>,
}

#[derive(Serialize)]
struct RespMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct RespChoice {
    index: usize,
    message: RespMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<RespChoice>,
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelInfo>,
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn req_id() -> String {
    format!("chatcmpl-{:x}", now_unix().wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(std::process::id() as u64))
}

fn sample_config_for(req: &ChatCompletionRequest) -> SampleConfig {
    let mut sample = SampleConfig::chat(req.seed.unwrap_or(0xC0DE_F00D_2026));
    if let Some(t) = req.temperature {
        sample.temperature = t;
    }
    if let Some(p) = req.top_p {
        sample.top_p = Some(p);
    }
    sample
}

fn jamba_gen_config(req: &ChatCompletionRequest) -> JambaGenConfig {
    JambaGenConfig {
        max_new_tokens: req.max_tokens.unwrap_or(256).min(4096),
        sample: sample_config_for(req),
        rep_window: 64,
        thinking_prefix: req.thinking_prefix,
        batched_prefill: true,
    }
}

fn nemotron_gen_config(req: &ChatCompletionRequest) -> NemotronGenConfig {
    NemotronGenConfig {
        max_new_tokens: req.max_tokens.unwrap_or(256).min(4096),
        sample: sample_config_for(req),
        rep_window: 64,
        enable_thinking: req.thinking_prefix,
    }
}

fn mamba2_gen_config(req: &ChatCompletionRequest) -> Mamba2GenConfig {
    Mamba2GenConfig {
        max_new_tokens: req.max_tokens.unwrap_or(256).min(4096),
        sample: sample_config_for(req),
        rep_window: 64,
    }
}

fn mamba1_gen_config(req: &ChatCompletionRequest) -> Mamba1GenConfig {
    Mamba1GenConfig {
        max_new_tokens: req.max_tokens.unwrap_or(256).min(4096),
        sample: sample_config_for(req),
        rep_window: 64,
    }
}

fn bamba_gen_config(req: &ChatCompletionRequest) -> BambaGenConfig {
    BambaGenConfig {
        max_new_tokens: req.max_tokens.unwrap_or(256).min(4096),
        sample: sample_config_for(req),
        rep_window: 64,
    }
}

/// One-request runtime handle: holds the Arc<Runtime> and the pre-built
/// per-arch gen config, ready to ship across thread boundaries.
enum AnyRuntimeHandle {
    Jamba(Arc<JambaRuntime>, JambaGenConfig),
    Nemotron(Arc<NemotronRuntime>, NemotronGenConfig),
    Mamba2(Arc<Mamba2Runtime>, Mamba2GenConfig),
    Mamba1(Arc<Mamba1Runtime>, Mamba1GenConfig),
    Bamba(Arc<BambaRuntime>, BambaGenConfig),
}

/// Synchronously run a chat against the appropriate runtime, streaming
/// generated text via `on_token`. Returns the full decoded text on
/// success.
fn run_chat(
    handle: AnyRuntimeHandle,
    msgs_owned: &[(String, String)],
    mut on_token: impl FnMut(&str),
) -> Result<String, String> {
    match handle {
        AnyRuntimeHandle::Jamba(rt, cfg) => {
            let msgs: Vec<(&str, &str)> = msgs_owned.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
            rt.chat(&msgs, &cfg, |t| on_token(t))
        }
        AnyRuntimeHandle::Nemotron(rt, cfg) => {
            let msgs: Vec<(&str, &str)> = msgs_owned.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
            rt.chat(&msgs, &cfg, |t| on_token(t))
        }
        AnyRuntimeHandle::Mamba2(rt, cfg) => {
            let prompt = flatten_messages_for_base_lm(msgs_owned);
            rt.generate(&prompt, &cfg, |t| on_token(t))
        }
        AnyRuntimeHandle::Mamba1(rt, cfg) => {
            // Falcon3-Mamba-Instruct (and any Mamba-1 instruct GGUF with
            // ChatML special tokens) takes a proper chat template; fall
            // back to base-LM continuation if `<|im_start|>` is missing
            // (e.g. raw Mamba-1 paper checkpoints).
            let msgs: Vec<(&str, &str)> = msgs_owned.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
            match rt.chat(&msgs, &cfg, |t| on_token(t)) {
                Ok(s) => Ok(s),
                Err(e) if e.contains("ChatML") => {
                    let prompt = flatten_messages_for_base_lm(msgs_owned);
                    rt.generate(&prompt, &cfg, |t| on_token(t))
                }
                Err(e) => Err(e),
            }
        }
        AnyRuntimeHandle::Bamba(rt, cfg) => {
            // Bamba-9B-v1 is a base (pretrained) model. The Llama-3
            // ChatML template would only work on a future Bamba-Instruct.
            // For HTTP we flatten messages and run base-LM continuation;
            // when an instruct variant ships set RCE_BAMBA_CHAT=1 to
            // route to the chat template instead.
            if std::env::var("RCE_BAMBA_CHAT").is_ok() {
                let msgs: Vec<(&str, &str)> = msgs_owned.iter().map(|(r, c)| (r.as_str(), c.as_str())).collect();
                rt.chat(&msgs, &cfg, |t| on_token(t))
            } else {
                let prompt = flatten_messages_for_base_lm(msgs_owned);
                rt.generate(&prompt, &cfg, |t| on_token(t))
            }
        }
    }
}

/// Flatten OpenAI-style chat messages into a single raw prompt for base
/// LMs (Mamba-2 has no chat template). Last user turn becomes the
/// continuation prompt; preceding turns are concatenated as "Role:
/// content" lines. Imperfect for multi-turn but the only sane fallback
/// for a base-model chat endpoint.
fn flatten_messages_for_base_lm(messages: &[(String, String)]) -> String {
    let mut s = String::new();
    for (role, content) in messages {
        match role.as_str() {
            "system" => { s.push_str(content); s.push('\n'); }
            "user" => { s.push_str("User: "); s.push_str(content); s.push('\n'); }
            "assistant" => { s.push_str("Assistant: "); s.push_str(content); s.push('\n'); }
            _ => { s.push_str(content); s.push('\n'); }
        }
    }
    s.push_str("Assistant:");
    s
}

// ── Handlers ───────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(st): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelInfo {
            id: st.model_id.clone(),
            object: "model",
            created: now_unix(),
            owned_by: st.arch,
        }],
    })
}

async fn chat_completions(
    State(st): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let model = req.model.clone().unwrap_or_else(|| st.model_id.clone());
    let msgs_owned: Vec<(String, String)> =
        req.messages.iter().map(|m| (m.role.clone(), m.content.clone())).collect();

    // Clone the right arch handle for the inference thread.
    let rt_handle: AnyRuntimeHandle = match &st.rt {
        AnyRuntime::Jamba(r)    => AnyRuntimeHandle::Jamba(r.clone(),    jamba_gen_config(&req)),
        AnyRuntime::Nemotron(r) => AnyRuntimeHandle::Nemotron(r.clone(), nemotron_gen_config(&req)),
        AnyRuntime::Mamba2(r)   => AnyRuntimeHandle::Mamba2(r.clone(),   mamba2_gen_config(&req)),
        AnyRuntime::Mamba1(r)   => AnyRuntimeHandle::Mamba1(r.clone(),   mamba1_gen_config(&req)),
        AnyRuntime::Bamba(r)    => AnyRuntimeHandle::Bamba(r.clone(),    bamba_gen_config(&req)),
    };

    if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<String>();
        std::thread::spawn(move || {
            let _ = run_chat(rt_handle, &msgs_owned, |tok| {
                let _ = tx.send(tok.to_string());
            });
            // dropping tx ends the stream
        });

        let id = req_id();
        let created = now_unix();
        // First chunk announces the assistant role.
        let role_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model: model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
        };
        let id_for_content = id.clone();
        let model_for_content = model.clone();
        let content_stream = UnboundedReceiverStream::new(rx).map(move |tok| {
            let chunk = ChatCompletionChunk {
                id: id_for_content.clone(),
                object: "chat.completion.chunk",
                created,
                model: model_for_content.clone(),
                choices: vec![ChunkChoice {
                    index: 0,
                    delta: Delta { role: None, content: Some(tok) },
                    finish_reason: None,
                }],
            };
            Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&chunk).unwrap()))
        });
        let stop_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk",
            created,
            model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: None },
                finish_reason: Some("stop"),
            }],
        };
        let head = stream::once(async move {
            Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&role_chunk).unwrap()))
        });
        let tail = stream::iter([
            Ok::<Event, Infallible>(Event::default().data(serde_json::to_string(&stop_chunk).unwrap())),
            Ok::<Event, Infallible>(Event::default().data("[DONE]")),
        ]);
        let sse: Sse<_> = Sse::new(head.chain(content_stream).chain(tail));
        sse.into_response()
    } else {
        let result: Result<String, String> = tokio::task::spawn_blocking(move || {
            run_chat(rt_handle, &msgs_owned, |_| {})
        })
        .await
        .unwrap_or_else(|e| Err(format!("inference task panicked: {e}")));

        match result {
            Ok(content) => Json(ChatCompletionResponse {
                id: req_id(),
                object: "chat.completion",
                created: now_unix(),
                model,
                choices: vec![RespChoice {
                    index: 0,
                    message: RespMessage { role: "assistant", content },
                    finish_reason: "stop",
                }],
            })
            .into_response(),
            Err(e) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": { "message": e, "type": "inference_error" } })),
            )
                .into_response(),
        }
    }
}

// ── Boilerplate stream type alias (axum needs the concrete type) ────

#[allow(dead_code)]
type SseStream = std::pin::Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>;

// ── main ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let mut model_path: Option<String> = None;
    let mut addr = "127.0.0.1:8080".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" | "-m" => model_path = args.next(),
            "--addr" | "-a" => {
                if let Some(v) = args.next() {
                    addr = v;
                }
            }
            other => {
                if other.ends_with(".gguf") && model_path.is_none() {
                    model_path = Some(other.to_string());
                } else {
                    eprintln!("unknown arg: {other}");
                }
            }
        }
    }
    let model_path = model_path
        .or_else(|| std::env::var("RCE_SERVE_GGUF").ok())
        .or_else(|| std::env::var("RCE_JAMBA_GGUF").ok())  // legacy
        .unwrap_or_else(|| {
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/jamba-reasoning-3b-gguf/jamba-reasoning-3b-Q4_K_M.gguf"
                .to_string()
        });

    if !std::path::Path::new(&model_path).exists() {
        eprintln!("model not found: {model_path}");
        eprintln!("pass --model <path> or set $RCE_SERVE_GGUF");
        std::process::exit(1);
    }

    tracing::info!("loading {model_path} …");
    let t0 = std::time::Instant::now();
    let (rt, arch) = match AnyRuntime::detect_and_load(&model_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("load failed: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("loaded {arch} in {:?}", t0.elapsed());

    let model_id = std::path::Path::new(&model_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(arch)
        .to_string();

    let state = Arc::new(AppState { rt, model_id, arch });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| {
        eprintln!("bind {addr} failed: {e}");
        std::process::exit(1);
    });
    tracing::info!("rce-serve listening on http://{addr}");
    tracing::info!("  POST /v1/chat/completions   (stream + non-stream)");
    tracing::info!("  GET  /v1/models");
    tracing::info!("  GET  /health");
    axum::serve(listener, app).await.unwrap();
}
