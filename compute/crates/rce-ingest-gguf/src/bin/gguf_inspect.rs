//! gguf-inspect — mmap a GGUF file, parse the header, and print a
//! coverage report: architecture, key arch params, tensor count,
//! dtype histogram, total tensor bytes, and a name-pattern fingerprint
//! that lets us spot which architecture-family blocks each model
//! exposes (Llama-style `attn_q/k/v/o`, Mamba-style `ssm_in/conv1d/x`,
//! etc).
//!
//! Layer-12 dogfood support: this is the ground-truth "what loads
//! cleanly?" check before committing to a `ForwardPass` build for
//! any specific architecture. Run it across every GGUF in the model
//! library and produce one report per file.

use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::process::ExitCode;

use rce_ingest_gguf::parser::{GgufDtype, GgufFile, GgufValue};
use rce_ingest_gguf::quant::quant_info;
use memmap2::Mmap;

fn main() -> ExitCode {
    let raw: Vec<String> = env::args().skip(1).collect();
    let (filter, tokens, paths): (Option<String>, Vec<i32>, Vec<String>) = {
        let mut filter = None;
        let mut tokens: Vec<i32> = Vec::new();
        let mut paths = Vec::new();
        let mut iter = raw.into_iter();
        while let Some(a) = iter.next() {
            if a == "--detail" {
                filter = iter.next();
            } else if a == "--tokens" {
                if let Some(list) = iter.next() {
                    for s in list.split(',') {
                        if let Ok(n) = s.parse::<i32>() {
                            tokens.push(n);
                        }
                    }
                }
            } else {
                paths.push(a);
            }
        }
        (filter, tokens, paths)
    };
    if paths.is_empty() {
        eprintln!("usage: gguf-inspect [--detail <name-prefix>] <path.gguf> [<path.gguf> ...]");
        return ExitCode::from(2);
    }

    let mut any_failed = false;
    for arg in &paths {
        let path = PathBuf::from(arg);
        match inspect_one(&path, filter.as_deref(), &tokens) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("✗ {}: {e}", path.display());
                any_failed = true;
            }
        }
        println!();
    }

    if any_failed { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

fn inspect_one(path: &PathBuf, filter: Option<&str>, tokens: &[i32]) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let file_len = file.metadata().map_err(|e| format!("metadata: {e}"))?.len();

    // SAFETY: mmap is unsafe because the file could be modified by another
    // process while we're reading it. We're inspecting read-only model
    // files so the standard caveat is acceptable.
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
    let data: &[u8] = &mmap;

    let gguf = GgufFile::parse(data).map_err(|e| format!("parse: {e}"))?;

    if !tokens.is_empty() {
        // Decode token IDs from the vocab metadata.
        let vocab_kv = gguf.metadata.iter().find(|kv| kv.key == "tokenizer.ggml.tokens");
        match vocab_kv.map(|kv| &kv.value) {
            Some(GgufValue::Array(arr)) => {
                println!("══ {} (decoding {} token ids) ══", path.display(), tokens.len());
                for tok in tokens {
                    let raw = if (*tok as usize) < arr.len() {
                        match &arr[*tok as usize] {
                            GgufValue::String(s) => s.clone(),
                            other => format!("(non-string: {other:?})"),
                        }
                    } else {
                        "(OOV)".to_string()
                    };
                    let bytes: Vec<u8> = raw.bytes().collect();
                    println!("  {:>6}: raw={raw:?}  bytes={:02x?}", tok, &bytes[..bytes.len().min(40)]);
                }
            }
            _ => println!("⚠ no tokenizer.ggml.tokens metadata"),
        }
        return Ok(());
    }
    if let Some(prefix) = filter {
        println!("══ {} (detail: tensors+metadata matching `{prefix}`) ══", path.display());
        println!("  data_offset: {}, alignment: {}", gguf.data_offset, gguf.alignment);
        for t in &gguf.tensors {
            if t.name.contains(prefix) {
                println!(
                    "  T  {:<48}  shape={:?}  dtype={:?}",
                    t.name, t.dims, t.dtype
                );
            }
        }
        for kv in &gguf.metadata {
            if kv.key.contains(prefix) {
                let summary = match &kv.value {
                    GgufValue::Array(arr) => format!("array[{}]", arr.len()),
                    other => fmt_value(other),
                };
                println!("  M  {:<48}  {summary}", kv.key);
            }
        }
    } else {
        print_report(path, file_len, &gguf);
    }
    Ok(())
}

fn print_report(path: &PathBuf, file_len: u64, gguf: &GgufFile) {
    println!("══ {} ══", path.display());
    println!("  file size       : {}", fmt_bytes(file_len));
    println!("  GGUF version    : {}", gguf.version);
    println!(
        "  architecture    : {}",
        gguf.architecture().unwrap_or("(unspecified)")
    );
    println!(
        "  model name      : {}",
        gguf.model_name().unwrap_or("(unspecified)")
    );

    println!("  arch params:");
    print_kv("    n_layers      ", gguf.block_count());
    print_kv("    embedding_dim ", gguf.embedding_length());
    print_kv("    n_heads       ", gguf.head_count());
    print_kv("    n_kv_heads    ", gguf.head_count_kv());

    let arch = gguf.architecture().unwrap_or("");
    print_arch_kv(gguf, arch, "context_length");
    print_arch_kv(gguf, arch, "vocab_size");
    print_arch_kv(gguf, arch, "feed_forward_length");
    print_arch_kv(gguf, arch, "rope.dimension_count");
    print_arch_kv(gguf, arch, "rope.freq_base");
    print_arch_kv(gguf, arch, "attention.layer_norm_rms_epsilon");
    print_arch_kv(gguf, arch, "attention.sliding_window");

    // ── Tensor summary ─────────────────────────────────────────
    let n_tensors = gguf.tensors.len();
    let mut dtype_count: BTreeMap<String, u64> = BTreeMap::new();
    let mut dtype_bytes: BTreeMap<String, u64> = BTreeMap::new();
    let mut total_bytes = 0u64;
    let mut unsupported_dtypes: BTreeMap<String, u64> = BTreeMap::new();

    for t in &gguf.tensors {
        let info = quant_info(t.dtype);
        let bytes = tensor_byte_size(t.numel(), t.dtype);
        let name = info.rce_type.to_string();
        *dtype_count.entry(name.clone()).or_default() += 1;
        *dtype_bytes.entry(name).or_default() += bytes;
        total_bytes += bytes;

        if let GgufDtype::Unknown(v) = t.dtype {
            *unsupported_dtypes
                .entry(format!("Unknown({v})"))
                .or_default() += 1;
        }
    }

    println!("  tensors         : {n_tensors}");
    println!("  total tensor bytes: {}", fmt_bytes(total_bytes));
    println!("  dtype histogram :");
    for (name, count) in &dtype_count {
        let bytes = dtype_bytes.get(name).copied().unwrap_or(0);
        println!(
            "    {:<6}  ×{:>5}  ({})",
            name,
            count,
            fmt_bytes(bytes)
        );
    }
    if !unsupported_dtypes.is_empty() {
        println!("  ⚠ unsupported dtypes:");
        for (k, v) in &unsupported_dtypes {
            println!("    {k}  ×{v}");
        }
    }

    // ── Tensor-name fingerprint ────────────────────────────────
    // Collapse `blk.<n>.<rest>` to `blk.*.<rest>` so we see one
    // entry per tensor *family* rather than per layer.
    let mut family: BTreeMap<String, u64> = BTreeMap::new();
    for t in &gguf.tensors {
        let key = collapse_blk_index(&t.name);
        *family.entry(key).or_default() += 1;
    }
    println!("  tensor families ({}):", family.len());
    for (k, v) in &family {
        println!("    {:<48} ×{}", k, v);
    }

    // ── Architecture verdict ───────────────────────────────────
    print_verdict(&family, arch);
}

fn print_kv<T: std::fmt::Display>(label: &str, val: Option<T>) {
    match val {
        Some(v) => println!("{label}: {v}"),
        None => println!("{label}: (missing)"),
    }
}

fn print_arch_kv(gguf: &GgufFile, arch: &str, suffix: &str) {
    if arch.is_empty() {
        return;
    }
    let key = format!("{arch}.{suffix}");
    if let Some(v) = gguf.get_metadata(&key) {
        println!("    {:<14}: {}", suffix, fmt_value(v));
    }
}

fn fmt_value(v: &GgufValue) -> String {
    match v {
        GgufValue::U8(x) => x.to_string(),
        GgufValue::I8(x) => x.to_string(),
        GgufValue::U16(x) => x.to_string(),
        GgufValue::I16(x) => x.to_string(),
        GgufValue::U32(x) => x.to_string(),
        GgufValue::I32(x) => x.to_string(),
        GgufValue::F32(x) => format!("{x}"),
        GgufValue::Bool(x) => x.to_string(),
        GgufValue::String(s) => format!("{s:?}"),
        GgufValue::U64(x) => x.to_string(),
        GgufValue::I64(x) => x.to_string(),
        GgufValue::F64(x) => format!("{x}"),
        GgufValue::Array(arr) => format!("array[{}]", arr.len()),
    }
}

/// Calculate the on-disk byte size of a tensor with `numel` elements
/// of the given dtype. For block-quantized types, rounds up to whole
/// blocks. Returns 0 for `Unknown`.
fn tensor_byte_size(numel: u64, dtype: GgufDtype) -> u64 {
    let info = quant_info(dtype);
    if matches!(dtype, GgufDtype::Unknown(_)) {
        return 0;
    }
    if info.block_size <= 1 {
        // Per-element types (F32, F16, BF16, F64, integer types).
        let bits = info.bits_per_weight as u64;
        return numel * bits / 8;
    }
    // Block-quantized: ceil(numel / block_size) * block_bytes.
    // block_bytes = bits_per_weight * block_size / 8.
    let block_size = info.block_size as u64;
    let n_blocks = numel.div_ceil(block_size);
    let block_bytes = (info.bits_per_weight * info.block_size as f32 / 8.0).round() as u64;
    n_blocks * block_bytes
}

fn collapse_blk_index(name: &str) -> String {
    // `blk.0.attn_q.weight` → `blk.*.attn_q.weight`
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            return format!("blk.*.{}", &rest[dot + 1..]);
        }
    }
    name.to_string()
}

fn fmt_bytes(b: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if b >= GB {
        format!("{:.2} GB", b as f64 / GB as f64)
    } else if b >= MB {
        format!("{:.2} MB", b as f64 / MB as f64)
    } else if b >= KB {
        format!("{:.2} KB", b as f64 / KB as f64)
    } else {
        format!("{b} B")
    }
}

/// Print a verdict on architecture-family fit based on the tensor
/// fingerprint. Loose pattern-match — names are stable across the
/// llama.cpp ecosystem so this catches the common families.
fn print_verdict(family: &BTreeMap<String, u64>, arch: &str) {
    let has = |key: &str| family.keys().any(|k| k.contains(key));

    let attn = has("attn_q") && has("attn_k") && has("attn_v") && has("attn_output");
    let ssm = has("ssm_in") || has("ssm_conv1d") || has("ssm_x") || has("ssm_dt");
    let ffn = has("ffn_gate") || has("ffn_up") || has("ffn_down");
    let moe = has("ffn_gate_exps") || has("ffn_up_exps") || has("ffn_down_exps");
    let rope = has("rope_freqs");

    println!("  verdict:");
    let arch_label = if arch.is_empty() { "(?)" } else { arch };
    println!("    architecture-tag = {arch_label}");
    println!(
        "    attention QKV/O  = {}",
        if attn { "✓ Llama-style" } else { "—" }
    );
    println!("    SSM blocks       = {}", if ssm { "✓ Mamba-style" } else { "—" });
    println!(
        "    FFN              = {}",
        if moe {
            "✓ MoE (per-expert weights)"
        } else if ffn {
            "✓ dense gated"
        } else {
            "—"
        }
    );
    println!("    RoPE             = {}", if rope { "✓" } else { "—" });

    let dogfood_fit = if attn && !ssm && !moe {
        "✓ STRAIGHT FIT for inv-ssi-inference Llama-shaped ForwardPass + rce-dispatch attention"
    } else if attn && ssm {
        "△ HYBRID — attention dogfood works, but Mamba blocks need separate impl"
    } else if attn && moe {
        "△ MoE TRANSFORMER — attention works, but per-expert dispatch is its own milestone"
    } else if ssm && !attn {
        "✗ PURE SSM — does NOT exercise the attention dogfood; pick a different model first"
    } else {
        "? UNCLEAR — manual inspection required"
    };
    println!("    layer-12 fit     : {dogfood_fit}");
}
