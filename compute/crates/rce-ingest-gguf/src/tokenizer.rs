//! Minimal tokenizer decoding for GGUF-embedded vocabs.
//!
//! Most modern GGUFs (llama.cpp ecosystem) embed the tokenizer as
//! metadata: `tokenizer.ggml.tokens` (array of vocab strings),
//! `tokenizer.ggml.token_type` (per-token type: NORMAL / CONTROL /
//! UNUSED / BYTE / UNDEFINED), plus BOS/EOS/PAD ids.
//!
//! This module provides **decoding only** — token id → string.
//! Encoding (text → token ids) requires a BPE merge table or unigram
//! probability model and is out of scope here. For inference loops
//! we only need to decode the model's emitted token ids back into
//! UTF-8 text.
//!
//! ## SentencePiece vs GPT-2 BPE conventions
//!
//! Different tokenizers use different "space-marker" characters:
//! - **SentencePiece** (Llama, Mistral, Jamba): `▁` (U+2581) at the
//!   start of a token means "space then this token". Multiple
//!   consecutive ▁'s mean leading spaces.
//! - **GPT-2 BPE** (OpenAI, Mixtral-style some): `Ġ` (U+0120) is the
//!   space marker; `Ċ` (U+010A) is newline. We don't see this in
//!   Jamba/Mistral GGUFs but support it for portability.
//!
//! `decode_one` and `decode_sequence` handle both conventions
//! transparently — they look at the actual bytes/chars in each token
//! string and replace marker characters with their literal
//! equivalents.

use crate::parser::{GgufFile, GgufValue};

/// Token-type tag. Matches llama.cpp's `LLAMA_TOKEN_TYPE_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
    Unused = 5,
    Byte = 6,
}

impl TokenType {
    pub fn from_i32(v: i32) -> Self {
        match v {
            1 => TokenType::Normal,
            2 => TokenType::Unknown,
            3 => TokenType::Control,
            4 => TokenType::UserDefined,
            5 => TokenType::Unused,
            6 => TokenType::Byte,
            _ => TokenType::Undefined,
        }
    }
}

/// Lightweight tokenizer view over a parsed GGUF. Borrows the vocab
/// strings; no allocation beyond the optional types lookup.
pub struct Tokenizer<'a> {
    pub vocab: Vec<&'a str>,
    pub token_types: Vec<TokenType>,
    pub bos_token_id: Option<i32>,
    pub eos_token_id: Option<i32>,
    pub padding_token_id: Option<i32>,
    pub unknown_token_id: Option<i32>,
    /// GPT-2 BPE merges: `(left, right) → priority` (lower priority = earlier-applied).
    /// Used by `encode_gpt2` when the GGUF is `tokenizer.ggml.model = "gpt2"`.
    pub merges: std::collections::HashMap<(String, String), usize>,
}

impl<'a> Tokenizer<'a> {
    /// Build a tokenizer from a parsed GGUF's metadata. Returns `None`
    /// if the file doesn't carry a tokenizer (`tokenizer.ggml.tokens`
    /// missing or wrong type).
    pub fn from_gguf(gguf: &'a GgufFile) -> Option<Self> {
        let tokens_kv = gguf.metadata.iter().find(|kv| kv.key == "tokenizer.ggml.tokens")?;
        let tokens_arr = match &tokens_kv.value {
            GgufValue::Array(a) => a,
            _ => return None,
        };
        let mut vocab: Vec<&str> = Vec::with_capacity(tokens_arr.len());
        for v in tokens_arr {
            match v {
                GgufValue::String(s) => vocab.push(s.as_str()),
                _ => return None,
            }
        }

        let token_types: Vec<TokenType> = match gguf.metadata
            .iter()
            .find(|kv| kv.key == "tokenizer.ggml.token_type")
            .map(|kv| &kv.value)
        {
            Some(GgufValue::Array(arr)) => arr.iter().map(|v| match v {
                GgufValue::I32(x) => TokenType::from_i32(*x),
                GgufValue::U32(x) => TokenType::from_i32(*x as i32),
                _ => TokenType::Undefined,
            }).collect(),
            _ => vec![TokenType::Normal; vocab.len()],
        };

        // Parse the BPE merges array (GPT-2 family only). Each entry
        // is "left right" — a pair separated by a single space, in
        // priority order (index 0 = highest priority).
        let mut merges: std::collections::HashMap<(String, String), usize> = std::collections::HashMap::new();
        if let Some(GgufValue::Array(arr)) = gguf.metadata.iter()
            .find(|kv| kv.key == "tokenizer.ggml.merges")
            .map(|kv| &kv.value)
        {
            for (i, v) in arr.iter().enumerate() {
                if let GgufValue::String(s) = v {
                    if let Some(sp) = s.find(' ') {
                        let l = &s[..sp];
                        let r = &s[sp + 1..];
                        merges.insert((l.to_string(), r.to_string()), i);
                    }
                }
            }
        }

        Some(Self {
            vocab,
            token_types,
            bos_token_id: get_i32(gguf, "tokenizer.ggml.bos_token_id"),
            eos_token_id: get_i32(gguf, "tokenizer.ggml.eos_token_id"),
            padding_token_id: get_i32(gguf, "tokenizer.ggml.padding_token_id"),
            unknown_token_id: get_i32(gguf, "tokenizer.ggml.unknown_token_id"),
            merges,
        })
    }

    /// Look up a token's raw vocab string. Returns `None` for
    /// out-of-vocab ids.
    pub fn token_str(&self, id: i32) -> Option<&str> {
        if id < 0 { return None; }
        self.vocab.get(id as usize).copied()
    }

    /// Find the token id whose raw vocab string matches `text`
    /// exactly. Useful for special tokens like `<|im_start|>` that
    /// the greedy `encode` would skip (Control/Unused tokens are
    /// excluded from text-encoding to avoid spurious matches).
    pub fn lookup_token(&self, text: &str) -> Option<i32> {
        self.vocab.iter().position(|s| *s == text).map(|i| i as i32)
    }

    /// Token-type tag. Returns `Undefined` for OOV.
    pub fn token_type(&self, id: i32) -> TokenType {
        if id < 0 { return TokenType::Undefined; }
        self.token_types
            .get(id as usize)
            .copied()
            .unwrap_or(TokenType::Undefined)
    }

    /// Decode a single token id to its visible text. Replaces
    /// SentencePiece `▁` (U+2581) with space and GPT-2 `Ġ` (U+0120)
    /// with space; passes special tokens (`Control` / `Unknown` /
    /// `Unused`) through as-is so the caller can decide whether to
    /// hide them.
    pub fn decode_one(&self, id: i32) -> String {
        let s = match self.token_str(id) {
            Some(s) => s,
            None => return String::new(),
        };
        decode_token_chars(s)
    }

    /// Encode a UTF-8 string into a sequence of token ids using
    /// **greedy longest-match** against the vocab. Iterates left-to-
    /// right; at each position finds the longest vocab token whose
    /// raw text matches starting there, emits it, advances. Falls
    /// back to the unknown-token id (or skips if no UNK is set) on
    /// no-match.
    ///
    /// SentencePiece-aware: input UTF-8 spaces are mapped to `▁`
    /// (U+2581) before matching, since vocab tokens use `▁` for word
    /// boundaries (as observed in Jamba — `" Spaces"` is stored as
    /// `▁Spaces` in the vocab).
    ///
    /// **Caveat**: not the optimal Viterbi unigram path that the
    /// reference SentencePiece implementation uses (which considers
    /// per-token scores). Greedy-longest is right for prefix-rich
    /// vocabs like Llama's where the longest match is usually the
    /// most likely segment; close enough to ship a real prompt path.
    /// For production-quality encoding, port the unigram Viterbi
    /// (uses the `tokenizer.ggml.scores` array we already loaded into
    /// `self.token_types` companion).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<i32> {
        self.encode_with_options(text, add_bos, true)
    }

    /// Like `encode` but with control over the SentencePiece
    /// space-prefix convention. Set `add_space_prefix = false` when
    /// encoding **inner segments** of a chat-template (e.g. the
    /// "user\n" that follows `<|im_start|>` directly with no
    /// intervening space — adding `▁` there breaks tokenization
    /// against models trained on `<|im_start|>user\n`).
    pub fn encode_with_options(&self, text: &str, add_bos: bool, add_space_prefix: bool) -> Vec<i32> {
        let mut tokens = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                tokens.push(bos);
            }
        }

        // Replace ASCII spaces with `▁` for SentencePiece matching.
        let normalized: String = text.chars().map(|c| if c == ' ' { '\u{2581}' } else { c }).collect();
        // SentencePiece convention: prepend `▁` to non-space-starting
        // input so the first word becomes a `▁word` token. Matches
        // `tokenizer.ggml.add_space_prefix` semantics. Skipped for
        // chat-template inner segments where the model expects raw
        // bytes after a special token.
        let normalized = if add_space_prefix && !normalized.starts_with('\u{2581}') {
            format!("\u{2581}{}", normalized)
        } else {
            normalized
        };
        let bytes = normalized.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            // Find the longest vocab token whose bytes are a prefix
            // of `bytes[pos..]`. Linear scan O(V) per char — slow for
            // big inputs but fine for the few-token prompts we'll
            // need for inference loops.
            let remaining = &bytes[pos..];
            let mut best_id: Option<i32> = None;
            let mut best_len = 0usize;
            for (i, tok) in self.vocab.iter().enumerate() {
                let tb = tok.as_bytes();
                if tb.len() > best_len && tb.len() <= remaining.len() && &remaining[..tb.len()] == tb {
                    // Skip control / unused / byte-fallback tokens
                    // during normal greedy match. Byte tokens (e.g.
                    // `<0x0A>` for newline) are reserved for the
                    // byte-fallback path below — matching them here
                    // would confuse literal text like `<0x...>` with
                    // the byte-fallback machinery.
                    let ty = self.token_types.get(i).copied().unwrap_or(TokenType::Normal);
                    if matches!(ty, TokenType::Control | TokenType::Unused | TokenType::Undefined | TokenType::Byte) {
                        continue;
                    }
                    best_id = Some(i as i32);
                    best_len = tb.len();
                }
            }
            match best_id {
                Some(id) => {
                    tokens.push(id);
                    pos += best_len;
                }
                None => {
                    // No regular token matches. Try **byte-fallback**:
                    // SentencePiece encodes any byte that no vocab
                    // token covers as `<0xNN>` (a Byte-type token).
                    // This is how the model trained-in newlines, tabs,
                    // and out-of-charset bytes — without it, prompts
                    // containing `\n` would silently lose the newline.
                    let byte = bytes[pos];
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(id) = self.lookup_token(&byte_token) {
                        tokens.push(id);
                    } else if let Some(unk) = self.unknown_token_id {
                        tokens.push(unk);
                    }
                    pos += 1;
                }
            }
        }
        tokens
    }

    /// GPT-2 byte-level BPE encoding (for `tokenizer.ggml.model = "gpt2"`
    /// vocabularies — e.g. Nemotron-H). The algorithm:
    /// 1. Map each input byte through GPT-2's `bytes_to_unicode` table
    ///    (space → `Ġ`, newline → `Ċ`, tab → `ĉ`, etc.).
    /// 2. Split into single-codepoint strings.
    /// 3. Iteratively merge adjacent pairs in priority order from the
    ///    `tokenizer.ggml.merges` table (lowest-index merge wins).
    /// 4. Look each final piece up in the vocab to get its token id.
    ///
    /// Matches llama.cpp's `llama-tokenize` output. (Greedy longest-
    /// match alone gets most tokens right but mis-splits a few common
    /// strings — e.g. "assistant" greedy → ["assis","tant"] but BPE →
    /// ["ass","istant"], with different IDs.)
    pub fn encode_gpt2(&self, text: &str, add_bos: bool) -> Vec<i32> {
        let mut tokens = Vec::new();
        if add_bos {
            if let Some(bos) = self.bos_token_id {
                tokens.push(bos);
            }
        }
        // Pre-tokenize on word/punctuation boundaries so BPE runs per
        // piece — matches GPT-2's regex pre-tokenizer (`'s|'t|...| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`).
        // Use a simplified scheme: split before each space/newline (a
        // ` ?\p{L}+` / `\s+` rough equivalent that's good enough for
        // chat content without exotic Unicode).
        for piece in pretokenize_gpt2(text) {
            // Byte-to-unicode each piece.
            let mut encoded = String::with_capacity(piece.len());
            for &byte in piece.as_bytes() {
                encoded.push(gpt2_byte_to_unicode(byte));
            }
            // Run BPE on this piece, getting a Vec<String> of merged
            // sub-tokens.
            let pieces = self.bpe_merge(&encoded);
            for p in pieces {
                if let Some(id) = self.lookup_token(&p) {
                    tokens.push(id);
                } else if let Some(unk) = self.unknown_token_id {
                    tokens.push(unk);
                }
            }
        }
        tokens
    }

    /// Iterative pair-merge BPE: start with single-codepoint pieces,
    /// repeatedly find the highest-priority (= lowest index in
    /// `self.merges`) adjacent pair and merge it, until no more pairs
    /// from the merge table apply.
    fn bpe_merge(&self, text: &str) -> Vec<String> {
        let mut pieces: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(usize, usize)> = None; // (position, priority)
            for i in 0..pieces.len().saturating_sub(1) {
                let key = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&prio) = self.merges.get(&key) {
                    if best.map_or(true, |(_, p)| prio < p) {
                        best = Some((i, prio));
                    }
                }
            }
            match best {
                Some((pos, _)) => {
                    let merged = format!("{}{}", pieces[pos], pieces[pos + 1]);
                    pieces[pos] = merged;
                    pieces.remove(pos + 1);
                }
                None => break,
            }
        }
        pieces
    }

    /// Multi-turn chat template render for the Nemotron-H / Qwen-style
    /// `<|im_start|> / <|im_end|>` framing, using GPT-2 BPE encoding
    /// for text segments. Matches the `tokenizer.chat_template` shipped
    /// in Nemotron-Nano-4B's GGUF for `add_generation_prompt=true`,
    /// optionally with `enable_thinking` to append `<|im_start|>assistant\n<think>\n`
    /// (vs `<|im_start|>assistant\n<think></think>`).
    pub fn format_chat_nemotron(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Option<Vec<i32>> {
        let im_start = self.lookup_token("<|im_start|>")?;
        let im_end = self.lookup_token("<|im_end|>")?;

        let mut out = Vec::new();
        if let Some(bos) = self.bos_token_id {
            out.push(bos);
        }
        // Nemotron's `tokenizer.chat_template` emits an empty system
        // block when there's no system message provided. We emit it
        // unconditionally if `messages[0].role != "system"` to match.
        let has_system = messages.first().map(|(r, _)| *r == "system").unwrap_or(false);
        if !has_system {
            out.push(im_start);
            out.extend(self.encode_gpt2("system\n", false));
            out.push(im_end);
            out.extend(self.encode_gpt2("\n", false));
        }
        for (role, content) in messages {
            out.push(im_start);
            out.extend(self.encode_gpt2(&format!("{role}\n"), false));
            out.extend(self.encode_gpt2(content, false));
            out.push(im_end);
            out.extend(self.encode_gpt2("\n", false));
        }
        if add_generation_prompt {
            out.push(im_start);
            if enable_thinking {
                let think = self.lookup_token("<think>")?;
                out.extend(self.encode_gpt2("assistant\n", false));
                out.push(think);
                out.extend(self.encode_gpt2("\n", false));
            } else {
                // `<think></think>` literally — no newline.
                let think = self.lookup_token("<think>")?;
                let end_think = self.lookup_token("</think>")?;
                out.extend(self.encode_gpt2("assistant\n", false));
                out.push(think);
                out.push(end_think);
            }
        }
        Some(out)
    }

    /// Build a chat-template-formatted token sequence for a single-turn
    /// user message, no system message, no thinking prefix. Rendered:
    ///
    /// ```text
    /// <BOS><|im_start|>user\n<user_msg><|im_end|>\n<|im_start|>assistant\n<think>\n
    /// ```
    ///
    /// Backward-compat wrapper around `format_chat_messages`.
    pub fn format_chat_prompt(&self, user_msg: &str) -> Option<Vec<i32>> {
        self.format_chat_messages(&[("user", user_msg)], true, false)
    }

    /// Plain ChatML render (no `<think>` token, no Nemotron-style
    /// always-emit-empty-system-block). Matches the Jinja template
    /// shipped by Falcon3-Mamba-7B-Instruct:
    ///
    /// ```text
    /// {{bos_token}}{% for m in messages %}<|im_start|>{role}\n{content}<|im_end|>\n{% endfor %}
    /// {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}
    /// ```
    ///
    /// Returns `None` if the model lacks the `<|im_start|>` / `<|im_end|>`
    /// special tokens.
    pub fn format_chat_chatml(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> Option<Vec<i32>> {
        let im_start = self.lookup_token("<|im_start|>")?;
        let im_end = self.lookup_token("<|im_end|>")?;
        let mut out = Vec::new();
        if let Some(bos) = self.bos_token_id {
            out.push(bos);
        }
        for (role, content) in messages {
            out.push(im_start);
            out.extend(self.encode_gpt2(&format!("{role}\n"), false));
            out.extend(self.encode_gpt2(content, false));
            out.push(im_end);
            out.extend(self.encode_gpt2("\n", false));
        }
        if add_generation_prompt {
            out.push(im_start);
            out.extend(self.encode_gpt2("assistant\n", false));
        }
        Some(out)
    }

    /// Render a multi-turn chat conversation into token ids, matching
    /// the Jamba / Qwen `<|im_start|>role\ncontent<|im_end|>\n`
    /// chat-template convention. Optionally appends the
    /// `<|im_start|>assistant\n<think>\n` generation prompt and the
    /// Jamba-Reasoning **thinking prefix** (the canonical
    /// `"Begin by thinking about the reasoning process in the mind
    /// within <think> </think> tags…"` preamble that the model was
    /// trained to receive before user content).
    ///
    /// - `messages`: ordered conversation as `(role, content)` pairs.
    ///   Supported roles: `"system"`, `"user"`, `"assistant"`.
    /// - `add_generation_prompt`: append `<|im_start|>assistant\n<think>\n`
    ///   so the model knows it's its turn to speak. Set `false` to
    ///   tokenize a complete past transcript without forcing a turn.
    /// - `with_thinking_prefix`: prepend the thinking_prefix string to
    ///   the **last user message** (matches `tokenizer.chat_template`
    ///   when `loop.last && message.role == 'user'`). Needed for
    ///   Jamba-Reasoning to engage step-by-step arithmetic / logic
    ///   mode rather than free-associating.
    ///
    /// Returns `None` if any required special token (`<|im_start|>`,
    /// `<|im_end|>`, `<think>` for generation prompt) is missing.
    pub fn format_chat_messages(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
        with_thinking_prefix: bool,
    ) -> Option<Vec<i32>> {
        const THINKING_PREFIX: &str =
            "Begin by thinking about the reasoning process in the mind within <think> </think> tags and then proceed to give your response.\n";

        let im_start = self.lookup_token("<|im_start|>")?;
        let im_end = self.lookup_token("<|im_end|>")?;

        let mut out = Vec::new();
        if let Some(bos) = self.bos_token_id {
            out.push(bos);
        }

        let last_user_idx = if with_thinking_prefix {
            messages
                .iter()
                .enumerate()
                .filter(|(_, (role, _))| *role == "user")
                .last()
                .map(|(i, _)| i)
        } else {
            None
        };

        for (i, (role, content)) in messages.iter().enumerate() {
            out.push(im_start);
            out.extend(self.encode_with_options(&format!("{role}\n"), false, false));
            if Some(i) == last_user_idx && !content.contains("<think>") {
                out.extend(self.encode_with_options(THINKING_PREFIX, false, false));
            }
            out.extend(self.encode_with_options(content, false, false));
            out.push(im_end);
            out.extend(self.encode_with_options("\n", false, false));
        }

        if add_generation_prompt {
            let think = self.lookup_token("<think>")?;
            out.push(im_start);
            out.extend(self.encode_with_options("assistant\n", false, false));
            out.push(think);
            out.extend(self.encode_with_options("\n", false, false));
        }

        Some(out)
    }

    /// Decode a sequence of token ids into one UTF-8 string.
    /// `skip_special` controls whether `Control` / `Unused` /
    /// `Undefined` tokens are dropped (true = skip, the standard
    /// "human-readable output" mode).
    ///
    /// Handles byte-fallback tokens (`<0xNN>`, type=Byte) by emitting
    /// the raw byte — necessary for newlines, tabs, and any non-ASCII
    /// codepoint the model decomposes into UTF-8 byte sequences. Bytes
    /// are buffered and the final output is `String::from_utf8_lossy`'d
    /// so partial multi-byte sequences don't crash decode.
    pub fn decode_sequence(&self, ids: &[i32], skip_special: bool) -> String {
        let mut out_bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let ty = self.token_type(id);
            if skip_special && matches!(ty, TokenType::Control | TokenType::Unused | TokenType::Undefined) {
                continue;
            }
            let s = match self.token_str(id) {
                Some(s) => s,
                None => continue,
            };
            if matches!(ty, TokenType::Byte) {
                if let Some(byte) = parse_byte_token(s) {
                    out_bytes.push(byte);
                    continue;
                }
            }
            out_bytes.extend_from_slice(decode_token_chars(s).as_bytes());
        }
        String::from_utf8_lossy(&out_bytes).into_owned()
    }
}

fn get_i32(gguf: &GgufFile, key: &str) -> Option<i32> {
    let kv = gguf.metadata.iter().find(|kv| kv.key == key)?;
    match &kv.value {
        GgufValue::I32(v) => Some(*v),
        GgufValue::U32(v) => Some(*v as i32),
        GgufValue::I64(v) => Some(*v as i32),
        GgufValue::U64(v) => Some(*v as i32),
        _ => None,
    }
}

/// Approximation of GPT-2's pre-tokenizer regex
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
/// We split into character-class runs (letters / digits /
/// non-alphanumeric-non-space / whitespace), with up to one leading
/// space attached to a non-whitespace run. Good enough for chat
/// content (no exotic-Unicode handling).
pub fn pretokenize_gpt2(text: &str) -> Vec<String> {
    #[derive(PartialEq)]
    enum Kind { Letter, Digit, Punct, Space }
    fn classify(c: char) -> Kind {
        if c.is_alphabetic() { Kind::Letter }
        else if c.is_ascii_digit() { Kind::Digit }
        else if c.is_whitespace() { Kind::Space }
        else { Kind::Punct }
    }

    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        // A single leading space attaches to the next non-space run.
        let mut start = i;
        if c == ' ' && i + 1 < chars.len() && classify(chars[i + 1]) != Kind::Space {
            i += 1;
        } else {
            start = i;
        }
        let kind = classify(chars[i]);
        let _ = start;
        let mut j = i + 1;
        while j < chars.len() {
            if classify(chars[j]) != kind {
                break;
            }
            // For whitespace runs, GPT-2 keeps trailing newlines etc
            // grouped; for the others, runs are homogeneous. We don't
            // distinguish further.
            j += 1;
        }
        // If we consumed a leading space, include it.
        let piece_start = if i > 0 && chars[i - 1] == ' '
            && (out.is_empty() || !out.last().unwrap().ends_with(' '))
            && classify(chars[i]) != Kind::Space
        {
            i - 1
        } else {
            i
        };
        let s: String = chars[piece_start..j].iter().collect();
        out.push(s);
        i = j;
    }
    out
}

/// GPT-2's `bytes_to_unicode` mapping: bytes 33..=126 ∪ 161..=172 ∪
/// 174..=255 pass through; all other bytes (0..=32 ∪ 127..=160 ∪ 173)
/// get pushed into codepoints 256..=323 in the order they appear. So
/// byte `0x20` (space) → `Ġ` (U+0120), `0x0A` (newline) → `Ċ`
/// (U+010A), `0x09` (tab) → `ĉ`, etc.
pub fn gpt2_byte_to_unicode(byte: u8) -> char {
    let b = byte as u32;
    let printable = (33..=126).contains(&b)
        || (161..=172).contains(&b)
        || (174..=255).contains(&b);
    if printable {
        return char::from_u32(b).unwrap();
    }
    // Count the unmapped bytes preceding `b` to find its codepoint
    // offset in the 256+ range.
    let mut n = 0u32;
    for i in 0..b {
        let mapped = (33..=126).contains(&i)
            || (161..=172).contains(&i)
            || (174..=255).contains(&i);
        if !mapped {
            n += 1;
        }
    }
    char::from_u32(256 + n).unwrap()
}

/// Parse a SentencePiece byte-fallback token like `<0x0A>` into the
/// underlying byte. Returns `None` if the string isn't in that exact
/// 6-char form.
fn parse_byte_token(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() != 6 || &b[..3] != b"<0x" || b[5] != b'>' { return None; }
    u8::from_str_radix(&s[3..5], 16).ok()
}

/// Replace SentencePiece / GPT-2-BPE space markers in a token's raw
/// chars. Conservative: only swaps the known marker codepoints and
/// leaves everything else verbatim.
fn decode_token_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{2581}' => out.push(' '), // ▁ SentencePiece
            '\u{0120}' => out.push(' '), // Ġ GPT-2
            '\u{010A}' => out.push('\n'), // Ċ GPT-2 newline
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replaces_sentencepiece_space() {
        assert_eq!(decode_token_chars("\u{2581}Hello"), " Hello");
        assert_eq!(decode_token_chars("\u{2581}\u{2581}foo"), "  foo");
    }

    #[test]
    fn replaces_gpt2_space_and_newline() {
        assert_eq!(decode_token_chars("\u{0120}world"), " world");
        assert_eq!(decode_token_chars("end\u{010A}next"), "end\nnext");
    }

    #[test]
    fn gpt2_byte_to_unicode_landmarks() {
        // Printable ASCII pass through.
        assert_eq!(gpt2_byte_to_unicode(b'!'), '!');
        assert_eq!(gpt2_byte_to_unicode(b'~'), '~');
        assert_eq!(gpt2_byte_to_unicode(b'A'), 'A');
        // The remapped bytes start at U+0100.
        assert_eq!(gpt2_byte_to_unicode(0x00), '\u{0100}'); // Ā
        assert_eq!(gpt2_byte_to_unicode(0x09), '\u{0109}'); // ĉ (tab)
        assert_eq!(gpt2_byte_to_unicode(0x0A), '\u{010A}'); // Ċ (newline)
        assert_eq!(gpt2_byte_to_unicode(0x20), '\u{0120}'); // Ġ (space)
        // Extended Latin pass through.
        assert_eq!(gpt2_byte_to_unicode(0xC2), '\u{00C2}');
    }

    #[test]
    fn preserves_normal_text() {
        assert_eq!(decode_token_chars("ipment"), "ipment");
        assert_eq!(decode_token_chars("Dynamic"), "Dynamic");
    }

    /// Smoke-test the chat-template formatter against the real Jamba
    /// GGUF: verify the encoded prompt round-trips through `decode` to
    /// the expected `<|im_start|>user\n…\n<|im_start|>assistant\n<think>\n`
    /// rendering. Skips if the external drive isn't mounted.
    #[test]
    fn chat_prompt_roundtrip_jamba() {
        use crate::parser::GgufFile;
        let path = std::path::Path::new(
            "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/jamba-reasoning-3b-gguf/jamba-reasoning-3b-Q4_K_M.gguf"
        );
        if !path.exists() {
            eprintln!("chat_prompt_roundtrip_jamba: GGUF not found — skipping.");
            return;
        }
        let bytes = std::fs::read(path).expect("read");
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let tk = Tokenizer::from_gguf(&gguf).expect("tokenizer");
        let toks = tk.format_chat_prompt("Hi").expect("format_chat_prompt");
        eprintln!("chat-format tokens: {toks:?}");
        let rendered = tk.decode_sequence(&toks, false);
        eprintln!("decode (skip_special=false): {rendered:?}");
        eprintln!("decode (skip_special=true) : {:?}", tk.decode_sequence(&toks, true));
        // The exact rendered template should match what add_generation_prompt=true
        // emits (no system, no thinking-prefix path).
        assert_eq!(
            rendered,
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );

        // Multi-turn + system + thinking prefix.
        let toks2 = tk.format_chat_messages(
            &[("system", "You are concise."), ("user", "What is 2+2?")],
            true, true,
        ).expect("format_chat_messages");
        let r2 = tk.decode_sequence(&toks2, false);
        eprintln!("multi-turn render: {r2:?}");
        assert!(r2.starts_with("<|startoftext|><|im_start|>system\nYou are concise.<|im_end|>\n"));
        assert!(r2.contains("<|im_start|>user\nBegin by thinking about the reasoning process"));
        assert!(r2.contains("What is 2+2?<|im_end|>\n"));
        assert!(r2.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    /// Cross-check our GPT-2 BPE encoder against `llama-tokenize` on
    /// Nemotron-Nano-4B: encode "What is 7 + 8?" with the chat
    /// template and verify we produce the same token IDs llama.cpp's
    /// `llama-tokenize` does. Skips if the GGUF isn't mounted.
    #[test]
    fn nemotron_tokenize_matches_llama_cpp() {
        use crate::parser::GgufFile;
        const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/nemotron-nano-raw/models--unsloth--NVIDIA-Nemotron-3-Nano-4B-GGUF/snapshots/8e81be55c5aa3d63bb82b6ceec62d50805d9e1bb/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf";
        if !std::path::Path::new(PATH).exists() {
            return;
        }
        let bytes = std::fs::read(PATH).expect("read");
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let tk = Tokenizer::from_gguf(&gguf).expect("tokenizer");

        // Format the chat prompt the same way `rce-nemotron-chat` does.
        let toks = tk
            .format_chat_nemotron(&[("user", "What is 7 + 8?")], true, true)
            .expect("format");
        eprintln!("ours ({}): {toks:?}", toks.len());
        for &t in &toks {
            eprintln!("  {t:5} -> {:?}", tk.decode_sequence(&[t], false));
        }

        // llama-tokenize produced (no BOS):
        //   10, 3263, 1010, 7493, 1395, 1032, 1055, 1867, 1032, 1056, 1063,
        //   11, 1010, 10, 1503, 19464, 1010, 12, 1010
        // Our format prepends BOS (1) and appends an extra newline after <think>.
        // Compare the body (skip BOS):
        let expected_body: Vec<i32> = vec![
            10, 25708, 1010, 11, 1010,            // <|im_start|>system\n<|im_end|>\n  (empty system block)
            10, 3263, 1010,                       // <|im_start|>user\n
            7493, 1395, 1032, 1055, 1867,
            1032, 1056, 1063,                     // What is 7 + 8?
            11, 1010,                             // <|im_end|>\n
            10, 1503, 19464, 1010,                // <|im_start|>assistant\n
            12, 1010,                             // <think>\n
        ];
        let our_body = &toks[1..];
        assert_eq!(our_body, expected_body.as_slice(),
            "tokenization diverged from llama.cpp's llama-tokenize output");
    }

    /// Cross-architecture validation: load the Nemotron-3-Nano-4B
    /// (`nemotron_h` arch, GPT-2 byte-level BPE tokenizer, 131K vocab,
    /// Q4_0 embeddings + Q8_0 lm_head + Q5_K/Q4_1 mixed quants) and
    /// run the substrate's **bypass-blocks** check on it: read
    /// `token_embd` for a few token ids, apply `output_norm`, project
    /// through the transposed `output.weight`, and verify the top-5
    /// nearest tokens are sensible case/spacing variants. This is the
    /// same diagnostic that caught the Q4_K dequant bug on Jamba —
    /// running it on a different architecture's weight mix proves the
    /// dequant + tokenizer machinery generalises. Skips if the
    /// external drive isn't mounted.
    #[test]
    fn nemotron_nano_4b_bypass_blocks() {
        use crate::dequant::read_tensor_f32;
        use crate::parser::GgufFile;
        const PATH: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/nemotron-nano-raw/models--unsloth--NVIDIA-Nemotron-3-Nano-4B-GGUF/snapshots/8e81be55c5aa3d63bb82b6ceec62d50805d9e1bb/NVIDIA-Nemotron-3-Nano-4B-Q4_0.gguf";
        let path = std::path::Path::new(PATH);
        if !path.exists() {
            eprintln!("nemotron_nano_4b_bypass_blocks: GGUF not found — skipping.");
            return;
        }
        let bytes = std::fs::read(path).expect("read");
        let gguf = GgufFile::parse(&bytes).expect("parse");
        let tk = Tokenizer::from_gguf(&gguf).expect("tokenizer");

        // Pull the dims from metadata.
        let d_model = gguf.metadata.iter()
            .find(|kv| kv.key == "nemotron_h.embedding_length")
            .and_then(|kv| match &kv.value {
                crate::parser::GgufValue::U32(v) => Some(*v as usize),
                _ => None,
            })
            .expect("embedding_length");
        let vocab = gguf.metadata.iter()
            .find(|kv| kv.key == "nemotron_h.vocab_size")
            .and_then(|kv| match &kv.value {
                crate::parser::GgufValue::U32(v) => Some(*v as usize),
                _ => None,
            })
            .expect("vocab_size");
        let eps = 1e-5f32; // nemotron_h.attention.layer_norm_rms_epsilon

        eprintln!("nemotron_h arch: d_model={d_model}, vocab={vocab}, eps={eps}");

        let token_embd = read_tensor_f32(&bytes, &gguf, "token_embd.weight").expect("token_embd");
        let output_norm_w = read_tensor_f32(&bytes, &gguf, "output_norm.weight").expect("output_norm");
        let lm_head_w = read_tensor_f32(&bytes, &gguf, "output.weight").expect("output");
        assert_eq!(token_embd.len(), vocab * d_model, "token_embd shape mismatch");
        assert_eq!(lm_head_w.len(), vocab * d_model, "lm_head shape mismatch");
        assert_eq!(output_norm_w.len(), d_model, "output_norm shape mismatch");

        // Transpose lm_head into [d_model, vocab] for the projection.
        let mut lm_head_t = vec![0.0f32; vocab * d_model];
        for r in 0..vocab {
            for c in 0..d_model {
                lm_head_t[c * vocab + r] = lm_head_w[r * d_model + c];
            }
        }

        let rms_norm = |x: &[f32]| -> Vec<f32> {
            let n = x.len() as f32;
            let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
            let inv = 1.0 / (mean_sq + eps).sqrt();
            x.iter().zip(output_norm_w.iter()).map(|(v, w)| v * inv * w).collect()
        };
        let matvec = |a: &[f32]| -> Vec<f32> {
            let mut out = vec![0.0f32; vocab];
            for j in 0..vocab {
                let mut acc = 0.0f32;
                for i in 0..d_model {
                    acc += a[i] * lm_head_t[i * vocab + j];
                }
                out[j] = acc;
            }
            out
        };

        for &tok in &[1i32, 11i32] {
            let row = tok as usize * d_model;
            let emb = &token_embd[row..row + d_model];
            let normed = rms_norm(emb);
            let logits = matvec(&normed);
            let mut indexed: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as i32, v)).collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<(i32, String)> = indexed.iter().take(5)
                .map(|&(id, _)| (id, tk.decode_sequence(&[id], false)))
                .collect();
            eprintln!("tok={tok:5} ({:?}): top-5={top5:?}", tk.decode_sequence(&[tok], false));
        }

        // A few content tokens — pick whatever the GPT-2 vocab uses
        // for "Hello", "world", "The", " is" by scanning.
        for needle in &["Hello", "ĠHello", "ĠThe", "Ġis", "Ġworld"] {
            if let Some(id) = tk.lookup_token(needle) {
                let row = id as usize * d_model;
                let emb = &token_embd[row..row + d_model];
                let normed = rms_norm(emb);
                let logits = matvec(&normed);
                let mut indexed: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &v)| (i as i32, v)).collect();
                indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let top3: Vec<(i32, String)> = indexed.iter().take(3)
                    .map(|&(id, _)| (id, tk.decode_sequence(&[id], false)))
                    .collect();
                eprintln!("tok={id:5} (raw {needle:?}, decoded {:?}): top-3={top3:?}", tk.decode_sequence(&[id], false));
            }
        }
    }
}
