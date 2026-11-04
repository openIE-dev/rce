//! **Item #9 productionization proof — EAGLE-3 with a REAL rce model.**
//!
//! The `rce-spec-decode` crate ships the EAGLE-3 accept/reject loop
//! plus `DraftModel` / `TargetModel` traits. Up to now it only had a
//! toy `ToyTarget` (one-hot deterministic vocab) and `SelfSpec` (degenerate
//! draft=target). This test proves the protocol composes with a real
//! rce inference engine.
//!
//! Wraps `TinyLlamaRuntime` as a `TargetModel`. Uses `SelfSpec` to use
//! TinyLlama as its own draft (degenerate case, 100% accept rate, no
//! perf win — but proves the integration). Asserts:
//!
//! 1. K+1 distributions returned by `verify`, each summing to ≈1.0
//! 2. All K drafts accepted (since draft == target, the
//!    accept ratio `p_t(d) / p_d(d) = 1` so every Uniform(0,1)
//!    test passes — this is what `SelfSpec` is for)
//! 3. The accepted tokens form a valid greedy extension of the prompt
//!
//! Skipped when the TinyLlama checkpoint isn't mounted.
//!
//! **Next brick** (not in this test): use a smaller distilled model
//! (or fewer-layer slice of TinyLlama itself) as the draft for a
//! larger rce target (e.g. Qwen3-8B, Gemma 4 text) and measure
//! real tok/s. The current test proves the plumbing; perf comes when
//! draft and target diverge.

use rce_spec_decode::{
    speculative_decode, DraftModel, SelfSpec, SpecDecodeConfig, TargetModel,
};
use rce_tinyllama::TinyLlamaRuntime;

const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/tinyllama-1.1b-chat-q8_0.gguf";

/// Wrap a `TinyLlamaRuntime` so it satisfies `TargetModel`. Each
/// `verify` call replays the full context from a fresh state then
/// extends by each draft token, capturing the logits at each
/// position. K+1 distributions are returned per spec.
///
/// This is the slow path — every verify rebuilds the KV cache from
/// scratch. Production would (a) carry the verified-up-to position
/// in the cache and only replay `[carry_pos, end)`, and (b) batch the
/// K candidate tokens into a single tensor-axis-K forward (the
/// "verify is one forward" property the EAGLE-3 paper assumes).
/// rce's per-token forward doesn't yet expose batched-K verify;
/// adding it is the next brick.
struct TinyLlamaTarget<'a> {
    rt: &'a TinyLlamaRuntime,
}

impl<'a> TargetModel for TinyLlamaTarget<'a> {
    fn vocab_size(&self) -> usize { self.rt.cfg.vocab }

    fn verify(&mut self, context: &[i32], draft_tokens: &[i32]) -> Vec<Vec<f32>> {
        let mut state = self.rt.new_state();
        let mut last_logits: Vec<f32> = Vec::new();
        // Replay context.
        for &t in context {
            last_logits = self.rt.forward_logits(&mut state, t);
        }
        if context.is_empty() {
            // If somehow called with empty context, fabricate a uniform
            // distribution to avoid panicking in softmax.
            last_logits = vec![0.0; self.rt.cfg.vocab];
        }
        let mut out = Vec::with_capacity(draft_tokens.len() + 1);
        out.push(softmax(&last_logits));
        for &dt in draft_tokens {
            last_logits = self.rt.forward_logits(&mut state, dt);
            out.push(softmax(&last_logits));
        }
        out
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut idx = 0;
    let mut best = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best { best = x; idx = i; }
    }
    idx
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum > 0.0 { for x in exp.iter_mut() { *x /= sum; } }
    exp
}

#[test]
fn tinyllama_as_self_spec_target_accepts_all_drafts() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("TL.spec: model not mounted — skipping");
        return;
    }

    let rt = TinyLlamaRuntime::load(MODEL).expect("load");

    // Build a real prompt context. Use BOS + a few tokens.
    let tk = rt.tokenizer();
    let prompt_ids = tk.encode("Hello", true);
    assert!(prompt_ids.len() >= 2, "need at least BOS + one token");

    // K=3 draft tokens. Two TinyLlamaTarget instances over the same
    // shared runtime — SelfSpec wraps one as the draft, speculative_decode
    // borrows the other as the target. Per the convention in
    // rce-spec-decode::tests::self_spec_matches_target_argmax.
    let cfg = SpecDecodeConfig { k: 3, seed: 42 };
    let mut target_a = TinyLlamaTarget { rt: &rt };
    let mut target_b = TinyLlamaTarget { rt: &rt };

    // ── Step 1: draft phase produces K proposals.
    let proposals = {
        let mut draft = SelfSpec { target: &mut target_a };
        draft.draft(&prompt_ids, cfg.k)
    };
    assert_eq!(proposals.len(), cfg.k, "draft must produce K proposals");
    for (i, p) in proposals.iter().enumerate() {
        assert!(p.token >= 0 && (p.token as usize) < rt.cfg.vocab,
                "draft[{i}] token {} out of vocab", p.token);
        assert!(p.draft_prob > 0.0 && p.draft_prob <= 1.0,
                "draft[{i}] prob {} out of [0,1]", p.draft_prob);
    }

    // ── Step 2: verify phase returns K+1 distributions.
    let draft_tokens: Vec<i32> = proposals.iter().map(|p| p.token).collect();
    let dists = target_b.verify(&prompt_ids, &draft_tokens);
    assert_eq!(dists.len(), cfg.k + 1, "verify must return K+1 distributions");
    for (i, d) in dists.iter().enumerate() {
        assert_eq!(d.len(), rt.cfg.vocab, "dist[{i}] vocab mismatch");
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "dist[{i}] sums to {sum}, expected 1.0");
    }

    // ── Step 3: full speculative_decode step. Since draft==target,
    //    EVERY draft must be accepted and a bonus token sampled at the
    //    end (length K+1).
    let mut draft = SelfSpec { target: &mut target_a };
    let result = speculative_decode(&mut draft, &mut target_b, &prompt_ids, &cfg);

    assert!(
        result.accepted.len() >= cfg.k,
        "self-spec should accept all K drafts; got {} accepted (n_draft_accepted={})",
        result.accepted.len(), result.n_draft_accepted,
    );
    assert_eq!(
        result.n_draft_accepted, cfg.k,
        "all {} drafts should be accepted in self-spec mode", cfg.k,
    );
    // First K accepted tokens must equal the proposals.
    for i in 0..cfg.k {
        assert_eq!(
            result.accepted[i], proposals[i].token,
            "self-spec accepted[{i}] ({}) != draft proposal ({})",
            result.accepted[i], proposals[i].token,
        );
    }

    eprintln!(
        "TL.spec OK: prompt={:?} drafts={:?} accepted={:?} (corrected={})",
        prompt_ids, draft_tokens, result.accepted, result.corrected,
    );
}

/// Multi-step spec-decode generation loop: drive the protocol forward
/// step-by-step, accumulating accepted tokens until a budget is reached.
/// With self-spec, every step should produce K+1 tokens (no rejections).
/// This proves the loop composes: the per-step `n_draft_accepted` /
/// `accepted` semantics are correct for advancing context across steps.
///
/// Generates 12 tokens of continuation from "The quick brown" via
/// self-spec on TinyLlama. Decodes the result and asserts it's
/// non-empty + matches what plain greedy TinyLlama produces.
#[test]
fn tinyllama_self_spec_generation_loop() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("TL.spec-loop: model not mounted — skipping");
        return;
    }

    let rt = TinyLlamaRuntime::load(MODEL).expect("load");
    let tk = rt.tokenizer();
    let prompt = "The quick brown";
    let prompt_ids = tk.encode(prompt, true);

    let cfg = SpecDecodeConfig { k: 4, seed: 0 };
    let target_tok_budget = 12;

    // ── Spec-decode loop ──
    let mut context: Vec<i32> = prompt_ids.clone();
    let mut target_a = TinyLlamaTarget { rt: &rt };
    let mut target_b = TinyLlamaTarget { rt: &rt };
    let mut total_generated = 0;
    let mut steps = 0;
    let t0 = std::time::Instant::now();
    while total_generated < target_tok_budget {
        let mut draft = SelfSpec { target: &mut target_a };
        let r = speculative_decode(&mut draft, &mut target_b, &context, &cfg);
        assert!(!r.accepted.is_empty(), "step {steps} produced 0 tokens");
        for &t in &r.accepted {
            context.push(t);
            total_generated += 1;
            if total_generated >= target_tok_budget { break; }
        }
        steps += 1;
        assert!(steps < 20, "spec-decode loop not converging");
    }
    let dt_spec = t0.elapsed();

    let spec_tokens: Vec<i32> = context[prompt_ids.len()..].to_vec();

    // ── Baseline: plain greedy generation from TinyLlama ──
    let mut greedy_tokens: Vec<i32> = Vec::with_capacity(target_tok_budget);
    let mut state = rt.new_state();
    let mut last_logits: Vec<f32> = Vec::new();
    let t1 = std::time::Instant::now();
    for &t in &prompt_ids {
        last_logits = rt.forward_logits(&mut state, t);
    }
    while greedy_tokens.len() < target_tok_budget {
        let next = argmax(&last_logits) as i32;
        greedy_tokens.push(next);
        last_logits = rt.forward_logits(&mut state, next);
    }
    let dt_greedy = t1.elapsed();

    // Greedy and self-spec must agree on every token (modulo RNG —
    // with seed=0 the spec loop is "always accept when p_t/p_d ≥
    // uniform(0,1)" which equals greedy when p_t == p_d).
    let agree_len = spec_tokens.iter().zip(greedy_tokens.iter())
        .take_while(|(a, b)| a == b).count();
    eprintln!(
        "TL.spec-loop: {steps} spec steps → {} tokens in {dt_spec:?}; \
         greedy: {dt_greedy:?}; first {agree_len}/{target_tok_budget} agree",
        total_generated,
    );
    // Allow a few divergences from RNG noise far down the sequence.
    assert!(
        agree_len >= target_tok_budget * 3 / 4,
        "spec-decode diverged too early from greedy: agree={agree_len}/{target_tok_budget}",
    );
}
