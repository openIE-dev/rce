//! **Placement-driven production proof — TinyLlama through PlacementQmmAdapter.**
//!
//! With the
//! `placement` feature on, the public `placement_thread_local` API
//! installs a `PlacementQmmAdapter` on the current thread; every
//! `qmm_1d` site (Q/K/V/O, FFN gate/up/down, lm_head) goes through
//! `rce_placement::pick_priced` for backend selection.
//!
//! This test runs one forward pass on the real TinyLlama checkpoint
//! through three configurations and asserts:
//!
//! 1. The adapter saw the expected number of qmm calls
//!    (= n_layers × 7 + 1).
//! 2. Per-backend dispatch counts add up (CPU-only: all CPU; hybrid:
//!    CPU plus native Metal where Metal is available).
//! 3. Logits match the direct (no-adapter) path within the same
//!    f32-reorder tolerance (abs < 1e-4 OR rel < 1e-2).
//! 4. Top-1 token argmax is stable across all three paths.
//!
//! Skipped when the checkpoint isn't mounted. Build with
//! `cargo test -p rce-tinyllama --features placement --test
//! placement_hot_path -- --nocapture`.

#![cfg(feature = "placement")]

use rce_tinyllama::{placement_thread_local, TinyLlamaRuntime};

const MODEL: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/tinyllama-1.1b-chat-q8_0.gguf";

fn top1(logits: &[f32]) -> usize {
    logits.iter().enumerate()
        .fold((0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
        .0
}

fn assert_logits_close(adapter_label: &str, got: &[f32], direct: &[f32]) {
    // Tolerance budget: abs < 1e-4 OR rel < 1e-2.
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    for (i, (m, d)) in got.iter().zip(direct.iter()).enumerate() {
        let abs = (*m - *d).abs() as f64;
        let denom = (*d as f64).abs().max(1e-6);
        let rel = abs / denom;
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
        assert!(abs < 1e-4 || rel < 1e-2,
            "{adapter_label} logit[{i}] drift: got={m}, direct={d}, abs={abs:.3e}, rel={rel:.3e}");
    }
    eprintln!("{adapter_label}: max abs={max_abs:.3e}, max rel={max_rel:.3e}");
}

#[test]
fn tinyllama_forward_routes_every_qmm_through_placement_cpu_only() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("TL.placement: model not mounted — skipping");
        return;
    }

    let rt = TinyLlamaRuntime::load(MODEL).expect("load");
    let n_layers = rt.cfg.n_layers;
    let expected_qmms_per_token = n_layers * 7 + 1;

    // Baseline: no adapter installed.
    let mut state_direct = rt.new_state();
    let logits_direct = rt.forward_logits(&mut state_direct, 1);
    let top_direct = top1(&logits_direct);

    // CPU-only placement adapter.
    assert!(placement_thread_local::uninstall().is_none(),
        "no placement adapter should be installed before the test");
    placement_thread_local::install(
        placement_thread_local::PlacementQmmAdapter::new(),
    );

    let mut state_pl = rt.new_state();
    let logits_pl = rt.forward_logits(&mut state_pl, 1);

    let stats = placement_thread_local::stats().expect("stats");
    assert_eq!(stats.calls as usize, expected_qmms_per_token,
        "placement adapter should have seen one qmm per site");
    assert_eq!(
        stats.calls_per_backend.get("cpu:op-qmm").copied().unwrap_or(0) as usize,
        expected_qmms_per_token,
        "CPU-only adapter: every call should land on cpu:op-qmm");

    // CPU-only path goes through mm_1d, so logits should match the
    // direct baseline to within the shared tolerance
    // (the call-into-adapter is identical to the direct call modulo
    // function-call overhead — bit-identical in practice; we use the
    // looser bound for symmetry with the other tests).
    assert_logits_close("placement[cpu]", &logits_pl, &logits_direct);
    assert_eq!(top1(&logits_pl), top_direct,
        "top-1 argmax should match direct path");

    placement_thread_local::uninstall();
}

#[test]
fn tinyllama_forward_routes_through_placement_hybrid_cpu_and_metal() {
    if !std::path::Path::new(MODEL).exists() {
        eprintln!("TL.placement.hybrid: model not mounted — skipping");
        return;
    }

    let rt = TinyLlamaRuntime::load(MODEL).expect("load");
    let n_layers = rt.cfg.n_layers;
    let expected_qmms_per_token = n_layers * 7 + 1;

    let mut state_direct = rt.new_state();
    let logits_direct = rt.forward_logits(&mut state_direct, 1);
    let top_direct = top1(&logits_direct);

    assert!(placement_thread_local::uninstall().is_none());
    placement_thread_local::install(
        placement_thread_local::PlacementQmmAdapter::with_metal(),
    );

    let mut state_h = rt.new_state();
    let logits_h = rt.forward_logits(&mut state_h, 1);

    let stats = placement_thread_local::stats().expect("stats");
    assert_eq!(stats.calls as usize, expected_qmms_per_token);

    let cpu = stats.calls_per_backend.get("cpu:op-qmm").copied().unwrap_or(0);
    let metal = stats.calls_per_backend.get("metal:op-qmm").copied().unwrap_or(0);
    assert_eq!(cpu + metal, stats.calls,
        "per-backend counts must sum to total calls: {:?}", stats.calls_per_backend);

    eprintln!("TL.placement.hybrid: {} CPU + {} Metal calls (est. {:.3} mJ total)",
        cpu, metal, stats.estimated_microjoules() / 1000.0);

    assert_logits_close("placement[hybrid]", &logits_h, &logits_direct);
    assert_eq!(top1(&logits_h), top_direct,
        "hybrid top-1 must match direct path");

    placement_thread_local::uninstall();
}
