//! End-to-end dispatch: visual_spatial query → mgai-reasoner picks
//! `v-jepa-2` → app dispatches to the matching rce-vjepa runtime
//! → encode video → embeddings.
//!
//! Closes the open frontier flagged in [pattern-lang chapter close]:
//! "mgai-reasoner ↔ rce-vjepa dispatch: registry knows about
//! v-jepa-2 by id but no example shows the runtime being invoked
//! when the router returns that id." This example IS that
//! invocation — the loose coupling is a string match on
//! `reasoner_id == "v-jepa-2"` plus a path lookup table.

use rce_vjepa::VJepa2;
use mgai_reasoner::{
    ArchitectureFamily, BackboneSelector, CapabilityStatus, ComplexityClass,
    Intent, Modality, QueryFeatures, ReasonerCapability, Section57Router,
};
use std::collections::HashMap;
use std::path::Path;

/// Static registry: model identifier → on-disk path. In production
/// this lives in the planner's deployment config; we hardcode the
/// V-JEPA 2 ViT-L 256px / 64f checkpoint here.
fn model_paths() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert(
        "v-jepa-2",
        "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/vjepa2-vitl-fpc64-256.safetensors",
    );
    m
}

fn main() {
    // 1) App composes a query: "describe this clip" with a video modality.
    //    The `visual_spatial` tag is the §5.7 routing key.
    let q = QueryFeatures {
        modalities_in: vec![Modality::Video],
        intent: Intent::Recommendation,
        domain_tags: vec!["visual_spatial".into()],
        estimated_context_tokens: 0,
        heterogeneous_difficulty: false,
        complexity: ComplexityClass::Medium,
    };

    // 2) Deployment advertises what's actually online. Just one model
    //    here, but the selector returns "no reasoner" cleanly when
    //    the wanted family is absent — that's the planner's signal to
    //    fall back or report suboptimality.
    let avail = vec![
        ReasonerCapability {
            reasoner_id: "v-jepa-2".into(),
            architecture_family: ArchitectureFamily::Jepa,
            model_identifier: "meta/v-jepa-2".into(),
            optimal_query_classes: vec!["visual_spatial".into()],
            typical_joules_per_query: HashMap::from([("visual_spatial".into(), 12.5)]),
            crossover_context_length: None,
            activation_ratio: None,
            total_parameters: 1_200_000_000,
            active_parameters: 1_200_000_000,
            status: CapabilityStatus::Healthy,
            license: "Meta research license — check terms".into(),
        },
    ];

    let sel = Section57Router.select(&q, &avail);
    println!("REASONER VERDICT");
    println!("  wanted family : {:?}", sel.wanted_family);
    println!("  actual family : {:?}", sel.actual_family);
    println!("  rule fired    : {}", sel.rule_fired);
    println!("  reasoner id   : {:?}", sel.reasoner_id);
    println!("  family match  : {}", sel.family_match);

    // 3) Dispatch: app maps `reasoner_id` → runtime crate + checkpoint.
    let Some(id) = sel.reasoner_id else {
        eprintln!("no reasoner selected; would fall back to MRL embedding lookup");
        return;
    };

    let paths = model_paths();
    let Some(&path) = paths.get(id.as_str()) else {
        eprintln!("no runtime registered for reasoner_id={id:?}; registry needs updating");
        return;
    };
    if !Path::new(path).exists() {
        eprintln!("checkpoint not mounted: {path}");
        return;
    }

    println!("\nDISPATCH → loading {path}");
    let t0 = std::time::Instant::now();
    let m = VJepa2::load(path).expect("load V-JEPA 2");
    let dt_load = t0.elapsed();
    println!("  loaded in {dt_load:?}  (enc {}L d{} + pred {}L d{})",
             m.cfg.enc_layers, m.cfg.enc_dim, m.cfg.pred_layers, m.cfg.pred_dim);

    // 4) Run a tiny synthetic clip through the encoder.
    let t_frames = 2;
    let h = 32;
    let w = 32;
    let video: Vec<f32> = (0..3 * t_frames * h * w)
        .map(|i| ((i as f32 * 0.013).sin()) * 0.5)
        .collect();
    let t0 = std::time::Instant::now();
    let (tokens, n_t, n_h, n_w) = m.encode(&video, t_frames, h, w);
    let dt_enc = t0.elapsed();

    let amax = tokens.iter().flat_map(|t| t.iter()).fold(0.0f32, |m, v| m.max(v.abs()));
    let n_finite = tokens.iter().flat_map(|t| t.iter()).filter(|v| v.is_finite()).count();
    let n_total = tokens.len() * m.cfg.enc_dim;

    println!("\nENCODE → {n_t}×{n_h}×{n_w} tubelet/patch grid = {} tokens × {} dim",
             tokens.len(), m.cfg.enc_dim);
    println!("  finite      : {n_finite}/{n_total}");
    println!("  amax        : {amax:.3}");
    println!("  wall time   : {dt_enc:?}");
    println!("\nOK: visual_spatial query routed to v-jepa-2 → rce-vjepa.encode → {} embeddings.",
             tokens.len());
    assert_eq!(id, "v-jepa-2");
    assert_eq!(n_finite, n_total);
}
