//! Regression test for the arch-dispatch generalization that lets
//! `rce-bamba` load **both** Bamba and Granite-4 GGUFs (they share
//! the Mamba-2 + Transformer hybrid pattern and tensor naming; only
//! the `general.architecture` string and metadata prefix differ).
//!
//! - Bamba check: gates on the existing `bamba-9b.gguf` checkpoint on
//!   the external library; skipped when not mounted.
//! - Granite check: gates on a future-mounted `granite-4.1-8b.gguf`;
//!   skipped when not present. Once the user mounts the file, this
//!   test starts running and validates the cross-arch loader.

use rce_bamba::BambaRuntime;

const BAMBA: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/bamba-9b-gguf/bamba-9b.gguf";
const GRANITE: &str = "/Volumes/macos_4TB_external_1/neuronexus_ai/model_library/granite-4.1-8b-gguf/granite-4.1-8b-Q8_0.gguf";

#[test]
fn bamba_loads_after_arch_generalization() {
    if !std::path::Path::new(BAMBA).exists() {
        eprintln!("bamba: GGUF not mounted at {BAMBA} — skipping");
        return;
    }
    let rt = BambaRuntime::load(BAMBA).expect("load Bamba");
    let _ = rt; // load alone is the regression signal
}

#[test]
fn granite_4_1_8b_loads_via_bamba_runtime() {
    if !std::path::Path::new(GRANITE).exists() {
        eprintln!(
            "granite: GGUF not mounted at {GRANITE} — skipping. \
             Download IBM Granite 4.1-8B GGUF and place it there to enable."
        );
        return;
    }
    let rt = BambaRuntime::load(GRANITE).expect("load Granite 4.1-8B");
    let _ = rt;
}
