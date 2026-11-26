use competence_catalog::adapter_gen;
use competence_catalog::editor_loop::{EditorLoop, EditOutcome};
use competence_catalog::lang_synth;
use competence_catalog::langpredict::FABRICS;
use std::collections::BTreeMap;

#[test]
fn generated_adapters_are_valid_and_cover_constructs() {
    let arts = adapter_gen::generate_all("rust").expect("rust artifacts");
    for f in ["package.json", "language-configuration.json", "snippets.json", "semanticTokens.json", "dap.json"] {
        let v: serde_json::Value = serde_json::from_str(&arts[f]).unwrap_or_else(|_| panic!("{f} invalid"));
        assert!(v.is_object());
    }
    let snips: serde_json::Value = serde_json::from_str(&arts["snippets.json"]).unwrap();
    assert!(snips.as_object().unwrap().len() > 20, "snippets cover the construct set");
    assert!(arts["extension.toml"].contains("id = \"rust\""));
    assert!(arts["extension.toml"].contains("[grammars.rust]"));
    let pkg: serde_json::Value = serde_json::from_str(&arts["package.json"]).unwrap();
    assert!(pkg["contributes"]["grammars"].is_array());
}

#[test]
fn zero_error_hypothesis_errors_caught_at_edit_none_escape() {
    let mut loop_ = EditorLoop::rust();
    let good = "let x = 1; return x + 1;";
    let probes: Vec<(BTreeMap<String, i128>, i128)> = vec![(BTreeMap::new(), 2)];
    // a bad edit: returns 3 but probes expect 2 -> rejected at edit time
    let bad = "return 3;";
    let r_bad = loop_.edit(bad, &probes);
    assert!(matches!(r_bad, EditOutcome::Rejected { .. }), "bad edit must be caught at edit time");
    // good edit admitted
    let r1 = loop_.edit(good, &probes);
    let key = match r1 { EditOutcome::Served { key, hit, .. } => { assert!(!hit); key } _ => panic!("good edit must be served") };
    // repeat -> G0 hit
    let r2 = loop_.edit(good, &probes);
    assert!(matches!(r2, EditOutcome::Served { hit: true, .. }));
    // serve re-verifies; no escaped error, no debug session
    assert!(loop_.serve(&key, &probes).is_some());
    assert_eq!(loop_.stats.escaped_errors, 0, "no error may escape to runtime");
    assert_eq!(loop_.stats.debug_sessions, 0, "no post-hoc debugging needed");
    assert!(loop_.stats.caught_at_edit >= 1);
    assert!(loop_.stats.served_hits >= 2);
    // reuse lowers cost/act
    let c_before = loop_.cost_per_act();
    for _ in 0..20 { loop_.edit(good, &probes); }
    assert!(loop_.cost_per_act() < c_before, "reuse must lower cost per act");
}

#[test]
fn divergence_localized_to_probe() {
    let stmt = competence_catalog::program::lift_program("return 3;").unwrap();
    let probes: Vec<(BTreeMap<String, i128>, i128)> = vec![(BTreeMap::new(), 2), (BTreeMap::new(), 3)];
    assert_eq!(competence_catalog::editor_loop::localize_divergence(&stmt, &probes), Some(0));
}

#[test]
fn synthesized_native_languages_are_real_and_replay_gated() {
    for f in FABRICS {
        let s = lang_synth::synthesize(f);
        assert_eq!(s.coverage(), 1.0, "{} grammar must cover its constructs", s.name);
        assert!(s.constructs.len() > 33, "core + native constructs");
        assert!(s.templates.contains_key(&f.native_constructs[0].to_string()));
        assert!(!s.native_semantics.is_empty(), "native semantics declared");
        assert!(s.sample_replays(), "{} sample must replay", s.name);
        // emission LUT renders the sample
        let stmt = competence_catalog::program::lift_program(&s.sample_src).unwrap();
        assert!(competence_catalog::lut_emit::render_stmt(&stmt, &s.lut()).is_ok());
    }
}

#[test]
fn quantum_native_has_no_clone_and_measure() {
    let q = FABRICS.iter().find(|f| f.id == "quantum").unwrap();
    let s = lang_synth::synthesize(q);
    assert!(s.constructs.contains(&"no_clone".to_string()));
    assert!(s.constructs.contains(&"measure".to_string()));
    assert!(s.native_semantics["no_clone"].contains("linearity"));
}
