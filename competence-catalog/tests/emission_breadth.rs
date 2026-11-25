//! Emission-breadth falsifiers: RCE-Compile covers the global language families
//! via data LUTs (not hand-written emitters), including non-Western languages.
//! Coverage is measured by emitting the shared expression fragment to every LUT.

use competence_catalog::lut_emit::{builtin_luts, emit_expr};

#[test]
fn emission_covers_global_language_families() {
    let luts = builtin_luts();
    let mut ok = 0;
    let mut failed = Vec::new();
    for lut in &luts {
        // The shared expression fragment: arithmetic every target must render.
        match emit_expr("(1 + 2) * 3", lut) {
            Ok(_) => ok += 1,
            Err(_) => failed.push(lut.name.clone()),
        }
    }
    assert!(failed.is_empty(), "languages that failed the fragment: {failed:?}");
    assert!(ok >= 45, "expected >= 45 language LUTs, got {ok}");
}

#[test]
fn non_western_languages_are_present() {
    let names: Vec<String> = builtin_luts().iter().map(|l| l.name.clone()).collect();
    // Global survey, not Western-first: Huawei Cangjie and IDEA Shenzhen MoonBit.
    assert!(names.contains(&"cangjie".to_string()), "Cangjie missing");
    assert!(names.contains(&"moonbit".to_string()), "MoonBit missing");
}

#[test]
fn emission_is_deterministic_per_language() {
    for lut in builtin_luts() {
        let a = emit_expr("1 + 2 * 3", &lut).unwrap();
        let b = emit_expr("1 + 2 * 3", &lut).unwrap();
        assert_eq!(a, b, "{} emission is not deterministic", lut.name);
    }
}

#[test]
fn logic_and_conditional_render_across_families() {
    // A logic + conditional expression renders in every family that defines if_expr.
    let src = "if (1 < 2) then (3 + 4) else (5 - 1)";
    let mut rendered = 0;
    for lut in builtin_luts() {
        if lut.template("if_expr").is_some() {
            assert!(emit_expr(src, &lut).is_ok(), "{} failed if-expr", lut.name);
            rendered += 1;
        }
    }
    assert!(rendered >= 40, "expected >= 40 targets with if-expr, got {rendered}");
}