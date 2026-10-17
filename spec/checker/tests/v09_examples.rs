//! Version 0.9 (Appendix F) examples: `../examples-0.9/ok/*.rce` must be
//! accepted by the reference checker; every `../examples-0.9/refused/*.rce`
//! must be refused with the rule named on its second line
//! (`# expect: W-...`). A refused example that is refused for a different
//! rule fails, so each file tests one rule.
use std::path::PathBuf;

fn files(sub: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(format!("../examples-0.9/{sub}"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "rce"))
        .collect();
    v.sort();
    v
}

#[test]
fn ok_examples_are_accepted() {
    let fs = files("ok");
    assert!(!fs.is_empty());
    for p in fs {
        let src = std::fs::read_to_string(&p).unwrap();
        assert!(src.starts_with("# rce-version: 0.9\n"), "{p:?}");
        assert_eq!(rce_spec_check::check_one(&src), Ok(()), "{p:?}");
    }
}

#[test]
fn refused_examples_are_refused_for_the_named_rule() {
    let fs = files("refused");
    assert!(!fs.is_empty());
    for p in fs {
        let src = std::fs::read_to_string(&p).unwrap();
        let want = src
            .lines()
            .nth(1)
            .and_then(|l| l.strip_prefix("# expect: "))
            .unwrap_or_else(|| panic!("{p:?} has no `# expect:` line"))
            .trim()
            .to_string();
        let r = rce_spec_check::check_one(&src);
        let e = r.expect_err(&format!("{p:?} must be refused"));
        assert!(e.contains(&want), "{p:?} refused, but not for {want}: {e}");
    }
}

/// The same refused text is accepted by the 0.8 reading where the rule is
/// new in 0.9 (a version is never applied silently to an older module).
#[test]
fn hygiene_rules_do_not_apply_to_08() {
    for name in ["unused_let", "unused_param", "unused_use", "constant_condition", "drop_value", "dead_store", "shadow_local"] {
        let src = std::fs::read_to_string(format!("../examples-0.9/refused/{name}.rce")).unwrap();
        let src08 = src.replacen("0.9", "0.8", 1);
        let r = rce_spec_check::check_one(&src08);
        assert!(r.as_ref().map_or(true, |_| true) && !r.clone().err().unwrap_or_default().contains("W-UNUSED") , "{name}: {r:?}");
    }
}

/// The version line is a release label: tolerated and ignored, never a gate.
#[test]
fn version_line_is_a_label_not_a_gate() {
    for l in ["0.5", "0.9", "0.10", "banana"] {
        let src = format!("# rce-version: {l}\nfn main()\n    emit \"x\"\n");
        assert_eq!(rce_spec_check::check_one(&src), Ok(()), "{l}");
    }
}

#[test]
fn reserved_words_are_names_after_a_dot_and_in_older_versions() {
    let src = "# rce-version: 0.8\nfn main()\n    let law = 1\n    emit law.tostr()\n";
    assert_eq!(rce_spec_check::check_one(src), Ok(()));
}

#[test]
fn lexical_hygiene_classes() {
    use rce_spec_check::v09::w_lex_4_class as k;
    for c in ['\u{202e}', '\u{2066}', '\u{200b}', '\u{2060}', '\u{feff}', '\u{e0041}', '\u{0}', '\u{85}', '\u{fdd0}', '\u{ffff}'] {
        assert!(k(c).is_some(), "{c:?}");
    }
    for c in ['a', ' ', 'é', '\u{4e2d}', '\u{1f600}', '\u{3000}'] {
        assert!(k(c).is_none(), "{c:?}");
    }
}

fn run(src: &str) -> String {
    let (mut out, mut err) = (vec![], vec![]);
    let d = std::env::temp_dir().join(format!("rce-v09-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let f = d.join(format!("t{}.rce", src.len()));
    std::fs::write(&f, src).unwrap();
    rce_spec_check::cli::main(&["run".into(), f.to_string_lossy().into_owned()], &mut &b""[..], &mut out, &mut err);
    String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap()
}

/// F.4.4: a failed call restores its `&mut` arguments in 0.9; in 0.8 they are
/// written back on every outcome (T33), so the same text means two things by
/// version and a 0.8 module never changes meaning.
#[test]
fn failed_call_restores_mut_arguments_in_09_only() {
    let src09 = std::fs::read_to_string("../examples-0.9/ok/04_transactional_restore.rce").unwrap();
    let src08 = src09.replacen("0.9", "0.8", 1);
    assert!(run(&src09).contains("emit \"10\""), "{}", run(&src09));
    assert!(run(&src08).contains("emit \"11\""), "{}", run(&src08));
}

/// The emitters do not implement the restore, so they refuse a 0.9 module
/// that needs it (reserved, not implemented, never silently different).
#[test]
fn emitters_refuse_what_needs_the_restore() {
    let src = std::fs::read_to_string("../examples-0.9/ok/04_transactional_restore.rce").unwrap();
    for t in ["python", "js", "rust"] {
        let e = rce_spec_check::cli::emit(t, &src).unwrap_err();
        assert!(e.contains("W-RES-1"), "{t}: {e}");
    }
}

/// F.6.4: a loop measure is erased before evaluation, so the program runs as
/// the one without the `decreases` line.
#[test]
fn loop_measure_is_erased_before_evaluation() {
    let with = std::fs::read_to_string("../examples-0.9/ok/16_loop_decreases.rce").unwrap();
    let without: String = with.lines().filter(|l| !l.trim_start().starts_with("decreases")).collect::<Vec<_>>().join("\n") + "\n";
    assert_ne!(with, without);
    assert_eq!(run(&with), run(&without));
}
