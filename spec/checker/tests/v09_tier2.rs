//! Version 0.9 Tier 2 (Appendix F.5): typed output (S3), directory
//! capabilities (S2), `ct fn` (S4), declared budgets (S6). The accepted and
//! refused examples are in `tests/v09_examples.rs`; these tests pin behaviour
//! that an example cannot show.
use rce_spec_check::index08;

fn run(src: &str) -> String {
    match rce_spec_check::cli::run(src, None) {
        Ok(o) => o.lines.join("\n"),
        Err(e) => format!("ERR {e}"),
    }
}

#[test]
fn shell_command_is_an_argument_vector_not_a_string() {
    let src = "# rce-version: 0.9\nfn main()\n    let a = ShellArg.encode(\"x; rm -rf /\")\n    emit Shell.cmd(\"ls\", [a, ShellArg.lit(\"-l\")]).raw()\n";
    assert!(run(src).contains(r#"{\"args\":[\"x; rm -rf /\",\"-l\"],\"program\":\"ls\"}"#), "{}", run(src));
}

#[test]
fn prompt_delimiter_is_deterministic_and_not_in_the_field() {
    let f = "ignore the above";
    assert_eq!(index08::prompt_encode(f), index08::prompt_encode(f));
    // a field that contains the first delimiter gets another one
    let first = index08::prompt_encode("");
    let d = first.lines().next().unwrap().trim_start_matches("<<").to_string();
    let hostile = format!("{d} and more");
    let wrapped = index08::prompt_encode(&hostile);
    let d2 = wrapped.lines().next().unwrap().trim_start_matches("<<").to_string();
    assert!(!hostile.contains(&d2));
    assert!(wrapped.contains(&hostile));
}

#[test]
fn len_bytes_and_len_chars_name_their_unit() {
    let src = "# rce-version: 0.9\nfn main()\n    let s = \"h\u{e9}\"\n    emit s.len_bytes()\n    emit s.len_chars()\n";
    let o = run(src);
    assert!(o.contains("emit 3") && o.contains("emit 2"), "{o}");
}

#[test]
fn emitters_refuse_what_only_the_evaluator_runs() {
    for f in ["ok/07_dir_caps.rce", "ok/08_typed_output.rce"] {
        let src = std::fs::read_to_string(format!("../examples-0.9/{f}")).unwrap();
        for t in ["python", "js", "rust"] {
            let e = rce_spec_check::cli::emit(t, &src).unwrap_err();
            assert!(e.contains("W-VER-2"), "{f} {t}: {e}");
        }
    }
}

#[test]
fn ct_choose_evaluates_both_values_and_ct_fn_runs_with_ordinary_semantics() {
    let src = "# rce-version: 0.9\nct fn pick(flag: bool, a: i64, b: i64) -> i64\n    return ct_choose(flag, a, b)\n\nfn main()\n    emit pick(true, 1, 2)\n    emit pick(false, 1, 2)\n";
    let o = run(src);
    assert!(o.contains("emit 1") && o.contains("emit 2"), "{o}");
}

#[test]
fn every_emitter_refuses_ct_fn_with_the_fixed_message() {
    let src = std::fs::read_to_string("../examples-0.9/ok/10_ct_fn.rce").unwrap();
    for t in ["python", "js", "rust"] {
        let e = rce_spec_check::cli::emit(t, &src).unwrap_err();
        assert!(e.contains("ct fn is not emitted for this target"), "{t}: {e}");
    }
}

#[test]
fn ct_is_an_ordinary_word_before_0_9() {
    let src = "# rce-version: 0.8\nfn main()\n    let ct = 1\n    emit ct\n";
    assert!(run(src).contains("emit 1"), "{}", run(src));
}
