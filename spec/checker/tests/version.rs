//! `rce --version` names the specification draft and the release label of the one language.
#[test]
fn version_flag_names_draft_and_release_label() {
    for flag in ["--version", "-V"] {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg(flag).output().unwrap();
        assert!(o.status.success());
        let s = String::from_utf8(o.stdout).unwrap();
        assert!(s.starts_with("rce "), "{s}");
        assert!(s.contains("specification draft 0.9"), "{s}");
        assert!(s.contains(rce_spec_check::parse::LANGUAGE_RELEASE), "{s}");
    }
}
