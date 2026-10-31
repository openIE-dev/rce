//! `rce --version` names the commit, the specification draft and the release
//! label of the one language, and no typed version number.
#[test]
fn version_flag_names_draft_and_release_label() {
    for flag in ["--version", "-V"] {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg(flag).output().unwrap();
        assert!(o.status.success());
        let s = String::from_utf8(o.stdout).unwrap();
        assert!(s.starts_with(&format!("rce commit {} (", rce_spec_check::GIT_COMMIT)), "{s}");
        assert!(!s.contains(env!("CARGO_PKG_VERSION")), "{s}");
        assert!(s.contains("specification draft 0.9"), "{s}");
        assert!(s.contains(rce_spec_check::parse::LANGUAGE_RELEASE), "{s}");
    }
}
