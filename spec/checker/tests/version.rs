//! `rce --version` names the specification draft and every accepted language version.
#[test]
fn version_flag_names_draft_and_language_versions() {
    for flag in ["--version", "-V"] {
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_rce")).arg(flag).output().unwrap();
        assert!(o.status.success());
        let s = String::from_utf8(o.stdout).unwrap();
        assert!(s.starts_with("rce "), "{s}");
        assert!(s.contains("specification draft 0.9"), "{s}");
        for v in rce_spec_check::parse::VERSIONS {
            assert!(s.contains(v), "{s} lacks {v}");
        }
    }
}
