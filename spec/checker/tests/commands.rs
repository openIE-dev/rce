//! 10.19 CMD-1: every command of `rce help` is named in section 10 of the
//! specification, in Appendix A of the book and in the README, so a reader of
//! any of the three finds every command `rce` runs.

use rce_spec_check::cli::COMMANDS;

fn read(rel: &str) -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

#[test]
fn every_command_is_in_section_10_appendix_a_and_the_readme() {
    let places = [
        ("spec section 10", read("spec/book/12-tooling.md")),
        ("Appendix A", read("book/src/appendix-a-commands.md")),
        ("README", read("README.md")),
    ];
    let mut missing = vec![];
    for c in COMMANDS {
        let name = format!("`rce {}", c.path.join(" "));
        for (place, text) in &places {
            let found = text.match_indices(&name).any(|(i, _)| {
                // the name ends there: a backtick, a space or the end of the path
                matches!(text[i + name.len()..].chars().next(), Some('`' | ' ' | ']'))
            });
            if !found {
                missing.push(format!("{name}` is not in {place}"));
            }
        }
    }
    assert!(missing.is_empty(), "CMD-1:\n{}", missing.join("\n"));
    assert!(COMMANDS.len() >= 39, "{} commands", COMMANDS.len());
}
