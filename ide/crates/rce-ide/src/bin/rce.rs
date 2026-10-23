//! `rce`: the RCE reference checker CLI shipped with the IDE, with the
//! energy-budget gate on `rce test`:
//!
//!   rce test FILES.. [--energy-budget J] [--measured J | --measure] [--strict]
//!
//! Without a budget flag, `rce test` behaves exactly as the reference
//! checker. Every other subcommand is passed through unchanged.
//!
//! `rce --version` (or `-V`) prints the release, the specification draft
//! the vendored checker implements and the language versions it accepts.

/// The specification draft of the vendored reference checker.
// The specification draft this checkout carries (spec/book/00-overview.md); one text, one language.
const SPEC_DRAFT: &str = "0.9";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() == 1 && matches!(args[0].as_str(), "--version" | "-V") {
        // One language, one text (decision 64): the release label is the
        // checker's LANGUAGE_RELEASE, not a list of accepted versions.
        println!(
            "rce {} (rce-ide; RCE specification draft {SPEC_DRAFT}; language release {})",
            env!("CARGO_PKG_VERSION"),
            rce_spec_check::parse::LANGUAGE_RELEASE
        );
        return;
    }
    let gated = args.first().map(String::as_str) == Some("test")
        && args.iter().any(|a| matches!(a.as_str(), "--energy-budget" | "--measured" | "--measure" | "--strict"));
    let code = if gated {
        rce_ide::ci::test(&args[1..], &mut std::io::stdout())
    } else {
        rce_spec_check::cli::main(&args, &mut std::io::stdin().lock(), &mut std::io::stdout(), &mut std::io::stderr())
    };
    std::process::exit(code);
}
