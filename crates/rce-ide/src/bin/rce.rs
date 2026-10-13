//! `rce`: the RCE reference checker CLI shipped with the IDE, with the
//! energy-budget gate on `rce test`:
//!
//!   rce test FILES.. [--energy-budget J] [--measured J | --measure] [--strict]
//!
//! Without a budget flag, `rce test` behaves exactly as the reference
//! checker. Every other subcommand is passed through unchanged.
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gated = args.first().map(String::as_str) == Some("test")
        && args.iter().any(|a| matches!(a.as_str(), "--energy-budget" | "--measured" | "--measure" | "--strict"));
    let code = if gated {
        rce_ide::ci::test(&args[1..], &mut std::io::stdout())
    } else {
        rce_spec_check::cli::main(&args, &mut std::io::stdin().lock(), &mut std::io::stdout(), &mut std::io::stderr())
    };
    std::process::exit(code);
}
