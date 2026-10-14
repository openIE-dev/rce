//! `rce`: the language commands of spec section 10 (check, run, test, emit,
//! fmt, lsp), as implemented by `cli::main` in this crate.
//! Usage: rce COMMAND ...   (run `rce` with no arguments for the list)
//! `rce --version` (or `-V`) prints the checker version, the specification
//! draft it implements and the language versions it accepts (2.7); it is a
//! convenience of this program, not one of the language commands.

/// The specification draft this checker implements.
const SPEC_DRAFT: &str = "0.8";

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if matches!(args.first().map(String::as_str), Some("--version" | "-V")) && args.len() == 1 {
        println!(
            "rce {} (RCE specification draft {SPEC_DRAFT}; language versions {})",
            env!("CARGO_PKG_VERSION"),
            rce_spec_check::parse::VERSIONS.join(", ")
        );
        return;
    }
    let code = rce_spec_check::cli::main(
        &args,
        &mut std::io::stdin().lock(),
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::exit(code);
}
