//! `rce`: the language commands of spec section 10 (check, run, test, emit,
//! fmt, lsp), as implemented by `cli::main` in this crate.
//! Usage: rce COMMAND WORD... OPERAND...   (`rce help` prints the list)
//! `rce version` prints the commit the checker was built from, the
//! specification draft it implements and the release label of the one
//! language (2.7). No typed version number is printed: a build is named by
//! its commit, read from git by build.rs.

fn main() {
    // HC-5: `rce host run|serve` enters the sandbox's OS enforcement target after opening D.
    rce_spec_check::sandbox::enable();
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `rce host serve` reads stdin on its own thread (the host loop), so the
    // main thread must not hold the stdin lock.
    let serve = args.first().is_some_and(|a| a == "host") && args.get(1).is_some_and(|a| a == "serve");
    let mut input: Box<dyn std::io::BufRead> = if serve { Box::new(std::io::empty()) } else { Box::new(std::io::stdin().lock()) };
    let code = rce_spec_check::cli::main(
        &args,
        &mut input,
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::exit(code);
}
