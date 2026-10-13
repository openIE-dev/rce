//! `rce`: the language commands of spec section 10 (check, run, test, emit,
//! fmt, lsp), as implemented by `cli::main` in this crate.
//! Usage: rce COMMAND ...   (run `rce` with no arguments for the list)
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = rce_spec_check::cli::main(
        &args,
        &mut std::io::stdin().lock(),
        &mut std::io::stdout(),
        &mut std::io::stderr(),
    );
    std::process::exit(code);
}
