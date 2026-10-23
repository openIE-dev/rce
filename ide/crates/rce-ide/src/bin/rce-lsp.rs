//! `rce-lsp`: the RCE language server (the checker's `cli::serve`, spec
//! 10.7) on stdin/stdout, for the IDE's stdio LSP transport.
fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    std::process::exit(rce_spec_check::cli::serve(&mut stdin.lock(), &mut out));
}
