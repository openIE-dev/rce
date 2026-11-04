//! `rce-conformance` binary: run the corpus, print the report.

use anyhow::Result;
use rce_conformance::{
    corpus::full_corpus,
    report::{print_summary, Verdict},
    run_corpus, RunOptions,
};

fn main() -> Result<()> {
    let opts = RunOptions::default();
    let corpus = full_corpus();
    eprintln!("[conformance] running {} cases…", corpus.len());
    let report = run_corpus(&corpus, &opts);
    print_summary(&report);

    // Exit non-zero if any case diverged or errored. Skips don't fail
    // the run (used when the host has no GPU).
    let any_failed = report
        .iter()
        .any(|l| matches!(l.verdict, Verdict::Diverge(_) | Verdict::Error(_)));
    if any_failed {
        std::process::exit(1);
    }
    Ok(())
}
