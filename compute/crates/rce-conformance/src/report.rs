use crate::runner::BackendId;

#[derive(Debug, Clone)]
pub enum Verdict {
    Pass,
    Diverge(String),
    Error(String),
    Skipped(String),
}

impl Verdict {
    pub fn label(&self) -> &'static str {
        match self {
            Verdict::Pass => "PASS",
            Verdict::Diverge(_) => "DIVERGE",
            Verdict::Error(_) => "ERROR",
            Verdict::Skipped(_) => "SKIP",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReportLine {
    pub case: String,
    pub backend: BackendId,
    pub verdict: Verdict,
}

pub fn print_summary(report: &[ReportLine]) {
    use std::collections::BTreeMap;
    let mut by_backend: BTreeMap<&'static str, (u32, u32, u32, u32)> = BTreeMap::new();
    for line in report {
        let entry = by_backend.entry(line.backend.label()).or_insert((0, 0, 0, 0));
        match &line.verdict {
            Verdict::Pass => entry.0 += 1,
            Verdict::Diverge(_) => entry.1 += 1,
            Verdict::Error(_) => entry.2 += 1,
            Verdict::Skipped(_) => entry.3 += 1,
        }
    }
    println!("# rce conformance report");
    println!();
    println!("| backend     | pass | diverge | error | skip |");
    println!("| ----------- | ----:| -------:| -----:| ----:|");
    for (b, (p, d, e, s)) in &by_backend {
        println!("| {b:<11} | {p:>4} | {d:>7} | {e:>5} | {s:>4} |");
    }
    println!();

    let failures: Vec<&ReportLine> = report
        .iter()
        .filter(|l| matches!(l.verdict, Verdict::Diverge(_) | Verdict::Error(_)))
        .collect();
    if !failures.is_empty() {
        println!("## Failures");
        println!();
        for line in failures {
            let detail = match &line.verdict {
                Verdict::Diverge(d) | Verdict::Error(d) => d.as_str(),
                _ => "",
            };
            println!(
                "- **{}** [{}]: {} — {}",
                line.case,
                line.backend.label(),
                line.verdict.label(),
                detail
            );
        }
    }
}
