//! `rce-ide telemetry probe|setup|measure`.
use crate::discover::{Status, TRACKED, probe_all};
use crate::setup::{Outcome, Terminal, apply, plan};

pub fn usage() -> &'static str {
    "rce-ide telemetry probe          list every energy source and whether it works here\n\
     rce-ide telemetry setup          grant the permissions some sources need (asks for each step)\n\
     rce-ide telemetry calibrate      run fixed workloads under the meters and fit the simulator\n"
}

fn status(s: &Status) -> String {
    match s {
        Status::Available => "available".into(),
        Status::NeedsSetup(m) => format!("needs setup: {m}"),
        Status::Unsupported(m) => format!("unsupported: {m}"),
        Status::Stale(m) => format!("stale: {m}"),
    }
}

/// Run a telemetry subcommand; returns the exit code.
pub fn telemetry(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        Some("probe") => {
            for p in probe_all() {
                println!(
                    "{:<13} {:<10} {}\n    {}\n    resolution: {}\n    accuracy: {}",
                    p.tag,
                    p.domain,
                    p.detail,
                    status(&p.status),
                    p.resolution,
                    p.accuracy
                );
            }
            println!("\ntracked, not implemented:");
            for t in TRACKED {
                println!("{:<13} {:<16} {} ({})", t.tag, t.domain, t.why, t.reference);
            }
            0
        }
        Some("setup") => {
            let user = std::env::var("USER").unwrap_or_default();
            if user.is_empty() || user == "root" {
                eprintln!(
                    "rce-ide telemetry setup: run it as the user who should get access, not root"
                );
                return 2;
            }
            let steps = plan(&probe_all(), &user);
            if steps.is_empty() {
                println!("nothing to set up: no source here needs a permission change");
                return 0;
            }
            let res = apply(&steps, &mut Terminal);
            let mut code = 0;
            for (t, o) in res {
                println!("{t}: {o:?}");
                if matches!(o, Outcome::Failed(_)) {
                    code = 1;
                }
            }
            code
        }
        _ => {
            eprint!("{}", usage());
            2
        }
    }
}
