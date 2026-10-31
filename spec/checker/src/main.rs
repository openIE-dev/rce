//! rce-check: parse and check RCE modules.
//! Usage: rce-check [parse-only] FILE...   (all FILEs form one program: `use` resolves among them)
use rce_spec_check::args::{Command, Fate, MANY, Removed, Word};
use std::collections::HashSet;

/// The command line of rce-check (PC-8).
const TABLE: &[Command] = &[Command {
    path: &[],
    words: &[Word::mode("parse-only")],
    operands: "FILE...",
    min: 1,
    max: MANY,
    help: "parse and check; all FILEs form one program (parse-only: parse alone)",
}];
const REMOVED: &[Removed] = &[Removed { flag: "--parse-only", paths: &[], values: 0, fate: Fate::Word }];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (_, p) = rce_spec_check::args::read_or_exit("rce-check", TABLE, REMOVED, &args);
    let parse_only = p.has("parse-only");
    let args = p.operands;
    let mut parsed = vec![];
    let mut bad = 0;
    for p in &args {
        let src = std::fs::read_to_string(p).unwrap_or_else(|e| {
            eprintln!("{p}: {e}");
            std::process::exit(2)
        });
        match rce_spec_check::parse::parse(&src) {
            Ok(m) => parsed.push((p.clone(), m)),
            Err(e) => {
                bad += 1;
                println!("requires change {p}: {e}")
            }
        }
    }
    let known: HashSet<String> = parsed
        .iter()
        .filter_map(|(_, m)| m.path.as_ref().map(|q| q.join(".")))
        .collect();
    for (p, m) in &parsed {
        if parse_only {
            println!("ok {p}");
            continue;
        }
        let mut c = rce_spec_check::check::Checker::new();
        c.check_module(m, &known);
        if c.errs.is_empty() {
            c.errs = rce_spec_check::interp::admit(m);
        }
        if c.errs.is_empty() {
            println!("ok {p}");
        } else {
            bad += 1;
            println!("requires change {p}: {}", c.errs.join("; "));
        }
    }
    std::process::exit(if bad > 0 { 1 } else { 0 });
}
