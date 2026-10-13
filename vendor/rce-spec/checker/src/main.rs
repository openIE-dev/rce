//! rce-check: parse and check RCE modules.
//! Usage: rce-check [--parse-only] FILE...   (all FILEs form one program: `use` resolves among them)
use std::collections::HashSet;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let parse_only = args.first().is_some_and(|a| a == "--parse-only");
    if parse_only {
        args.remove(0);
    }
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
                println!("REFUSED {p}: {e}")
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
            println!("REFUSED {p}: {}", c.errs.join("; "));
        }
    }
    std::process::exit(if bad > 0 { 1 } else { 0 });
}
