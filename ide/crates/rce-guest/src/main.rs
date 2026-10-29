//! `rce-guest check|test|run FILE [EVENTS]` inside WASI. Reads only through
//! the preopened directory. Exit codes follow spec 10.1: 0 ok, 1 refused, 2 usage.

use rce_spec_check::{cli, parse};
use std::collections::HashSet;

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let code = match a.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        [cmd @ ("check" | "test"), file] => match std::fs::read_to_string(file) {
            Err(e) => {
                eprintln!("{file}: {e}");
                2
            }
            Ok(src) => {
                let d = cli::diagnostics(&src, &HashSet::new());
                if d.is_empty() {
                    if *cmd == "test" {
                        let (e, c) = parse::parse(&src)
                            .map(|m| cli::count_tests(&m))
                            .unwrap_or((0, 0));
                        println!("ok {file} ({e} examples, {c} checks)");
                    } else {
                        println!("ok {file}");
                    }
                    0
                } else {
                    for (l, m) in d {
                        println!("{file}:{l}: {m}");
                    }
                    1
                }
            }
        },
        ["run", file, rest @ ..] => match std::fs::read_to_string(file) {
            Err(e) => {
                eprintln!("{file}: {e}");
                2
            }
            Ok(src) => match cli::run(&src, rest.first().copied()) {
                Ok(o) => {
                    for l in o.lines {
                        println!("{l}");
                    }
                    if o.kept { 1 } else { 0 }
                }
                Err(e) => {
                    println!("kept: {e}");
                    1
                }
            },
        },
        _ => {
            eprintln!("usage: rce-guest check|test FILE | run FILE [EVENTS]");
            2
        }
    };
    std::process::exit(code);
}
