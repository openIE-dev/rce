//! rce-book: build and check the spec book, verify read receipts (see rce_spec_check::book).
use rce_spec_check::book::{Book, build, verify};
use std::path::{Path, PathBuf};
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default_spec = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let usage =
        "usage: rce-book build [--check] [SPEC_DIR] | rce-book verify RECEIPT.json [SPEC_DIR]";
    let book = |d: Option<&String>| {
        let spec = d.map(PathBuf::from).unwrap_or(default_spec.clone());
        Book {
            here: spec.join("book"),
            spec,
        }
    };
    let code = match args.first().map(String::as_str) {
        Some("build") => {
            let check = args.iter().any(|a| a == "--check");
            let dir = args[1..].iter().find(|a| *a != "--check");
            build(&book(dir), check)
        }
        Some("verify") if args.len() >= 2 => verify(&book(args.get(2)), &args[1]),
        _ => {
            println!("{usage}");
            2
        }
    };
    exit(code);
}
