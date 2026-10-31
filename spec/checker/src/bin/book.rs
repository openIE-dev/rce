//! rce-book: build and check the spec book, verify read receipts (see rce_spec_check::book).
use rce_spec_check::args::{Command, Fate, Removed, Word, read_or_exit};
use rce_spec_check::book::{Book, build, verify};
use std::path::{Path, PathBuf};
use std::process::exit;

/// The commands of rce-book (PC-8).
const TABLE: &[Command] = &[
    Command {
        path: &["build"],
        words: &[Word::mode("check")],
        operands: "[SPEC_DIR]",
        min: 0,
        max: 1,
        help: "write the chapter footers, INDEX.md and RCE-SPEC.md (check: write nothing, exit 1 if any differs)",
    },
    Command { path: &["verify"], words: &[], operands: "RECEIPT.json [SPEC_DIR]", min: 1, max: 2, help: "verify a read receipt" },
];
const REMOVED: &[Removed] = &[Removed { flag: "--check", paths: &["build"], values: 0, fate: Fate::Word }];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let default_spec = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let book = |d: Option<&String>| {
        let spec = d.map(PathBuf::from).unwrap_or(default_spec.clone());
        Book {
            here: spec.join("book"),
            spec,
        }
    };
    let (c, p) = read_or_exit("rce-book", TABLE, REMOVED, &args);
    let o = &p.operands;
    let code = match c.path {
        ["build"] => build(&book(o.first()), p.has("check")),
        _ => verify(&book(o.get(1)), &o[0]),
    };
    exit(code);
}
