//! rce-spec-data: write (or with the word check, compare) spec/data/spec-data.json,
//! the spec's tables as data (W2; see rce_spec_check::specdata).
use rce_spec_check::args::{Command, Fate, Removed, Word, read_or_exit};
use std::path::Path;

/// The command line of rce-spec-data (PC-8).
const TABLE: &[Command] = &[Command {
    path: &[],
    words: &[Word::mode("check")],
    operands: "",
    min: 0,
    max: 0,
    help: "write spec/data/spec-data.json and spec-tables.json (check: write nothing, exit 1 if either differs)",
}];
const REMOVED: &[Removed] = &[Removed { flag: "--check", paths: &[], values: 0, fate: Fate::Word }];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (_, p) = read_or_exit("rce-spec-data", TABLE, REMOVED, &args);
    let spec = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let out = spec.join("data/spec-data.json");
    let s = rce_spec_check::specdata::export(&spec);
    // the part the checker compiles in (specread)
    let out2 = spec.join("data/spec-tables.json");
    let t = rce_spec_check::specread::tables_of(&s);
    if p.has("check") {
        for (f, want) in [(&out, &s), (&out2, &t)] {
            let have = std::fs::read_to_string(f).unwrap_or_default();
            if &have != want {
                println!("FAIL: {} is stale; run rce-spec-data", f.display());
                std::process::exit(1);
            }
        }
        println!("spec-data: in sync ({} bytes, tables {} bytes)", s.len(), t.len());
    } else {
        std::fs::create_dir_all(out.parent().unwrap()).unwrap();
        std::fs::write(&out, &s).unwrap();
        std::fs::write(&out2, &t).unwrap();
        println!("wrote {} ({} bytes) and {} ({} bytes)", out.display(), s.len(), out2.display(), t.len());
    }
}
