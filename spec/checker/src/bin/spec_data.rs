//! rce-spec-data: write (or with --check, compare) spec/data/spec-data.json,
//! the spec's tables as data (W2; see rce_spec_check::specdata).
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let spec = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let out = spec.join("data/spec-data.json");
    let s = rce_spec_check::specdata::export(&spec);
    // the part the checker compiles in (specread)
    let out2 = spec.join("data/spec-tables.json");
    let t = rce_spec_check::specread::tables_of(&s);
    if args.iter().any(|a| a == "--check") {
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
