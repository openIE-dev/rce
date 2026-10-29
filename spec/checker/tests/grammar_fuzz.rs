//! Grammar fuzzing with fixed seeds, no new dependency.
//!
//! Inputs are mutants of every `.rce` file under spec/: tokens are deleted,
//! duplicated, swapped, replaced or inserted from a pool of words, punctuation,
//! literals and invisible characters, and a text is sometimes cut short. A run
//! is a pure function of the seed: `RCE_FUZZ_SEED` (default below) and
//! `RCE_FUZZ_N` (inputs) change it, nothing else does.
//!
//! * `checker_never_panics_on_fuzzed_text`: lexing, parsing and checking a
//!   mutant returns `Ok` or `Err`; it never panics, aborts or overflows the
//!   stack. A failing input is printed with its seed and index.
//! * `tree_sitter_agrees_with_rust_parser_on_fuzz`: the tree-sitter CLI
//!   accepts a mutant exactly when the Rust parser does (same rule as
//!   tree_sitter_sync). It needs the CLI: skipped when it is missing, a
//!   failure under CI. Its name keeps it out of tests-fast (scripts/ci-tests.sh
//!   skips tree_sitter_agrees_with_rust_parser); the tree-sitter sync workflow runs it.
use std::path::{Path, PathBuf};
use std::process::Command;

const SEED: u64 = 0x5eed_0917_c0de_0001;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const POOL: &[&str] = &[
    "fn", "let", "var", "type", "enum", "struct", "if", "else", "match", "loop", "while", "bound", "return", "fail",
    "scope", "spawn", "select", "use", "pub", "mut", "ref", "as", "in", "true", "false", "ct", "label", "release",
    "requires", "ensures", "ghost", "decreases", "handler", "state", "on", "emit", "par", "template", "world",
    "{", "}", "(", ")", "[", "]", "<", ">", ",", ";", ":", "::", ".", "..", "->", "=>", "=", "==", "!=", "+", "-", "*",
    "/", "%", "&", "&&", "||", "!", "?", "@", "#", "|", "_", "0", "1", "-1", "9223372036854775808", "0x", "1e",
    "1.5", "\"", "\"a\"", "\"\\u{41}\"", "\"\\q\"", "'", "// c", "/*", "*/", "\u{200b}", "\u{202e}", "\u{feff}", "\t",
    "\r", "\u{e9}", "\u{1f600}", "x", "y", "_x", "# rce-version: 9.9\n",
];

fn rce_files(d: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(d).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            if !p.ends_with("target") && !p.ends_with("node_modules") {
                rce_files(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "rce") {
            out.push(p);
        }
    }
}

fn corpus() -> Vec<String> {
    let spec = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let mut files = vec![];
    rce_files(&spec, &mut files);
    files.sort();
    let c: Vec<String> = files.iter().filter_map(|f| std::fs::read_to_string(f).ok()).collect();
    assert!(c.len() > 150, "{}", c.len());
    c
}

/// Split into tokens: word runs, whitespace runs, single other characters.
fn pieces(s: &str) -> Vec<&str> {
    let mut v = vec![];
    let mut start = 0;
    let mut prev: Option<u8> = None;
    for (i, c) in s.char_indices() {
        let k = if c.is_alphanumeric() || c == '_' {
            0
        } else if c.is_whitespace() {
            1
        } else {
            2
        };
        if i > start && (Some(k) != prev.map(|p| p) || k == 2) {
            v.push(&s[start..i]);
            start = i;
        }
        prev = Some(k);
    }
    if start < s.len() {
        v.push(&s[start..]);
    }
    v
}

fn mutant(rng: &mut Rng, corpus: &[String]) -> String {
    let src = &corpus[rng.below(corpus.len())];
    let mut p: Vec<String> = pieces(src).into_iter().map(String::from).collect();
    if p.is_empty() {
        return String::new();
    }
    for _ in 0..1 + rng.below(3) {
        let i = rng.below(p.len());
        match rng.below(7) {
            0 => {
                p.remove(i);
            }
            1 => {
                let d = p[i].clone();
                p.insert(i, d);
            }
            2 if i + 1 < p.len() => p.swap(i, i + 1),
            3 => p[i] = POOL[rng.below(POOL.len())].to_string(),
            4 => p.insert(i, format!(" {} ", POOL[rng.below(POOL.len())])),
            5 => p.truncate(i + 1),
            _ => {
                let j = rng.below(p.len());
                let d = p[j].clone();
                p.insert(i, d);
            }
        }
        if p.is_empty() {
            break;
        }
    }
    p.concat()
}

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn rust_accepts(src: &str) -> bool {
    match rce_spec_check::parse::lex(src) {
        Ok(t) => {
            let mut p = rce_spec_check::parse::Parser::new(t);
            p.module().is_ok() && rce_spec_check::v09::stray_words(&p.t, &p.kw_used).is_ok()
        }
        _ => false,
    }
}

#[test]
fn checker_never_panics_on_fuzzed_text() {
    let seed = env_u64("RCE_FUZZ_SEED", SEED);
    let n = env_u64("RCE_FUZZ_N", 3000);
    let corpus = corpus();
    let mut rng = Rng(seed);
    let inputs: Vec<String> = (0..n).map(|_| mutant(&mut rng, &corpus)).collect();
    // a big stack, so a deep recursion is reported as a failure of the
    // property and not as a crash of the test process
    let h = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            let mut bad = vec![];
            let mut accepted = 0;
            for (i, s) in inputs.iter().enumerate() {
                let r = std::panic::catch_unwind(|| rce_spec_check::check_one(s));
                match r {
                    Ok(Ok(())) => accepted += 1,
                    Ok(Err(_)) => {}
                    Err(_) => bad.push((i, s.clone())),
                }
            }
            (accepted, bad)
        })
        .unwrap();
    let (accepted, bad) = h.join().expect("the fuzz thread itself died");
    println!("seed {seed:#x}: {n} inputs, {accepted} admitted, {} panics", bad.len());
    assert!(
        bad.is_empty(),
        "{} inputs panic the checker (seed {seed:#x}); first #{}:\n{}",
        bad.len(),
        bad[0].0,
        bad[0].1
    );
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3))
}

/// Known disagreements, one `hash  note` per line (tests/fuzz_known_divergences.txt).
/// The mutants are per file (seeded by the file's path), so adding a file adds
/// mutants and leaves the others as they were. A disagreement not listed fails;
/// a listed one that no longer disagrees is only reported, so a fix is never
/// blocked, and the line is then deleted.
fn known() -> std::collections::HashSet<u64> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fuzz_known_divergences.txt");
    std::fs::read_to_string(p)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| u64::from_str_radix(l.split_whitespace().next()?, 16).ok())
        .collect()
}

#[test]
fn tree_sitter_agrees_with_rust_parser_on_fuzz() {
    let spec = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf();
    let ts = spec.parent().unwrap().join("tree-sitter");
    if Command::new("tree-sitter").arg("--version").output().is_err() {
        assert!(std::env::var_os("CI").is_none(), "tree-sitter CLI required under CI");
        println!("tree-sitter CLI not installed: skipped");
        return;
    }
    let seed = env_u64("RCE_FUZZ_SEED", SEED);
    let per_file = env_u64("RCE_FUZZ_TS_K", 1);
    let mut files = vec![];
    rce_files(&spec, &mut files);
    files.sort();
    let known = known();
    let dir = std::env::temp_dir().join(format!("rce-fuzz-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (mut bad, mut seen_known, mut n) = (vec![], 0, 0);
    for p in &files {
        let Ok(text) = std::fs::read_to_string(p) else { continue };
        let one = [text];
        let rel = p.strip_prefix(&spec).unwrap().to_string_lossy().replace('\\', "/");
        let mut rng = Rng(seed ^ fnv(&rel));
        for k in 0..per_file {
            let s = mutant(&mut rng, &one);
            // a literal outside every type range is W-LEX-3, a well-formedness rule (section 7),
            // not syntax: the grammar cannot judge a value, so such mutants are skipped
            if matches!(rce_spec_check::parse::lex(&s), Err(e) if e.contains("out of range")) {
                continue;
            }
            n += 1;
            let rust_ok = std::panic::catch_unwind(|| rust_accepts(&s)).unwrap_or(false);
            let f = dir.join("m.rce");
            std::fs::write(&f, &s).unwrap();
            let ts_ok = Command::new("tree-sitter")
                .current_dir(&ts)
                .args(["parse", "-q"])
                .arg(&f)
                .output()
                .unwrap()
                .status
                .success();
            if rust_ok != ts_ok {
                if known.contains(&fnv(&s)) {
                    seen_known += 1;
                } else {
                    bad.push(format!("{:016x}  {rel} #{k}: rust {rust_ok}, tree-sitter {ts_ok}\n--- input ---\n{s}\n---", fnv(&s)));
                }
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
    println!("seed {seed:#x}: {n} mutants, {seen_known} known disagreements, {} new", bad.len());
    assert!(
        bad.is_empty(),
        "{} new disagreements (seed {seed:#x}); first {}:\n{}\nAdd a line `hash  note` to tests/fuzz_known_divergences.txt only for a triaged, filed divergence.",
        bad.len(),
        bad.len().min(3),
        bad[..bad.len().min(3)].join("\n")
    );
}

/// Found by the fuzzer (seed 7, input 7601): `Box()` indexed an empty argument list.
#[test]
fn box_without_its_value_is_kept_not_a_panic() {
    let src = "kind T\n    Leaf\n    Node(k: Box[T])\n\nfn f() -> i64\n    let c = Node(k: Box())\n    return 0\n";
    let r = std::panic::catch_unwind(|| rce_spec_check::check_one(src)).expect("the checker panicked");
    let e = r.unwrap_err();
    assert!(e.contains("W-TY-1") && e.contains("`Box` takes one value"), "{e}");
}
