//! rce-spec-check: the machine check of the RCE grammar and checker
//! (RCE-SPEC.md sections 2 to 5). Rust port of the former check.py.
//!
//! 1. Regenerates the tree-sitter parser from grammar.js with the
//!    tree-sitter CLI (its built-in QuickJS runtime, no Node.js) and fails
//!    on any conflict: grammar.js declares `conflicts: []`, so every LR(1)
//!    conflict not settled by a declared precedence is an error. The
//!    regenerated parser.c must equal the committed one byte for byte.
//! 2. Builds the LALR(1) tables of the grammar's single source, the EBNF of
//!    spec section 3 (book/03-syntax.md), and fails on any conflict (this
//!    replaces the former Lark check of rce.lark).
//!    Parses every examples/*.rce (must succeed) and examples/reject/*.rce
//!    (must fail) with three parsers: the Rust recursive-descent parser, the
//!    LALR(1) tables on the reference lexer's tokens, and the tree-sitter
//!    port. All must agree on every file.
//! 3. Runs the reference checker on all examples as one program (must be
//!    accepted), on examples/reject/ (must be refused), and on
//!    examples/ill_typed/ and examples/inadmissible/ (must parse in both
//!    parsers, then be refused by the type, well-formedness or admission
//!    rules).
//! 4. Checks that the spec book (footers, INDEX.md) and RCE-SPEC.md are in
//!    sync (rce-book build --check).
//!
//! Usage: rce-spec-check [--no-generate] [SPEC_DIR]
//! (SPEC_DIR defaults to the spec/ directory this crate lives in.)
use rce_spec_check::{check_program, parse};

/// The tree-sitter port as a second parser, when built with
/// `--features tree-sitter`. Without it, `accepts` is `None` and only the
/// Rust parser and LALR(1) are compared.
#[cfg(feature = "tree-sitter")]
struct Ts(rce_spec_check::ts::TsParser);
#[cfg(feature = "tree-sitter")]
impl Ts {
    const ON: bool = true;
    fn new() -> Self {
        Ts(rce_spec_check::ts::TsParser::new())
    }
    fn accepts(&mut self, s: &str) -> Option<bool> {
        Some(self.0.accepts(s))
    }
}
#[cfg(not(feature = "tree-sitter"))]
struct Ts;
#[cfg(not(feature = "tree-sitter"))]
impl Ts {
    const ON: bool = false;
    fn new() -> Self {
        Ts
    }
    fn accepts(&mut self, _: &str) -> Option<bool> {
        None
    }
}
use std::path::{Path, PathBuf};
use std::process::Command;

fn rce_files(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .map(|r| {
            r.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "rce"))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn load(ps: &[PathBuf]) -> Vec<(String, String)> {
    ps.iter()
        .map(|p| (p.display().to_string(), std::fs::read_to_string(p).unwrap()))
        .collect()
}

fn name(p: &str) -> &str {
    Path::new(p)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(p)
}

/// Regenerate the tree-sitter parser in a scratch copy; Err on a conflict
/// or when the committed parser.c is stale. Ok(number of grammar rules).
fn generate(spec: &Path) -> Result<usize, String> {
    let src = spec.join("tree-sitter");
    let tmp = std::env::temp_dir().join(format!("rce-ts-gen-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(tmp.join("src")).map_err(|e| e.to_string())?;
    std::fs::copy(src.join("grammar.js"), tmp.join("grammar.js")).map_err(|e| e.to_string())?;
    std::fs::copy(src.join("src/scanner.c"), tmp.join("src/scanner.c"))
        .map_err(|e| e.to_string())?;
    let out = Command::new(std::env::var("TREE_SITTER").unwrap_or_else(|_| "tree-sitter".into()))
        .args(["generate", "--js-runtime", "native"])
        .current_dir(&tmp)
        .output()
        .map_err(|e| {
            format!(
                "tree-sitter CLI not runnable ({e}); install with `cargo install tree-sitter-cli`"
            )
        })?;
    if !out.status.success() {
        return Err(format!(
            "GRAMMAR CONFLICTS or errors:\n{}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let new = std::fs::read(tmp.join("src/parser.c")).map_err(|e| e.to_string())?;
    let old = std::fs::read(src.join("src/parser.c")).map_err(|e| e.to_string())?;
    if new != old {
        return Err("committed tree-sitter/src/parser.c is stale: run `tree-sitter generate --js-runtime native` in spec/tree-sitter".into());
    }
    let g: serde_json::Value =
        serde_json::from_slice(&std::fs::read(tmp.join("src/grammar.json")).unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(g["rules"].as_object().map_or(0, |m| m.len()))
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let no_gen = args.iter().any(|a| a == "--no-generate");
    args.retain(|a| a != "--no-generate");
    let spec = args
        .first()
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."));
    let mut bad = 0usize;
    if !Ts::ON {
        println!(
            "grammar: tree-sitter checks skipped (default pure-Rust build; run with --features tree-sitter)"
        );
    } else if no_gen {
        println!("grammar: generation check skipped (--no-generate)");
    } else {
        match generate(&spec) {
            Ok(n) => println!(
                "grammar: tree-sitter LR(1), 0 conflicts, {n} rules; committed parser.c is current"
            ),
            Err(e) => {
                println!("{e}");
                std::process::exit(1);
            }
        }
    }
    // LALR(1) from the grammar's single source, spec section 3.
    let lalr = match std::fs::read_to_string(spec.join("book/03-syntax.md"))
        .map_err(|e| e.to_string())
        .and_then(|md| rce_spec_check::lalr::book_grammar(&md))
    {
        Ok(g) => {
            let (rules, prods) = (g.source_rules, g.prods.len());
            let l = rce_spec_check::lalr::Lalr::build(g);
            println!(
                "grammar: LALR(1) from book/03-syntax.md: {rules} rules, {prods} productions, {} states, {} conflicts",
                l.states,
                l.conflicts.len()
            );
            for c in &l.conflicts {
                println!("FAIL (LALR conflict): {c}");
            }
            bad += l.conflicts.len();
            Some(l)
        }
        Err(e) => {
            println!("FAIL (section 3 grammar): {e}");
            bad += 1;
            None
        }
    };
    let lalr_ok = |s: &str| lalr.as_ref().is_some_and(|l| l.accepts(s));
    let ex = spec.join("examples");
    let ok = load(&rce_files(&ex));
    let rej = load(&rce_files(&ex.join("reject")));
    let ill = load(&rce_files(&ex.join("ill_typed")));
    let inad = load(&rce_files(&ex.join("inadmissible")));
    let mut ts = Ts::new();

    let mut gok = 0;
    for (p, s) in &ok {
        let (t, r, l) = (ts.accepts(s), parse::parse(s).is_ok(), lalr_ok(s));
        if t.unwrap_or(true) && r && l {
            gok += 1
        } else {
            bad += 1;
            println!(
                "FAIL (should parse): {} tree-sitter={t:?} rust={r} lalr={l}",
                name(p)
            )
        }
    }
    // Spec 2.6 rule 4: a file without a final line end is the same program.
    let mut gnofinal = 0;
    for (p, s) in &ok {
        let cut = s.trim_end_matches(['\n', '\r', ' ']);
        let (t, r) = (ts.accepts(cut), parse::parse(cut).is_ok() && lalr_ok(cut));
        if t.unwrap_or(true) && r {
            gnofinal += 1
        } else {
            bad += 1;
            println!(
                "FAIL (should parse without final newline): {} tree-sitter={t:?} rust={r}",
                name(p)
            )
        }
    }
    let mut grej = 0;
    for (p, s) in &rej {
        let (t, r, l) = (ts.accepts(s), parse::parse(s).is_ok(), lalr_ok(s));
        if !t.unwrap_or(false) && !r && !l {
            grej += 1
        } else {
            bad += 1;
            println!(
                "FAIL (should be refused): {} tree-sitter={t:?} rust={r} lalr={l}",
                name(p)
            )
        }
    }
    println!(
        "grammar: accepted {gok}/{} valid examples, refused {grej}/{} invalid ({} agree)",
        ok.len(),
        rej.len(),
        if Ts::ON {
            "tree-sitter, LALR(1) and Rust parser"
        } else {
            "LALR(1) and Rust parser"
        }
    );
    println!(
        "grammar: accepted {gnofinal}/{} valid examples with the final newline removed (spec 2.6 rule 4)",
        ok.len()
    );

    let r = check_program(&ok, false);
    let nok = r.iter().filter(|(_, v)| v.is_ok()).count();
    for (p, v) in &r {
        if let Err(e) = v {
            bad += 1;
            println!("FAIL (checker should accept): {}: {e}", name(p))
        }
    }
    let mut nrej = 0;
    for f in &rej {
        if check_program(std::slice::from_ref(f), false)[0].1.is_err() {
            nrej += 1
        } else {
            bad += 1;
            println!("FAIL (checker should refuse): {}", name(&f.0))
        }
    }
    let mut nill = 0;
    for f in ill.iter().chain(&inad) {
        if !ts.accepts(&f.1).unwrap_or(true) || parse::parse(&f.1).is_err() {
            bad += 1;
            println!("FAIL (ill_typed must parse): {}", name(&f.0));
            continue;
        }
        if check_program(std::slice::from_ref(f), false)[0].1.is_err() {
            nill += 1
        } else {
            bad += 1;
            println!("FAIL (checker should refuse): {}", name(&f.0))
        }
    }
    println!(
        "checker: accepted {nok}/{} examples, refused {nrej}/{} syntax rejects, refused {nill}/{} ill-typed or inadmissible ({} of them inadmissible)",
        ok.len(),
        rej.len(),
        ill.len() + inad.len(),
        inad.len()
    );
    // 4. The book and the single-file RCE-SPEC.md are in sync.
    bad += rce_spec_check::book::build(&rce_spec_check::book::Book::new(&spec), true) as usize;
    println!(
        "{}",
        if bad == 0 {
            "ALL OK".to_string()
        } else {
            format!("{bad} failures")
        }
    );
    std::process::exit(if bad > 0 { 1 } else { 0 });
}
