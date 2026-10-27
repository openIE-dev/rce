//! The standard library is built in (4.1): `use` resolves to it with no
//! file and no directive, in check, run, trace and emit. `use` is the one
//! way to bring a library in: a `#!lib NAME` line is a comment.
use rce_spec_check::cli;

fn rce(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut out = Vec::new();
    let mut err = Vec::new();
    let code = cli::main(&a, &mut std::io::empty(), &mut out, &mut err);
    (code, String::from_utf8(out).unwrap() + &String::from_utf8(err).unwrap())
}

fn tmp(name: &str, text: &str) -> String {
    let p = std::env::temp_dir().join(format!("rce_stdlib_{}_{name}.rce", std::process::id()));
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

const USE_ENC: &str = "module demo\nuse enc\n\nfn main()\n    let b = [104, 105]\n    emit enc.b64_encode(&b, false, true)\n";
const USE_CRYPTO: &str = "module demo2\nuse crypto\n\nfn main()\n    emit crypto.sha256([97]).len()\n";
const LIB_LINE: &str = "module demo4\n#!lib crypto\n\nfn main()\n    emit sha256([97]).len()\n";

#[test]
fn use_resolves_to_the_built_in_library_in_check_run_and_emit() {
    assert!(rce_spec_check::stdlib::has("enc") && rce_spec_check::stdlib::has("data.events"));
    assert!(rce_spec_check::stdlib::names().len() >= 46);
    let f = tmp("use_enc", USE_ENC);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, out) = rce(&["run", &f]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("emit \"aGk=\""), "{out}");
    let (st, out) = rce(&["emit", "python", &f]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("b64_encode"), "{out}");
}

#[test]
fn use_brings_in_crypto_and_its_names_are_qualified() {
    let f = tmp("use_crypto", USE_CRYPTO);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, out) = rce(&["run", &f]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("emit 32"), "{out}");
    // pq, hpke and mls import crypto themselves
    for m in ["pq", "hpke", "mls", "energy"] {
        let src = rce_spec_check::stdlib::source(m).unwrap();
        assert!(src.lines().any(|l| l == "use crypto"), "{m}");
    }
}

#[test]
fn a_lib_line_is_a_comment() {
    // the retired `#!lib crypto` places nothing in front of the module
    let f = tmp("lib_line", LIB_LINE);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 1, "{out}");
    assert!(out.contains("W-NAME-1") && out.contains("`sha256`"), "{out}");
}

#[test]
fn an_unknown_module_is_still_refused() {
    let f = tmp("unknown", "module demo3\nuse nothere\n\nfn main()\n    emit 1\n");
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 1, "{out}");
    assert!(out.contains("W-NAME-1 unknown module `nothere`"), "{out}");
}

#[test]
fn rce_lib_lists_the_modules_and_the_hash_that_pins_them() {
    let (st, out) = rce(&["lib"]);
    assert_eq!(st, 0, "{out}");
    assert!(out.lines().any(|l| l.starts_with("enc ")), "{out}");
    let last = out.lines().last().unwrap();
    assert!(last.contains("pinned by stdlib/lib/MANIFEST.json") && last.ends_with(&rce_spec_check::manifest::stdlib_hash()), "{last}");
}
