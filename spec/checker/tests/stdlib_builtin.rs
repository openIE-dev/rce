//! The standard library is built in (4.1): `use` resolves to it with no
//! file and no directive, in check, run, trace and emit; the `#!lib NAME`
//! directive of the shipped binary is honoured too.
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
const LIB_CRYPTO: &str = "module demo2\n#!lib crypto\n\nfn main()\n    emit sha256([97]).len()\n";

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
fn the_lib_directive_places_the_library_in_front() {
    let f = tmp("lib_crypto", LIB_CRYPTO);
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 0, "{out}");
    let (st, out) = rce(&["run", &f]);
    assert_eq!(st, 0, "{out}");
    assert!(out.contains("emit 32"), "{out}");
}

#[test]
fn an_unknown_module_is_still_refused() {
    let f = tmp("unknown", "module demo3\nuse nothere\n\nfn main()\n    emit 1\n");
    let (st, out) = rce(&["check", &f]);
    assert_eq!(st, 1, "{out}");
    assert!(out.contains("W-NAME-1 unknown module `nothere`"), "{out}");
}
