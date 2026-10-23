//! Draft 0.6 syntax is gated by the module's language version (spec 2.7):
//! `# rce-version: 0.6` on the first line admits it in every build; a 0.5
//! module (undeclared in the default build, or declared 0.5 in any build)
//! reads the draft 0.6 words as ordinary names, so a valid 0.5 program is
//! never refused because of them. The `conc06` feature only changes the
//! default for undeclared modules. Version 0.7 has the same syntax as
//! 0.6 (C.13).
use rce_spec_check::parse::parse;

const PAR: &str = "fn main()\n    var a: i64 = 0\n    var b: i64 = 0\n    par\n        branch one\n            a = 1\n        branch two\n            b = 2\n    emit a + b\n";

#[test]
fn par_needs_version_06() {
    let r = parse(PAR);
    if cfg!(feature = "conc06") {
        assert!(r.is_ok(), "{r:?}");
    } else {
        assert!(r.is_err(), "undeclared module is 0.5: `par` is a name");
    }
    let r = parse(&format!("# rce-version: 0.6\n{PAR}"));
    assert!(r.is_ok(), "{r:?}");
    assert!(r.unwrap().draft06);
    let r = parse(&format!("# rce-version: 0.5\n{PAR}"));
    assert!(r.is_err(), "declared 0.5 never admits par");
}

/// Version 0.7 (C.13) has the 0.6 syntax and rules; only its method
/// costs differ, so the AST records the version.
#[test]
fn version_07_has_the_06_syntax() {
    let m = parse(&format!("# rce-version: 0.7\n{PAR}")).unwrap();
    assert!(m.draft06 && m.v07);
    let m = parse(&format!("# rce-version: 0.6\n{PAR}")).unwrap();
    assert!(m.draft06 && !m.v07);
    assert!(parse(&format!("# rce-version: 0.7\n{NAMES05}")).is_err(), "in a 0.7 module `node` is reserved");
    assert_eq!(rce_spec_check::check_one(&format!("# rce-version: 0.7\n{PAR}")), Ok(()));
}

/// The keyword-collision bug: 0.6 words used as names in a valid 0.5 program.
const NAMES05: &str = "fn f(node: i64, scope: i64, group: i64) -> i64\n    let child = node + scope\n    let par = child + group\n    return par\n\ncheck f(1, 2, 3) == 6\n";

#[test]
fn draft06_words_are_names_in_05() {
    let src = format!("# rce-version: 0.5\n{NAMES05}");
    assert_eq!(rce_spec_check::check_one(&src), Ok(()));
    if !cfg!(feature = "conc06") {
        assert_eq!(rce_spec_check::check_one(NAMES05), Ok(()));
    }
    let r = parse(&format!("# rce-version: 0.6\n{NAMES05}"));
    assert!(r.is_err(), "in a 0.6 module `node` is reserved");
}

#[test]
fn unknown_version_is_w_ver_1() {
    let e = parse("# rce-version: 0.9\nfn main()\n    pass\n").unwrap_err();
    assert!(e.contains("W-VER-1"), "{e}");
}

/// W-SCOPE-1 (structured spawn) is a 0.6 rule; a 0.5 task function may spawn.
const SPAWN05: &str = "fn worker(c: chan i64)\n    c <- 1\n\nfn boss(c: chan i64)\n    spawn worker(c)\n\nfn main()\n    let c = chan[i64](1)\n    spawn boss(c)\n";

#[test]
fn spawn_in_task_is_05() {
    let src = format!("# rce-version: 0.5\n{SPAWN05}");
    let r = rce_spec_check::check_one(&src);
    assert_eq!(r, Ok(()));
}

/// The tree-sitter port reserves the draft 0.6 words only in a module that
/// declares version 0.6 (scanner.c), as the Rust parser does.
#[cfg(feature = "tree-sitter")]
#[test]
fn tree_sitter_follows_the_version() {
    let mut ts = rce_spec_check::ts::TsParser::new();
    assert!(ts.accepts(NAMES05), "0.5: draft 0.6 words are names");
    assert!(ts.accepts(&format!("# rce-version: 0.5\n{NAMES05}")));
    assert!(!ts.accepts(&format!("# rce-version: 0.6\n{NAMES05}")));
    assert!(ts.accepts(&format!("# rce-version: 0.6\n{PAR}")));
    assert!(ts.accepts(&format!("# rce-version: 0.7\n{PAR}")));
    assert!(!ts.accepts(&format!("# rce-version: 0.7\n{NAMES05}")));
    assert!(!ts.accepts(&format!("fn f() -> i64 # rce-version: 0.6\n    return 1\n\n{PAR}")));
}
