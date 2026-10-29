//! One language, one globally reserved word list (spec 2.7). The concurrency
//! words (`par`, `node`, `scope`, `group`, ...) are reserved in every module,
//! whatever its `# rce-version:` label says (a label gates nothing), so a
//! program that uses one as a name requires change, and a program that uses one as
//! its construct is accepted.
use rce_spec_check::parse::parse;

const PAR: &str = "fn main()\n    var a: i64 = 0\n    var b: i64 = 0\n    par\n        branch one\n            a = 1\n        branch two\n            b = 2\n    emit a + b\n";

/// A program that uses the reserved words as names.
const NAMES: &str = "fn f(node: i64, scope: i64, group: i64) -> i64\n    let child = node + scope\n    let par = child + group\n    return par\n\ncheck f(1, 2, 3) == 6\n";

#[test]
fn par_is_a_construct_in_every_module() {
    for label in ["", "# rce-version: 0.5\n", "# rce-version: 0.6\n", "# rce-version: 0.9\n"] {
        let r = parse(&format!("{label}{PAR}"));
        assert!(r.is_ok(), "{label:?}: {r:?}");
        assert_eq!(rce_spec_check::check_one(&format!("{label}{PAR}")), Ok(()), "{label:?}");
    }
}

#[test]
fn reserved_words_are_not_names_in_any_module() {
    for label in ["", "# rce-version: 0.5\n", "# rce-version: 0.6\n", "# rce-version: 0.9\n"] {
        let r = parse(&format!("{label}{NAMES}"));
        assert!(r.is_err(), "{label:?}: `node` is reserved");
    }
}

/// The label is ignored: any text is tolerated.
#[test]
fn an_unknown_label_is_tolerated() {
    let r = parse("# rce-version: 0.10\nfn main()\n    pass\n");
    assert!(r.is_ok(), "{r:?}");
}

/// W-SCOPE-1 (structured spawn) holds in every module.
const SPAWN: &str = "fn worker(c: chan i64)\n    c <- 1\n\nfn boss(c: chan i64)\n    spawn worker(c)\n\nfn main()\n    let c = chan[i64](1)\n    spawn boss(c)\n";

#[test]
fn spawn_outside_a_scope_requires_change_in_every_module() {
    let r = rce_spec_check::check_one(&format!("# rce-version: 0.5\n{SPAWN}"));
    assert!(r.unwrap_err().contains("W-SCOPE-1"));
}

/// The tree-sitter port reserves the same words in every module.
#[cfg(feature = "tree-sitter")]
#[test]
fn tree_sitter_reserves_the_same_words() {
    let mut ts = rce_spec_check::ts::TsParser::new();
    assert!(!ts.accepts(NAMES));
    assert!(!ts.accepts(&format!("# rce-version: 0.5\n{NAMES}")));
    assert!(ts.accepts(PAR));
    assert!(ts.accepts(&format!("# rce-version: 0.7\n{PAR}")));
}
