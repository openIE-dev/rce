//! F.4.3: definition hashes and `rce api-diff`. Properties (T50 is stated and
//! unproved; these are tests, not a proof): a hash does not depend on
//! layout, comments or declaration order, nor on the names of items (a
//! rename applied everywhere); it changes with the body, with a callee, with
//! the version; the api-diff table rows are as F.4.3 states.
use rce_spec_check::apidiff::{defs, diff};
use rce_spec_check::parse::parse;

const BASE: &str = "# rce-version: 0.9\npub fn helper(a: i64) -> i64\n    return a + 1\n\npub fn top(a: i64) -> i64\n    return helper(a) * 2\n";

fn h(src: &str, name: &str) -> String {
    defs(&parse(src).unwrap()).into_iter().find(|d| d.name == name).unwrap().hash
}

#[test]
fn hash_ignores_layout_comments_and_order() {
    let reordered = "# rce-version: 0.9\n# a comment\npub fn top(a: i64) -> i64\n    # inner\n    return   helper( a )   *  2\n\n\npub fn helper(a: i64) -> i64\n    return a + 1\n";
    assert_eq!(h(BASE, "top"), h(reordered, "top"));
    assert_eq!(h(BASE, "helper"), h(reordered, "helper"));
}

#[test]
fn hash_ignores_item_names_when_renamed_everywhere() {
    let renamed = "# rce-version: 0.9\npub fn bump(a: i64) -> i64\n    return a + 1\n\npub fn outer(a: i64) -> i64\n    return bump(a) * 2\n";
    assert_eq!(h(BASE, "top"), h(renamed, "outer"));
    assert_eq!(h(BASE, "helper"), h(renamed, "bump"));
}

#[test]
fn hash_changes_with_body_and_callee() {
    let body = BASE.replace("a + 1", "a + 2");
    assert_ne!(h(BASE, "helper"), h(&body, "helper"));
    // the caller's text is unchanged, but its callee's hash changed
    assert_ne!(h(BASE, "top"), h(&body, "top"));
    // the release label is not hashed
    let relabelled = BASE.replace("0.9", "0.8");
    assert_eq!(h(BASE, "helper"), h(&relabelled, "helper"));
}

#[test]
fn mutually_recursive_group_hashes_are_name_free_and_stable() {
    let a = "# rce-version: 0.9\npub fn even(n: u32) -> bool\n    if n == 0\n        return true\n    return odd(n - 1)\n\npub fn odd(n: u32) -> bool\n    if n == 0\n        return false\n    return even(n - 1)\n";
    let b = a.replace("even", "e1").replace("odd", "o1");
    assert_eq!(h(a, "even"), h(&b, "e1"));
    assert_eq!(h(a, "odd"), h(&b, "o1"));
    assert_ne!(h(a, "even"), h(a, "odd"));
}

fn rows(old: &str, new: &str) -> Vec<String> {
    diff(&[parse(old).unwrap()], &[parse(new).unwrap()]).iter().map(|r| r.show()).collect()
}

#[test]
fn api_diff_table() {
    // body change under unchanged signature: compatible
    let r = rows(BASE, &BASE.replace("a + 1", "a + 2"));
    // `top` is listed too: its definition hash includes its callee's hash
    assert_eq!(r, vec!["compatible\tfn helper\tbody changed under an unchanged signature", "compatible\tfn top\tbody changed under an unchanged signature"], "{r:?}");
    // an export is added: compatible
    let r = rows(BASE, &format!("{BASE}\npub fn extra() -> i64\n    return 1\n"));
    assert_eq!(r, vec!["compatible\tfn extra\texport added"], "{r:?}");
    // an export is removed: breaking
    let r = rows(BASE, "# rce-version: 0.9\npub fn helper(a: i64) -> i64\n    return a + 1\n");
    assert_eq!(r, vec!["breaking\tfn top\texport removed or renamed"], "{r:?}");
    // a parameter changes from borrow to &mut, consume to borrow
    let old = "# rce-version: 0.9\npub fn f(a: &List[i64]) -> i64\n    return a.len() as i64\n";
    let new = "# rce-version: 0.9\npub fn f(a: &mut List[i64]) -> i64\n    return a.len() as i64\n";
    let r = rows(old, new);
    assert_eq!(r, vec!["breaking\tfn f\tparameter 1 changed from borrow to &mut"], "{r:?}");
    // a record field added, an enum variant added: breaking
    let old = "# rce-version: 0.9\npub struct P\n    x: i64\n\npub kind K\n    A\n    B\n";
    let new = "# rce-version: 0.9\npub struct P\n    x: i64\n    y: i64\n\npub kind K\n    A\n    B\n    C\n";
    let r = rows(old, new);
    assert_eq!(r, vec!["breaking\tkind K\tenum variant added: C", "breaking\tstruct P\trecord field added: y"], "{r:?}");
    // the kind of an export changes: breaking
    let old = "# rce-version: 0.9\npub fn f() -> i64\n    return 1\n";
    let new = "# rce-version: 0.9\npub const f: i64 = 1\n";
    let r = rows(old, new);
    assert_eq!(r, vec!["breaking\tconst f\tkind changed from fn to const"], "{r:?}");
    // a signature change that is not a mode change: breaking
    let r = rows(BASE, &BASE.replace("fn helper(a: i64)", "fn helper(a: i64, b: i64)").replace("helper(a)", "helper(a, 1)"));
    assert!(r.iter().any(|x| x == "breaking\tfn helper\tsignature changed"), "{r:?}");
}

#[test]
fn api_diff_command_exit_status() {
    let d = std::env::temp_dir().join(format!("rce-apidiff-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let (a, b, c) = (d.join("a.rce"), d.join("b.rce"), d.join("c.rce"));
    std::fs::write(&a, BASE).unwrap();
    std::fs::write(&b, BASE.replace("a + 1", "a + 2")).unwrap();
    std::fs::write(&c, "# rce-version: 0.9\npub fn helper(a: i64) -> i64\n    return a + 1\n").unwrap();
    let run = |x: &std::path::Path, y: &std::path::Path| {
        let (mut o, mut e) = (vec![], vec![]);
        let code = rce_spec_check::cli::main(&["api-diff".into(), x.to_string_lossy().into_owned(), y.to_string_lossy().into_owned()], &mut &b""[..], &mut o, &mut e);
        (code, String::from_utf8(o).unwrap())
    };
    assert_eq!(run(&a, &b).0, 0);
    assert_eq!(run(&a, &c).0, 1);
    assert!(run(&a, &c).1.starts_with("breaking\tfn top"));
}
