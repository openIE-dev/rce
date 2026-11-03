//! W-UI-1 (spec 7, 7.1; Appendix E, E.4f, HC-29): a module that emits
//! frames is held to `ui.rules` at admission, with no `example` or `check`
//! line that names `ui.rules` (decision on Q3 of
//! `spec/proposals/human-interface.md`, 2026-11-03).
//!
//! * A module whose examples derive frames that meet WCAG 2.2 AA is
//!   admitted.
//! * The same module with `tabindex` removed from its button requires
//!   change under W-UI-1, one report per finding, naming the criterion.
//! * A module that emits no frame is not affected, whatever its views.
//! * Frames that cannot be judged are undecided and require change: no
//!   example derives a frame, or an emitted `Frame` is not a tree.
//! * `rce check`, `rce test`, the language server and the attention line
//!   see the same reports, through the one admission.
//! * The receipt lists W-UI-1 in its rule set and records, per example,
//!   the frames judged.
use rce_spec_check::admission::RuleSet;
use rce_spec_check::{check_one, cli, receipt};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::path::PathBuf;

fn rce(args: &[&str]) -> (i32, String, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let (mut out, mut err) = (vec![], vec![]);
    let code = cli::main(&a, &mut &b""[..], &mut out, &mut err);
    (code, String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
}

fn scratch(name: &str, text: &str) -> String {
    let d = std::env::temp_dir().join(format!("rce-ui-rules-admission-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p: PathBuf = d.join(name);
    std::fs::write(&p, text).unwrap();
    p.to_string_lossy().into_owned()
}

/// A door panel: a heading and one button that opens or closes the door,
/// emitted as a frame after every step when `frames` is true. `button` is
/// the button's extra attributes. No line names `ui.rules`.
fn door(frames: bool, button: &str, examples: &str) -> String {
    let (decl, emit) = if frames { ("struct Frame\n    view: View\n    ev: text\n\n", true) } else { ("", false) };
    let em = |open: &str, focused: &str, ev: &str| if emit { format!("    emit Frame(view: panel({open}, {focused}), ev: \"{ev}\")\n") } else { format!("    let _ = panel({open}, {focused})\n") };
    format!(
        r#"module door
use ui.view

intent "a door panel: one button opens and closes the door"
{examples}
{decl}state Door
    opened: bool = false
    focused: bool = false

event Open
event Close
event Focus

step Door on Open
    opened = true
{open}step Door on Close
    opened = false
{close}step Door on Focus
    focused = true
{focus}
# the parts no rule decides, attested by the person who reviewed them
fn attest(v: &mut View, ids: text)
    for i in ids.words()
        view.set_attr(v, 0, "attest-" + i, "A. Reviewer;2026-11-03")

fn panel(is_open: bool, focused: bool) -> View
    var v = view.view_new("div", "")
    for kv in [["lang", "en"], ["title", "Door"], ["fg", "#000000"], ["bg", "#ffffff"], ["font-px", "16"]]
        view.set_attr(&mut v, 0, copy(kv[0]), copy(kv[1]))
    attest(&mut v, "1.3.1 1.3.2 1.3.3 1.4.1 1.4.11 2.4.2 2.4.5 2.4.6 3.1.1 3.1.2 3.3.4")
    let m = view.add_el(&mut v, 0, "main", "")
    view.set_attr(&mut v, m, "role", "main")
    let h = view.add_el(&mut v, m, "h1", "")
    view.set_attr(&mut v, h, "role", "heading")
    view.set_attr(&mut v, h, "aria-level", "1")
    let _ = view.add_text(&mut v, h, "Door")
    let b = view.add_el(&mut v, m, "button", "toggle")
    for kv in [["role", "button"]{button}, ["w", "44"], ["h", "44"], ["border", "#000000"], ["focus-outline", "#005fcc"]]
        view.set_attr(&mut v, b, copy(kv[0]), copy(kv[1]))
    view.set_attr(&mut v, b, "action", if is_open then "Close" else "Open")
    if focused
        view.set_attr(&mut v, b, "focused", "true")
    let _ = view.add_text(&mut v, b, if is_open then "Close" else "Open")
    return v
"#,
        open = em("true", "focused", "activate toggle"),
        close = em("false", "focused", "activate toggle"),
        focus = em("opened", "true", "focus toggle"),
    )
}

const EXAMPLES: &str = "    example Open, Focus -> opened\n    example Open, Focus, Close -> opened == false\n";
const TAB: &str = r#", ["tabindex", "0"]"#;

fn errs(src: &str) -> Vec<String> {
    match check_one(src) {
        Ok(()) => vec![],
        Err(e) => e.split("; line ").enumerate().map(|(i, x)| if i == 0 { x.to_string() } else { format!("line {x}") }).collect(),
    }
}

fn ui(errs: &[String]) -> Vec<&String> {
    errs.iter().filter(|e| e.contains("W-UI-1")).collect()
}

#[test]
fn a_compliant_module_is_admitted() {
    let src = door(true, TAB, EXAMPLES);
    assert!(!src.contains("rules"), "no line names ui.rules");
    assert_eq!(check_one(&src), Ok(()));
    // the fixture of HC-29 is the same panel, emitted with its own words
    let fx = std::fs::read_to_string("tests/host/frames.rce").unwrap();
    assert_eq!(check_one(&fx), Ok(()));
}

#[test]
fn a_button_without_tabindex_requires_change_with_no_ui_rules_line() {
    let src = door(true, "", EXAMPLES);
    assert!(!src.contains("rules"), "no line names ui.rules");
    let es = errs(&src);
    let u = ui(&es);
    assert_eq!(u.len(), es.len(), "only W-UI-1 reports: {es:#?}");
    // one report per finding, each on its example's line, naming the criterion
    assert!(u.iter().all(|e| e.starts_with("line 5: W-UI-1 ") || e.starts_with("line 6: W-UI-1 ")), "{u:#?}");
    assert!(u.iter().all(|e| e.contains(": WCAG ")), "{u:#?}");
    let k = u.iter().filter(|e| e.contains("WCAG 2.1.1 fails")).count();
    // the two frames of the first example, the three of the second
    assert_eq!(k, 5, "2.1.1 in every frame: {u:#?}");
    assert!(u.iter().any(|e| e.starts_with("line 5: W-UI-1 frame 1 of the example (seq 0, activate toggle): WCAG 2.1.1 fails at node ")), "{u:#?}");
    assert!(u.iter().any(|e| e.starts_with("line 6: W-UI-1 frame 2 of the example (seq 1, focus toggle): WCAG 2.1.1 fails at node ")), "{u:#?}");
    // a criterion over the trace as a whole: no step can focus the button
    assert!(u.iter().any(|e| e.contains("WCAG 3.2.1")), "{u:#?}");
    // the inadmissible example of 7 says the same
    let ex = std::fs::read_to_string("../examples/change/admission/frame_no_tabindex.rce").unwrap();
    let es = errs(&ex);
    assert!(!es.is_empty() && es.iter().all(|e| e.contains("W-UI-1")), "{es:#?}");
    assert!(es.iter().any(|e| e.contains("WCAG 2.1.1 fails")), "{es:#?}");
}

#[test]
fn a_module_that_emits_no_frame_is_not_affected() {
    // the same views, built and not emitted: no frame, no W-UI-1
    assert_eq!(check_one(&door(false, "", EXAMPLES)), Ok(()));
    assert_eq!(check_one(&door(false, "", "    example Open -> opened\n")), Ok(()));
    // a library struct named Frame with no view, and no step, is not a frame
    assert_eq!(check_one("module lib\n\npub struct Frame\n    op: text\n\npub fn op(f: &Frame) -> text = copy(f.op)\n\ncheck op(&Frame(op: \"a\")) == \"a\"\n"), Ok(()));
}

#[test]
fn a_module_whose_examples_derive_no_frame_is_undecided() {
    // no example at all: the frames are undecided, reported at struct Frame
    let src = door(true, TAB, "    check 1 + 1 == 2\n");
    let es = errs(&src);
    assert_eq!(es.len(), 1, "{es:#?}");
    assert!(es[0].starts_with("line 7: W-UI-1 the module emits Frame and no example derives a frame, so its frames are undecided"), "{es:#?}");
    // an example whose steps emit no frame derives none either
    let src = door(true, TAB, "    example Open -> opened\n").replace("    emit Frame(view: panel(true, focused), ev: \"activate toggle\")\n", "    let _ = panel(true, focused)\n");
    let es = errs(&src);
    assert!(es.len() == 1 && es[0].contains("no example derives a frame"), "{es:#?}");
}

#[test]
fn a_frame_that_is_not_one_is_undecided() {
    // node 0 is its own child: not a tree, so not a frame (HC-29)
    let src = door(true, TAB, "    example Open -> opened\n").replace("    return v\n", "    if is_open\n        v.kids[0].push(0)\n    return v\n");
    let es = errs(&src);
    assert_eq!(es.len(), 1, "{es:#?}");
    assert!(es[0].starts_with("line 5: W-UI-1 the example's frames are undecided: seq 0: Frame is not delivered: "), "{es:#?}");
}

#[test]
fn every_tool_reports_the_same_findings() {
    let src = door(true, "", EXAMPLES);
    let es = errs(&src);
    let f = scratch("door.rce", &src);
    // rce check: one requires-change line, every report counted once
    let (c, out, err) = rce(&["check", &f]);
    assert_eq!(c, 1);
    assert!(out.starts_with(&format!("requires change {f}: ")), "{out}");
    for e in &es {
        assert!(out.contains(e.as_str()), "{e}\n{out}");
    }
    assert_eq!(err.trim_end(), format!("attention: {} requires change, 0 recommended, 0 altered, 0 observed", es.len()));
    // rce test
    let (c, out, _) = rce(&["test", &f]);
    assert_eq!(c, 1);
    assert!(out.contains("FAIL") && out.contains("W-UI-1"), "{out}");
    // the language server's diagnostics
    let ds = cli::diagnostics(&src, &HashSet::new());
    let lines: Vec<&String> = ds.iter().map(|(_, m)| m).collect();
    assert_eq!(lines, es.iter().collect::<Vec<_>>());
    assert!(ds.iter().all(|(l, _)| *l == 5 || *l == 6), "{ds:?}");
    // rce check write: no receipt for a program that requires change
    let r = scratch("door.receipt", "");
    let (c, out, _) = rce(&["check", "write", &r, &f]);
    assert_eq!(c, 1);
    assert!(out.contains(&format!("no receipt {r}: the program requires change")), "{out}");
}

#[test]
fn the_receipt_records_the_rule_and_the_frames() {
    let rules = RuleSet::in_force();
    assert_eq!(rules.rows["W-UI-1"].text(), "requires change");
    let src = door(true, TAB, EXAMPLES);
    let r = receipt::write(rules, &[("door.rce".into(), src.clone())]).unwrap();
    let v: Value = serde_json::from_str(&r).unwrap();
    assert!(v["rules"]["table"].as_array().unwrap().contains(&json!(["W-UI-1", "requires change"])));
    let mut frames: Vec<u64> = v["obligations"].as_array().unwrap().iter().filter_map(|o| o["frames"].as_u64()).collect();
    frames.sort();
    assert_eq!(frames, vec![2, 3], "{v:#}");
    // the receipt holds when revalidated, through the command line too
    let f = scratch("door-ok.rce", &src);
    let p = scratch("door-ok.receipt", "");
    let (c, out, _) = rce(&["check", "write", &p, &f]);
    assert_eq!(c, 0, "{out}");
    let (c, out, _) = rce(&["check", "receipt", &p, &f]);
    assert_eq!(c, 0, "{out}");
    assert!(out.contains("receipt holds"), "{out}");
    // a module that emits no frame records none
    let r = receipt::write(rules, &[("d.rce".into(), door(false, TAB, EXAMPLES))]).unwrap();
    let v: Value = serde_json::from_str(&r).unwrap();
    assert!(v["obligations"].as_array().unwrap().iter().all(|o| o.get("frames").is_none()), "{v:#}");
}
